from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from distill_utils import (
    ResidualCodebook,
    batched_encode_embeddings,
    batched_reconstruction_mse,
    collision_stats,
    load_item_embeddings,
    load_teacher_codebook,
    remap_assignments_by_cluster_weight,
    resolve_target_sizes,
    save_distilled_codebook,
    sid_from_codes,
    weighted_kmeans,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Distill a lightweight target codebook from target item embeddings and a frozen "
            "OpenOneRec residual tokenizer."
        ),
    )
    parser.add_argument("--teacher_model_path", required=True, help="Path to the teacher ResKmeans checkpoint.")
    parser.add_argument(
        "--emb_path",
        required=True,
        help="Target-domain embedding parquet with item_id/pid and embedding columns.",
    )
    parser.add_argument("--output_dir", required=True, help="Directory to save distilled artifacts.")
    parser.add_argument(
        "--target_codebook_sizes",
        default="256,256,256",
        help="Comma-separated per-layer target codebook sizes. A single value applies to all layers.",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=None,
        help="Number of residual layers to use. Defaults to all teacher layers.",
    )
    parser.add_argument(
        "--count_power",
        type=float,
        default=1.0,
        help="Raise teacher per-code usage counts to this power before teacher-centroid compression.",
    )
    parser.add_argument(
        "--teacher_init_n_init",
        type=int,
        default=8,
        help="Number of weighted k-means restarts when compressing teacher centroids into target-size initializers.",
    )
    parser.add_argument(
        "--teacher_init_max_iter",
        type=int,
        default=50,
        help="Maximum weighted k-means iterations for teacher-centroid compression.",
    )
    parser.add_argument(
        "--refine_max_iter",
        type=int,
        default=20,
        help="Maximum warm-start k-means iterations on target residual embeddings.",
    )
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size for teacher encoding.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device used for teacher encoding and optional later inference.",
    )
    return parser.parse_args()


def build_transport_rows(
    layer_index: int,
    teacher_centroids: np.ndarray,
    target_centroids: np.ndarray,
    usage_counts: np.ndarray,
) -> list[dict[str, Any]]:
    teacher_norm = np.sum(teacher_centroids**2, axis=1, keepdims=True)
    target_norm = np.sum(target_centroids**2, axis=1, keepdims=True).T
    distances = np.maximum(teacher_norm + target_norm - 2.0 * teacher_centroids @ target_centroids.T, 0.0)
    nearest_target = distances.argmin(axis=1)
    nearest_distance = distances[np.arange(len(teacher_centroids)), nearest_target]

    rows: list[dict[str, Any]] = []
    for teacher_code in range(len(teacher_centroids)):
        rows.append(
            {
                "layer": int(layer_index),
                "teacher_code": int(teacher_code),
                "target_code": int(nearest_target[teacher_code]),
                "distance_sq": float(nearest_distance[teacher_code]),
                "usage_count": int(usage_counts[teacher_code]),
                "is_active_in_target_items": bool(usage_counts[teacher_code] > 0),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    emb_df = load_item_embeddings(args.emb_path)
    embeddings = np.stack(emb_df["embedding"].to_numpy(), axis=0).astype(np.float32, copy=False)

    teacher_codebook = load_teacher_codebook(args.teacher_model_path, n_layers=args.n_layers)
    if embeddings.shape[1] != teacher_codebook.dim:
        raise ValueError(
            f"Embedding dim {embeddings.shape[1]} does not match teacher codebook dim {teacher_codebook.dim}.",
        )

    num_layers = teacher_codebook.n_layers
    target_sizes = resolve_target_sizes(args.target_codebook_sizes, num_layers)

    teacher_codes = batched_encode_embeddings(
        embeddings=embeddings,
        codebook=teacher_codebook,
        batch_size=args.batch_size,
        device=args.device,
    )
    teacher_mse = batched_reconstruction_mse(
        embeddings=embeddings,
        codes=teacher_codes,
        codebook=teacher_codebook,
        batch_size=args.batch_size,
        device=args.device,
    )

    item_output_df = emb_df[["item_id"]].copy()
    item_output_df["teacher_codes"] = teacher_codes.tolist()
    item_output_df["teacher_sid"] = item_output_df["teacher_codes"].apply(sid_from_codes)

    residual = embeddings.astype(np.float32, copy=True)
    distilled_layers: list[torch.Tensor] = []
    distilled_code_columns: list[np.ndarray] = []
    layer_stats: list[dict[str, Any]] = []
    transport_rows: list[dict[str, Any]] = []

    sample_weights = np.ones(len(embeddings), dtype=np.float32)

    for layer_index in range(num_layers):
        teacher_layer_centroids = teacher_codebook.centroids[layer_index].cpu().numpy().astype(np.float32)
        teacher_size = teacher_layer_centroids.shape[0]
        layer_teacher_codes = teacher_codes[:, layer_index]
        usage_counts = np.bincount(layer_teacher_codes, minlength=teacher_size)
        active_indices = np.flatnonzero(usage_counts > 0)
        active_count = int(len(active_indices))

        if active_count <= 0:
            raise ValueError(f"Layer {layer_index} has no active teacher codes on target embeddings.")

        requested_size = int(target_sizes[layer_index])
        actual_size = min(requested_size, active_count, len(embeddings))
        if actual_size <= 0:
            raise ValueError(f"Layer {layer_index} resulted in an invalid target size: {actual_size}")

        active_teacher_centroids = teacher_layer_centroids[active_indices]
        active_teacher_weights = usage_counts[active_indices].astype(np.float64) ** float(args.count_power)

        if actual_size == active_count:
            order = np.argsort(-active_teacher_weights, kind="stable")
            init_centers = active_teacher_centroids[order].copy()
            teacher_init_objective = 0.0
            teacher_cluster_weights = active_teacher_weights[order].copy()
        else:
            init_centers, init_assignments, teacher_init_objective = weighted_kmeans(
                points=active_teacher_centroids,
                weights=active_teacher_weights,
                k=actual_size,
                n_init=args.teacher_init_n_init,
                max_iter=args.teacher_init_max_iter,
                seed=args.seed + layer_index,
                device=args.device,
            )
            init_centers, init_assignments, teacher_cluster_weights = remap_assignments_by_cluster_weight(
                centers=init_centers,
                assignments=init_assignments,
                weights=active_teacher_weights,
            )

        refined_centers, refined_assignments, refine_objective = weighted_kmeans(
            points=residual,
            weights=sample_weights,
            k=actual_size,
            n_init=1,
            max_iter=args.refine_max_iter,
            seed=args.seed + 1000 + layer_index,
            init_centers=init_centers,
            device=args.device,
        )
        refined_centers, refined_assignments, refined_cluster_weights = remap_assignments_by_cluster_weight(
            centers=refined_centers,
            assignments=refined_assignments,
            weights=sample_weights,
        )

        quantized = refined_centers[refined_assignments]
        layer_mse = float(np.mean((residual - quantized) ** 2))
        residual = residual - quantized

        distilled_layers.append(torch.as_tensor(refined_centers, dtype=torch.float32))
        distilled_code_columns.append(refined_assignments.astype(np.int64))
        transport_rows.extend(
            build_transport_rows(
                layer_index=layer_index,
                teacher_centroids=teacher_layer_centroids,
                target_centroids=refined_centers.astype(np.float32),
                usage_counts=usage_counts,
            )
        )
        layer_stats.append(
            {
                "layer": layer_index,
                "teacher_codebook_size": int(teacher_size),
                "requested_target_size": int(requested_size),
                "actual_target_size": int(actual_size),
                "active_teacher_codes": int(active_count),
                "usage_coverage_ratio": active_count / float(teacher_size),
                "teacher_init_objective": float(teacher_init_objective),
                "refine_objective": float(refine_objective),
                "layer_quantization_mse": layer_mse,
                "top_teacher_cluster_weights": [
                    float(weight) for weight in teacher_cluster_weights[: min(10, len(teacher_cluster_weights))]
                ],
                "top_refined_cluster_weights": [
                    float(weight) for weight in refined_cluster_weights[: min(10, len(refined_cluster_weights))]
                ],
            }
        )

    distilled_codebook = ResidualCodebook(centroids=distilled_layers)
    distilled_codes = np.stack(distilled_code_columns, axis=1)
    distilled_mse = batched_reconstruction_mse(
        embeddings=embeddings,
        codes=distilled_codes,
        codebook=distilled_codebook,
        batch_size=args.batch_size,
        device=args.device,
    )

    item_output_df["codes"] = distilled_codes.tolist()
    item_output_df["sid"] = item_output_df["codes"].apply(sid_from_codes)

    item_output_path = output_dir / "distilled_item_sid_map.parquet"
    item_output_df.to_parquet(item_output_path, index=False)

    transport_df = pd.DataFrame(transport_rows)
    transport_path = output_dir / "transport_map.parquet"
    transport_df.to_parquet(transport_path, index=False)

    metadata = {
        "teacher_model_path": str(Path(args.teacher_model_path).resolve()),
        "emb_path": str(Path(args.emb_path).resolve()),
        "target_codebook_sizes": target_sizes,
        "count_power": args.count_power,
        "teacher_init_n_init": args.teacher_init_n_init,
        "teacher_init_max_iter": args.teacher_init_max_iter,
        "refine_max_iter": args.refine_max_iter,
        "seed": args.seed,
        "layer_stats": layer_stats,
        "teacher_embedding_mse": teacher_mse,
        "distilled_embedding_mse": distilled_mse,
    }
    codebook_path = output_dir / "distilled_codebook.pt"
    save_distilled_codebook(codebook_path, distilled_codebook, metadata)

    stats = {
        "teacher_model_path": str(Path(args.teacher_model_path).resolve()),
        "emb_path": str(Path(args.emb_path).resolve()),
        "num_items": int(len(item_output_df)),
        "embedding_dim": int(embeddings.shape[1]),
        "n_layers": int(num_layers),
        "teacher_codebook_sizes": teacher_codebook.codebook_sizes,
        "target_codebook_sizes": distilled_codebook.codebook_sizes,
        "teacher_embedding_mse": teacher_mse,
        "distilled_embedding_mse": distilled_mse,
        "layer_stats": layer_stats,
        "sid_collision": collision_stats(item_output_df["sid"].tolist()),
        "outputs": {
            "distilled_codebook": str(codebook_path.resolve()),
            "distilled_item_sid_map": str(item_output_path.resolve()),
            "transport_map": str(transport_path.resolve()),
        },
    }
    stats_path = output_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
