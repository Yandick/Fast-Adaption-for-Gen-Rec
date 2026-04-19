from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch


SID_BEGIN = "<|sid_begin|>"
SID_END = "<|sid_end|>"
SID_TOKEN_PATTERN = re.compile(r"<s_[a-z]_([0-9]+)>")


def parse_codes(raw_value: Any) -> list[int] | None:
    if raw_value is None:
        return None

    if isinstance(raw_value, np.ndarray):
        raw_value = raw_value.tolist()
    elif hasattr(raw_value, "tolist") and not isinstance(raw_value, (str, bytes)):
        raw_value = raw_value.tolist()

    if isinstance(raw_value, list):
        values = raw_value
    elif isinstance(raw_value, tuple):
        values = list(raw_value)
    elif isinstance(raw_value, str):
        text = raw_value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
            values = parsed if isinstance(parsed, list) else None
        except json.JSONDecodeError:
            values = [part.strip() for part in text.strip("[]").split(",") if part.strip()]
        if values is None:
            return None
    else:
        return None

    try:
        return [int(value) for value in values]
    except (TypeError, ValueError):
        return None


def parse_sid(raw_sid: Any) -> list[int] | None:
    if raw_sid is None:
        return None
    text = str(raw_sid).strip()
    if not text:
        return None
    matches = SID_TOKEN_PATTERN.findall(text)
    if not matches:
        return None
    return [int(value) for value in matches]


def sid_from_codes(codes: Sequence[int]) -> str:
    layer_tokens: list[str] = []
    for layer_index, code in enumerate(codes):
        token_suffix = chr(ord("a") + layer_index)
        layer_tokens.append(f"<s_{token_suffix}_{int(code)}>")
    return f"{SID_BEGIN}{''.join(layer_tokens)}{SID_END}"


def collision_stats(values: Sequence[str]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    total = len(values)
    unique = len(counts)
    collision_clusters = [count for count in counts.values() if count > 1]
    return {
        "num_total": total,
        "num_unique": unique,
        "collision_rate": 0.0 if total == 0 else 1.0 - unique / total,
        "num_collision_clusters": len(collision_clusters),
        "max_cluster_size": max(collision_clusters, default=1),
    }


def load_item_codes(path: str | Path, n_layers: int | None = None) -> pd.DataFrame:
    resolved = Path(path)
    suffix = resolved.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(resolved)
    elif suffix == ".csv":
        df = pd.read_csv(resolved)
    else:
        raise ValueError(f"Unsupported item code file format: {resolved}")

    identifier_column = None
    for candidate in ("item_id", "pid", "parent_asin"):
        if candidate in df.columns:
            identifier_column = candidate
            break
    if identifier_column is None:
        raise ValueError("Input item code file must contain one of: item_id, pid, parent_asin.")

    normalized_rows: list[dict[str, Any]] = []
    inferred_layers: int | None = n_layers
    for row in df.to_dict("records"):
        item_id = str(row.get(identifier_column) or "").strip()
        if not item_id:
            continue
        codes = parse_codes(row.get("codes"))
        if codes is None:
            codes = parse_sid(row.get("sid"))
        if codes is None:
            continue

        if inferred_layers is None:
            inferred_layers = len(codes)
        if len(codes) < inferred_layers:
            raise ValueError(f"Item {item_id} has only {len(codes)} codes, expected {inferred_layers}.")

        normalized_rows.append(
            {
                "item_id": item_id,
                "codes": [int(code) for code in codes[:inferred_layers]],
                "source_sid": row.get("sid"),
            }
        )

    if not normalized_rows:
        raise ValueError(f"No valid item codes could be loaded from {resolved}.")

    out_df = pd.DataFrame(normalized_rows)
    if out_df["item_id"].duplicated().any():
        raise ValueError("Input item code file contains duplicated item identifiers.")
    return out_df.reset_index(drop=True)


def load_item_embeddings(path: str | Path) -> pd.DataFrame:
    resolved = Path(path)
    df = pd.read_parquet(resolved)

    identifier_column = None
    for candidate in ("item_id", "pid", "parent_asin"):
        if candidate in df.columns:
            identifier_column = candidate
            break
    if identifier_column is None:
        raise ValueError("Embedding parquet must contain one of: item_id, pid, parent_asin.")
    if "embedding" not in df.columns:
        raise ValueError("Embedding parquet must contain an embedding column.")

    item_ids = df[identifier_column].fillna("").astype(str).str.strip()
    embedding_values = df["embedding"].to_numpy()

    normalized_ids: list[str] = []
    normalized_embeddings: list[np.ndarray] = []
    inferred_dim: int | None = None

    for item_id, embedding_value in zip(item_ids, embedding_values, strict=False):
        if not item_id:
            continue
        if embedding_value is None:
            continue

        if isinstance(embedding_value, np.ndarray):
            embedding = np.asarray(embedding_value, dtype=np.float32)
        elif hasattr(embedding_value, "tolist") and not isinstance(embedding_value, (str, bytes)):
            embedding = np.asarray(embedding_value.tolist(), dtype=np.float32)
        elif isinstance(embedding_value, list):
            embedding = np.asarray(embedding_value, dtype=np.float32)
        else:
            raise TypeError(f"Unsupported embedding type for item {item_id}: {type(embedding_value)!r}")

        if embedding.ndim != 1:
            raise ValueError(f"Embedding for item {item_id} must be 1D, got {embedding.shape}.")
        if inferred_dim is None:
            inferred_dim = int(embedding.shape[0])
        if int(embedding.shape[0]) != inferred_dim:
            raise ValueError(
                f"Embedding dimension mismatch for item {item_id}: got {embedding.shape[0]}, expected {inferred_dim}.",
            )

        normalized_ids.append(item_id)
        normalized_embeddings.append(embedding)

    if not normalized_ids:
        raise ValueError(f"No valid embeddings could be loaded from {resolved}.")

    out_df = pd.DataFrame({"item_id": normalized_ids, "embedding": normalized_embeddings})
    if out_df["item_id"].duplicated().any():
        raise ValueError("Embedding parquet contains duplicated item identifiers.")
    return out_df.reset_index(drop=True)


@dataclass
class ResidualCodebook:
    centroids: list[torch.Tensor]

    @property
    def n_layers(self) -> int:
        return len(self.centroids)

    @property
    def dim(self) -> int:
        if not self.centroids:
            raise ValueError("ResidualCodebook has no centroids.")
        return int(self.centroids[0].shape[1])

    @property
    def codebook_sizes(self) -> list[int]:
        return [int(layer.shape[0]) for layer in self.centroids]

    def to(self, device: str | torch.device) -> "ResidualCodebook":
        return ResidualCodebook([layer.to(device) for layer in self.centroids])

    def cpu(self) -> "ResidualCodebook":
        return self.to("cpu")

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        output = torch.zeros((codes.shape[0], self.dim), dtype=torch.float32, device=codes.device)
        for layer_index, layer_centroids in enumerate(self.centroids):
            output += layer_centroids[codes[:, layer_index]]
        return output

    def encode(self, embeddings: torch.Tensor) -> torch.Tensor:
        residual = embeddings.clone()
        assignments: list[torch.Tensor] = []
        for layer_centroids in self.centroids:
            residual_norm_sq = residual.pow(2.0).sum(dim=1, keepdim=True)
            centroid_norm_sq = layer_centroids.T.pow(2.0).sum(dim=0, keepdim=True)
            distances = torch.addmm(
                residual_norm_sq + centroid_norm_sq,
                residual,
                layer_centroids.T,
                alpha=-2.0,
            )
            code = distances.argmin(dim=-1)
            assignments.append(code)
            residual = residual - layer_centroids[code]
        return torch.stack(assignments, dim=1)


def load_teacher_codebook(model_path: str | Path, n_layers: int | None = None) -> ResidualCodebook:
    checkpoint = torch.load(model_path, map_location="cpu")

    if isinstance(checkpoint, dict) and checkpoint.get("format") == "distilled_residual_codebook_v1":
        raw_centroids = checkpoint["centroids"]
        centroids = [torch.as_tensor(layer, dtype=torch.float32) for layer in raw_centroids]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model") or checkpoint.get("state_dict") or checkpoint
        centroid_keys = sorted(
            [key for key in state_dict.keys() if key.startswith("centroids.")],
            key=lambda key: int(key.split(".")[1]),
        )
        if not centroid_keys:
            raise ValueError(f"No centroid weights found in checkpoint: {model_path}")
        centroids = [torch.as_tensor(state_dict[key], dtype=torch.float32) for key in centroid_keys]
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)!r}")

    if n_layers is not None:
        if n_layers > len(centroids):
            raise ValueError(f"Requested {n_layers} layers but checkpoint only has {len(centroids)}.")
        centroids = centroids[:n_layers]

    dim = int(centroids[0].shape[1])
    for layer_index, layer_centroids in enumerate(centroids):
        if layer_centroids.ndim != 2:
            raise ValueError(f"Layer {layer_index} centroids must be 2D, got {layer_centroids.shape}.")
        if int(layer_centroids.shape[1]) != dim:
            raise ValueError("All centroid layers must share the same embedding dimension.")
    return ResidualCodebook(centroids=centroids)


def save_distilled_codebook(
    output_path: str | Path,
    codebook: ResidualCodebook,
    metadata: dict[str, Any],
) -> None:
    payload = {
        "format": "distilled_residual_codebook_v1",
        "n_layers": codebook.n_layers,
        "dim": codebook.dim,
        "codebook_sizes": codebook.codebook_sizes,
        "centroids": [layer.detach().cpu() for layer in codebook.centroids],
        "metadata": metadata,
    }
    resolved = Path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, resolved)


def resolve_target_sizes(raw_value: str, n_layers: int) -> list[int]:
    values = [int(part.strip()) for part in raw_value.split(",") if part.strip()]
    if not values:
        raise ValueError("--target_codebook_sizes must contain at least one integer.")
    if len(values) == 1:
        values = values * n_layers
    if len(values) != n_layers:
        raise ValueError(f"Expected {n_layers} target sizes, got {values}.")
    if any(value <= 0 for value in values):
        raise ValueError(f"Target codebook sizes must be positive, got {values}.")
    return values


def _squared_l2_distance(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    points_norm = (points**2).sum(axis=1, keepdims=True)
    centers_norm = (centers**2).sum(axis=1, keepdims=True).T
    return np.maximum(points_norm + centers_norm - 2.0 * points @ centers.T, 0.0)


def _squared_l2_distance_torch(points: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    points_norm = points.pow(2.0).sum(dim=1, keepdim=True)
    centers_norm = centers.pow(2.0).sum(dim=1, keepdim=True).T
    distances = torch.addmm(points_norm + centers_norm, points, centers.T, alpha=-2.0)
    return distances.clamp_min_(0.0)


def _weighted_kmeans_plus_plus(
    points: np.ndarray,
    weights: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    num_points = points.shape[0]
    first_index = int(rng.choice(num_points, p=weights / weights.sum()))
    center_indices = [first_index]
    min_distances = _squared_l2_distance(points, points[[first_index]]).reshape(-1)

    for _ in range(1, k):
        probs = weights * np.maximum(min_distances, 1e-12)
        if probs.sum() <= 0:
            next_index = int(rng.integers(0, num_points))
        else:
            next_index = int(rng.choice(num_points, p=probs / probs.sum()))
        center_indices.append(next_index)
        new_distances = _squared_l2_distance(points, points[[next_index]]).reshape(-1)
        min_distances = np.minimum(min_distances, new_distances)

    return points[np.asarray(center_indices)].copy()


def _weighted_kmeans_torch(
    points: np.ndarray,
    weights: np.ndarray,
    k: int,
    n_init: int,
    max_iter: int,
    seed: int,
    init_centers: np.ndarray | None,
    device: str | torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    points_np = np.asarray(points, dtype=np.float32)
    weights_np = np.maximum(np.asarray(weights, dtype=np.float32), 1e-12)

    points_t = torch.as_tensor(points_np, dtype=torch.float32, device=device)
    weights_t = torch.as_tensor(weights_np, dtype=torch.float32, device=device)
    weighted_points = points_t * weights_t.unsqueeze(1)

    best_centers: torch.Tensor | None = None
    best_assignments: torch.Tensor | None = None
    best_objective = math.inf

    num_inits = max(n_init, 1)
    for init_index in range(num_inits):
        if init_index == 0 and init_centers is not None:
            centers_np = np.asarray(init_centers, dtype=np.float32).copy()
            if centers_np.shape != (k, points_np.shape[1]):
                raise ValueError(
                    f"init_centers must have shape {(k, points_np.shape[1])}, got {centers_np.shape}.",
                )
        else:
            centers_np = _weighted_kmeans_plus_plus(
                points=points_np,
                weights=weights_np,
                k=k,
                rng=np.random.default_rng(seed + init_index),
            ).astype(np.float32, copy=False)
        centers_t = torch.as_tensor(centers_np, dtype=torch.float32, device=device)
        assignments_t = torch.full((points_t.shape[0],), -1, dtype=torch.long, device=device)

        for _ in range(max(max_iter, 1)):
            distances = _squared_l2_distance_torch(points_t, centers_t)
            new_assignments = distances.argmin(dim=1)
            if torch.equal(new_assignments, assignments_t):
                assignments_t = new_assignments
                break
            assignments_t = new_assignments

            cluster_weight_sums = torch.zeros((k,), dtype=torch.float32, device=device)
            cluster_weight_sums.index_add_(0, assignments_t, weights_t)
            center_sums = torch.zeros((k, points_t.shape[1]), dtype=torch.float32, device=device)
            center_sums.index_add_(0, assignments_t, weighted_points)

            updated_centers = center_sums / cluster_weight_sums.clamp_min(1e-12).unsqueeze(1)
            empty_mask = cluster_weight_sums <= 1e-12
            if empty_mask.any():
                min_distances = distances.gather(1, assignments_t[:, None]).squeeze(1)
                fallback_indices = torch.topk(
                    weights_t * min_distances,
                    k=int(empty_mask.sum().item()),
                ).indices
                updated_centers[empty_mask] = points_t[fallback_indices]
            centers_t = updated_centers

        final_distances = _squared_l2_distance_torch(points_t, centers_t)
        assignments_t = final_distances.argmin(dim=1)
        min_distances = final_distances.gather(1, assignments_t[:, None]).squeeze(1)
        objective = float((weights_t * min_distances).sum().item())
        if objective < best_objective:
            best_objective = objective
            best_centers = centers_t.detach().clone()
            best_assignments = assignments_t.detach().clone()

    assert best_centers is not None
    assert best_assignments is not None
    return best_centers.cpu().numpy().astype(np.float32), best_assignments.cpu().numpy().astype(np.int64), best_objective


def weighted_kmeans(
    points: np.ndarray,
    weights: np.ndarray,
    k: int,
    n_init: int,
    max_iter: int,
    seed: int,
    init_centers: np.ndarray | None = None,
    device: str | torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    if points.ndim != 2:
        raise ValueError(f"Expected 2D point matrix, got {points.shape}.")
    if len(points) != len(weights):
        raise ValueError("points and weights must have the same length.")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}.")
    if len(points) < k:
        raise ValueError(f"k={k} exceeds number of points={len(points)}.")

    if len(points) == k:
        assignments = np.arange(k, dtype=np.int64)
        return points.copy(), assignments, 0.0

    if device is not None:
        torch_device = torch.device(device)
        if torch_device.type == "cuda" and torch.cuda.is_available():
            return _weighted_kmeans_torch(
                points=points,
                weights=weights,
                k=k,
                n_init=n_init,
                max_iter=max_iter,
                seed=seed,
                init_centers=init_centers,
                device=torch_device,
            )

    points = np.asarray(points, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    weights = np.maximum(weights, 1e-12)

    best_centers: np.ndarray | None = None
    best_assignments: np.ndarray | None = None
    best_objective = math.inf

    num_inits = max(n_init, 1)
    for init_index in range(num_inits):
        rng = np.random.default_rng(seed + init_index)
        if init_index == 0 and init_centers is not None:
            centers = np.asarray(init_centers, dtype=np.float32).copy()
            if centers.shape != (k, points.shape[1]):
                raise ValueError(
                    f"init_centers must have shape {(k, points.shape[1])}, got {centers.shape}.",
                )
        else:
            centers = _weighted_kmeans_plus_plus(points, weights, k, rng)
        assignments = np.zeros(len(points), dtype=np.int64)

        for _ in range(max(max_iter, 1)):
            distances = _squared_l2_distance(points, centers)
            new_assignments = distances.argmin(axis=1)
            if np.array_equal(new_assignments, assignments):
                assignments = new_assignments
                break
            assignments = new_assignments

            for cluster_index in range(k):
                mask = assignments == cluster_index
                if not mask.any():
                    weighted_distances = weights * distances.min(axis=1)
                    fallback_index = int(weighted_distances.argmax())
                    centers[cluster_index] = points[fallback_index]
                    assignments[fallback_index] = cluster_index
                    continue
                centers[cluster_index] = np.average(points[mask], axis=0, weights=weights[mask])

        final_distances = _squared_l2_distance(points, centers)
        assignments = final_distances.argmin(axis=1)
        min_distances = final_distances[np.arange(len(points)), assignments]
        objective = float((weights * min_distances).sum())
        if objective < best_objective:
            best_objective = objective
            best_centers = centers.copy()
            best_assignments = assignments.copy()

    assert best_centers is not None
    assert best_assignments is not None
    return best_centers.astype(np.float32), best_assignments.astype(np.int64), best_objective


def summarize_cluster_weights(assignments: np.ndarray, weights: np.ndarray, k: int) -> np.ndarray:
    cluster_weights = np.zeros(k, dtype=np.float64)
    for cluster_index in range(k):
        cluster_weights[cluster_index] = float(weights[assignments == cluster_index].sum())
    return cluster_weights


def remap_assignments_by_cluster_weight(
    centers: np.ndarray,
    assignments: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cluster_weights = summarize_cluster_weights(assignments, weights, centers.shape[0])
    order = np.argsort(-cluster_weights, kind="stable")
    inverse_order = np.zeros_like(order)
    inverse_order[order] = np.arange(len(order))
    remapped_centers = centers[order]
    remapped_assignments = inverse_order[assignments]
    remapped_cluster_weights = cluster_weights[order]
    return remapped_centers, remapped_assignments, remapped_cluster_weights


def batched_reencode_items(
    source_codes: np.ndarray,
    teacher_codebook: ResidualCodebook,
    target_codebook: ResidualCodebook,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, dict[str, float]]:
    teacher = teacher_codebook.to(device)
    target = target_codebook.to(device)

    all_codes: list[np.ndarray] = []
    total_examples = 0
    total_mse = 0.0

    with torch.no_grad():
        for start in range(0, len(source_codes), batch_size):
            end = min(start + batch_size, len(source_codes))
            batch_codes = torch.as_tensor(source_codes[start:end], dtype=torch.long, device=device)
            teacher_embeddings = teacher.decode(batch_codes)
            distilled_codes = target.encode(teacher_embeddings)
            reconstructed = target.decode(distilled_codes)
            mse = torch.mean((reconstructed - teacher_embeddings) ** 2).item()
            total_mse += mse * (end - start)
            total_examples += end - start
            all_codes.append(distilled_codes.cpu().numpy())

    return np.concatenate(all_codes, axis=0), {
        "teacher_reconstruction_mse": total_mse / max(total_examples, 1),
        "num_items_reencoded": total_examples,
    }


def batched_encode_embeddings(
    embeddings: np.ndarray,
    codebook: ResidualCodebook,
    batch_size: int,
    device: str,
) -> np.ndarray:
    codebook_device = codebook.to(device)
    all_codes = np.empty((len(embeddings), codebook.n_layers), dtype=np.int64)

    with torch.no_grad():
        for start in range(0, len(embeddings), batch_size):
            end = min(start + batch_size, len(embeddings))
            batch_embeddings = torch.as_tensor(embeddings[start:end], dtype=torch.float32, device=device)
            batch_codes = codebook_device.encode(batch_embeddings)
            all_codes[start:end] = batch_codes.cpu().numpy()

    return all_codes


def batched_reconstruction_mse(
    embeddings: np.ndarray,
    codes: np.ndarray,
    codebook: ResidualCodebook,
    batch_size: int,
    device: str,
) -> float:
    codebook_device = codebook.to(device)
    total_squared_error = 0.0
    total_values = 0

    with torch.no_grad():
        for start in range(0, len(embeddings), batch_size):
            end = min(start + batch_size, len(embeddings))
            batch_embeddings = torch.as_tensor(embeddings[start:end], dtype=torch.float32, device=device)
            batch_codes = torch.as_tensor(codes[start:end], dtype=torch.long, device=device)
            reconstructed = codebook_device.decode(batch_codes)
            diff = reconstructed - batch_embeddings
            total_squared_error += float(diff.pow(2.0).sum().item())
            total_values += int(diff.numel())

    return total_squared_error / max(total_values, 1)
