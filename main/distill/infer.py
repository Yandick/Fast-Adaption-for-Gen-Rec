from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from distill_utils import load_teacher_codebook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode item embeddings with a distilled residual codebook.",
    )
    parser.add_argument("--model_path", required=True, help="Path to distilled_codebook.pt.")
    parser.add_argument(
        "--emb_path",
        required=True,
        help="Parquet file with item_id/pid and embedding columns.",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="Output parquet path. Defaults to {emb_path}_distilled_codes.parquet.",
    )
    parser.add_argument("--batch_size", type=int, default=4096, help="Inference batch size.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device used for encoding.",
    )
    return parser.parse_args()


def load_embeddings(path: str | Path) -> tuple[str, list[str], torch.Tensor]:
    df = pd.read_parquet(path)

    identifier_column = None
    for candidate in ("item_id", "pid", "parent_asin"):
        if candidate in df.columns:
            identifier_column = candidate
            break
    if identifier_column is None:
        raise ValueError("Embedding parquet must contain one of: item_id, pid, parent_asin.")
    if "embedding" not in df.columns:
        raise ValueError("Embedding parquet must contain an embedding column.")

    identifiers = [str(value) for value in df[identifier_column].tolist()]
    embeddings = np.stack(df["embedding"].values).astype(np.float32)
    return identifier_column, identifiers, torch.as_tensor(embeddings, dtype=torch.float32)


def main() -> None:
    args = parse_args()
    codebook = load_teacher_codebook(args.model_path)
    codebook = codebook.to(args.device)

    identifier_column, identifiers, embeddings = load_embeddings(args.emb_path)

    all_codes: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(embeddings), args.batch_size):
            end = min(start + args.batch_size, len(embeddings))
            batch_embeddings = embeddings[start:end].to(args.device)
            batch_codes = codebook.encode(batch_embeddings)
            all_codes.append(batch_codes.cpu())

    codes = torch.cat(all_codes, dim=0)
    reconstructed = codebook.decode(codes.to(args.device)).cpu()
    mse = torch.mean((reconstructed - embeddings) ** 2).item()

    output_path = Path(args.output_path) if args.output_path else Path(args.emb_path).with_name(
        Path(args.emb_path).stem + "_distilled_codes.parquet"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_df = pd.DataFrame(
        {
            identifier_column: identifiers,
            "codes": codes.numpy().tolist(),
        }
    )
    output_df.to_parquet(output_path, index=False)

    print(f"[INFO] Saved distilled codes to {output_path.resolve()}")
    print(f"[INFO] Reconstruction MSE: {mse:.6f}")


if __name__ == "__main__":
    main()

