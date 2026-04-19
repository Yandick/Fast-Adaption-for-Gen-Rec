from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from utils import sid_from_codes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge tokenizer codes with item metadata and build the final item->SID map.",
    )
    parser.add_argument(
        "--item_metadata_parquet",
        default="experiments/constrained_decoding_amazon/outputs/item_metadata.parquet",
    )
    parser.add_argument(
        "--codes_parquet",
        default="experiments/constrained_decoding_amazon/outputs/item_embeddings_codes.parquet",
    )
    parser.add_argument(
        "--output_parquet",
        default="experiments/constrained_decoding_amazon/outputs/item_sid_map.parquet",
    )
    parser.add_argument("--expected_code_layers", type=int, default=3)
    return parser.parse_args()


def normalize_codes(raw_codes: Any, expected_code_layers: int) -> list[int]:
    if hasattr(raw_codes, "tolist"):
        raw_codes = raw_codes.tolist()
    if not isinstance(raw_codes, (list, tuple)):
        raise TypeError(f"Tokenizer codes must be a list or tuple, got {type(raw_codes)!r}")

    codes = [int(code) for code in raw_codes]
    if len(codes) != expected_code_layers:
        raise ValueError(f"Expected {expected_code_layers} code layers, got {codes}")
    return codes


def normalize_path(raw_path: str) -> Path:
    return Path(raw_path.strip())


def validate_nonempty_parquet(path: Path, arg_name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{arg_name} does not exist: {path}")
    if path.stat().st_size == 0:
        raise ValueError(
            f"{arg_name} points to an empty file: {path}. "
            "This usually means the upstream export did not finish or wrote to a different path.",
        )


def main() -> None:
    args = parse_args()
    metadata_path = normalize_path(args.item_metadata_parquet)
    codes_path = normalize_path(args.codes_parquet)
    output_path = normalize_path(args.output_parquet)

    validate_nonempty_parquet(metadata_path, "--item_metadata_parquet")
    validate_nonempty_parquet(codes_path, "--codes_parquet")

    metadata_df = pd.read_parquet(metadata_path)
    if "item_id" not in metadata_df.columns:
        raise ValueError("item metadata parquet must contain an item_id column.")

    codes_df = pd.read_parquet(codes_path)
    if "pid" in codes_df.columns:
        codes_df = codes_df.rename(columns={"pid": "item_id"})
    elif "item_id" not in codes_df.columns:
        raise ValueError("codes parquet must contain either a pid column or an item_id column.")

    if "codes" not in codes_df.columns:
        raise ValueError("codes parquet must contain a codes column.")
    if codes_df["item_id"].duplicated().any():
        raise ValueError("codes parquet contains duplicated item identifiers.")

    merged_df = metadata_df.merge(
        codes_df[["item_id", "codes"]],
        on="item_id",
        how="left",
        validate="one_to_one",
    )

    missing_mask = merged_df["codes"].isna()
    if missing_mask.any():
        missing_count = int(missing_mask.sum())
        raise ValueError(f"Missing tokenizer codes for {missing_count} items.")

    merged_df["codes"] = merged_df["codes"].apply(
        lambda value: normalize_codes(value, args.expected_code_layers),
    )
    merged_df["sid"] = merged_df["codes"].apply(sid_from_codes)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(output_path, index=False)
    print(f"[INFO] Saved item SID map to {output_path}")


if __name__ == "__main__":
    main()
