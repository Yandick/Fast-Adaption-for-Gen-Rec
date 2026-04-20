from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from utils import load_meta_filtered


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class LocalQwenEmbeddingClient:
    def __init__(
        self,
        model_path: str,
        max_length: int,
        device: str,
    ) -> None:
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto" if device == "cuda" else None,
        )
        if device != "cuda":
            self.model = self.model.to(device)
        self.model.eval()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        batch_dict = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if hasattr(self.model, "device"):
            model_device = self.model.device
            batch_dict = {key: value.to(model_device) for key, value in batch_dict.items()}
        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare item metadata and embedding inputs for tokenizer inference.",
    )
    parser.add_argument(
        "--user_sequences_csv",
        required=True,
        help=(
            "Raw interaction CSV with columns `user_id,parent_asin,rating,timestamp`. "
            "This script only uses the unique `parent_asin` values."
        ),
    )
    parser.add_argument("--target_meta_path", required=True)
    parser.add_argument("--target_domain", required=True)
    parser.add_argument("--source_meta_paths", nargs="*", default=[])
    parser.add_argument("--source_domains", nargs="*", default=[])
    parser.add_argument("--item_metadata_output", required=True)
    parser.add_argument("--embedding_output", required=True)
    parser.add_argument("--target_item_scope", choices=["all_meta", "labels_only"], default="all_meta")
    parser.add_argument("--embedding_model_path", required=True)
    parser.add_argument("--embedding_max_length", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_unique_item_ids(csv_path: str | Path) -> set[str]:
    path = Path(csv_path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])

        if "parent_asin" not in fieldnames:
            raise ValueError(
                "user_sequences_csv must contain a `parent_asin` column. "
                "Expected raw interaction CSV with columns `user_id,parent_asin,rating,timestamp`.",
            )

        unique_item_ids: set[str] = set()
        for row in reader:
            item_id = str(row.get("parent_asin") or "").strip()
            if item_id:
                unique_item_ids.add(item_id)

        if not unique_item_ids:
            raise ValueError(f"No valid parent_asin values found in {path}.")

        return unique_item_ids


def collect_item_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    unique_item_ids = load_unique_item_ids(args.user_sequences_csv)
    print(
        f"[INFO] Loaded {len(unique_item_ids)} unique items from {args.user_sequences_csv}",
    )
    source_ids = unique_item_ids
    target_filter_ids = unique_item_ids

    source_domain_names = args.source_domains
    if args.source_meta_paths and source_domain_names and len(args.source_meta_paths) != len(source_domain_names):
        raise ValueError("--source_meta_paths and --source_domains must have the same length.")

    items: dict[str, dict[str, Any]] = {}

    keep_target_ids = None if args.target_item_scope == "all_meta" else target_filter_ids
    target_items = load_meta_filtered(args.target_meta_path, args.target_domain, keep_target_ids)
    for item in target_items.values():
        item["is_target_candidate"] = True
    items.update(target_items)

    for idx, meta_path in enumerate(args.source_meta_paths):
        domain = source_domain_names[idx] if idx < len(source_domain_names) else Path(meta_path).stem.replace("meta_", "")
        source_items = load_meta_filtered(meta_path, domain, source_ids)
        items.update(source_items)

    missing_source = source_ids.difference(items.keys())
    missing_target = target_filter_ids.difference(items.keys())
    if missing_source:
        print(f"[WARN] Missing metadata for {len(missing_source)} source items.")
    if missing_target:
        print(f"[WARN] Missing metadata for {len(missing_target)} target label items.")

    rows = list(items.values())
    if not rows:
        raise ValueError("No items loaded for tokenizer input preparation.")
    return rows


def main() -> None:
    args = parse_args()
    rows = collect_item_rows(args)

    metadata_output = Path(args.item_metadata_output)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_df = pd.DataFrame(rows)
    metadata_df.to_parquet(metadata_output, index=False)
    print(f"[INFO] Saved item metadata to {metadata_output}")

    embedding_client = LocalQwenEmbeddingClient(
        model_path=args.embedding_model_path,
        max_length=args.embedding_max_length,
        device=args.device,
    )

    embedding_records: list[dict[str, Any]] = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        texts = [row["text"] for row in batch]
        embeddings = embedding_client.embed_batch(texts)
        for row, embedding in zip(batch, embeddings):
            embedding_records.append(
                {
                    "pid": row["item_id"],
                    "embedding": embedding,
                }
            )
        print(f"[INFO] Embedded {min(start + args.batch_size, len(rows))}/{len(rows)} items")

    embedding_output = Path(args.embedding_output)
    embedding_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(embedding_records).to_parquet(embedding_output, index=False)
    print(f"[INFO] Saved tokenizer embedding inputs to {embedding_output}")


if __name__ == "__main__":
    main()
