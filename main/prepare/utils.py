from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SID_BEGIN = "<|sid_begin|>"
SID_END = "<|sid_end|>"


@dataclass
class Sample:
    user_id: str
    source_items: list[str]
    target_items: list[str]
    source_count: int
    target_count: int

    @property
    def label_item(self) -> str | None:
        return self.target_items[0] if self.target_items else None


def split_pipe(value: str | None) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part for part in text.split("|") if part]


def load_user_sequences(csv_path: str | Path) -> list[Sample]:
    path = Path(csv_path)
    rows: list[Sample] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                Sample(
                    user_id=row["user_id"],
                    source_items=split_pipe(row.get("source_items")),
                    target_items=split_pipe(row.get("target_items")),
                    source_count=int(row["source_count"]),
                    target_count=int(row["target_count"]),
                )
            )
    return rows


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_item_text(meta: dict[str, Any]) -> str:
    title = str(meta.get("title") or "").strip()
    store = str(meta.get("store") or "").strip()
    main_category = str(meta.get("main_category") or "").strip()
    categories = [str(x).strip() for x in meta.get("categories") or [] if str(x).strip()]
    features = [str(x).strip() for x in meta.get("features") or [] if str(x).strip()]
    description = [str(x).strip() for x in meta.get("description") or [] if str(x).strip()]

    parts: list[str] = []
    if title:
        parts.append(f"Title: {title}")
    if store:
        parts.append(f"Store: {store}")
    if main_category:
        parts.append(f"Main category: {main_category}")
    if categories:
        parts.append(f"Categories: {'; '.join(categories[:10])}")
    if features:
        parts.append(f"Features: {'; '.join(features[:8])}")
    if description:
        parts.append(f"Description: {' '.join(description[:4])}")

    if not parts:
        parts.append(f"Item id: {meta.get('parent_asin', '')}")
    return "\n".join(parts)


def load_meta_filtered(
    meta_path: str | Path,
    domain_name: str,
    keep_ids: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for meta in iter_jsonl(meta_path):
        item_id = str(meta.get("parent_asin") or "").strip()
        if not item_id:
            continue
        if keep_ids is not None and item_id not in keep_ids:
            continue
        result[item_id] = {
            "item_id": item_id,
            "domain": domain_name,
            "is_target_candidate": False,
            "text": build_item_text(meta),
            "title": meta.get("title"),
        }
    return result


def sid_from_codes(codes: list[int]) -> str:
    if len(codes) != 3:
        raise ValueError(f"Expected 3-layer codes, got {codes}")
    return f"{SID_BEGIN}<s_a_{codes[0]}><s_b_{codes[1]}><s_c_{codes[2]}>{SID_END}"


def sid_suffix_token_ids(codes: list[int], tokenizer: Any) -> list[int]:
    tokens = [
        f"<s_a_{codes[0]}>",
        f"<s_b_{codes[1]}>",
        f"<s_c_{codes[2]}>",
        SID_END,
    ]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    if any(token_id is None or token_id == tokenizer.unk_token_id for token_id in token_ids):
        raise ValueError(f"Failed to convert SID tokens to ids: {tokens}")
    return token_ids


class TokenTrie:
    def __init__(self) -> None:
        self.root: dict[int, dict] = {}

    def add(self, sequence: list[int]) -> None:
        node = self.root
        for token_id in sequence:
            node = node.setdefault(token_id, {})

    def allowed(self, prefix: list[int]) -> list[int]:
        node = self.root
        for token_id in prefix:
            if token_id not in node:
                return []
            node = node[token_id]
        return list(node.keys())


def build_metrics(prediction_lists: list[list[str]], label: str, k_values: list[int]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in k_values:
        hit_rank = None
        for rank, candidates in enumerate(prediction_lists[:k], start=1):
            if label in candidates:
                hit_rank = rank
                break
        metrics[f"recall@{k}"] = 1.0 if hit_rank is not None else 0.0
        metrics[f"ndcg@{k}"] = 1.0 / math.log2(hit_rank + 1) if hit_rank is not None else 0.0
    return metrics


def mean_metric(records: list[dict[str, float]], key: str) -> float:
    if not records:
        return 0.0
    return sum(record[key] for record in records) / len(records)

