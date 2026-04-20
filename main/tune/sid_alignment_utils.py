from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


IM_END = "<|im_end|>"
SID_TOKEN_TEMPLATE = "<s_{layer}_{index}>"
SID_TOKEN_PATTERN = re.compile(r"<s_([a-z])_([0-9]+)>")


def layer_letter(layer_index: int) -> str:
    return chr(ord("a") + layer_index)


def parse_json_list(raw_value: Any) -> list[Any] | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, list):
        return raw_value
    if isinstance(raw_value, tuple):
        return list(raw_value)
    if isinstance(raw_value, np.ndarray):
        return raw_value.tolist()
    if hasattr(raw_value, "tolist") and not isinstance(raw_value, (str, bytes)):
        return raw_value.tolist()
    if isinstance(raw_value, str):
        text = raw_value.strip()
        if not text:
            return None
        try:
            value = json.loads(text)
            return value if isinstance(value, list) else None
        except json.JSONDecodeError:
            parts = [part.strip() for part in text.strip("[]").split(",") if part.strip()]
            return parts
    return None


def parse_codes(raw_value: Any) -> list[int] | None:
    parsed = parse_json_list(raw_value)
    if parsed is None:
        return None
    try:
        return [int(value) for value in parsed]
    except (TypeError, ValueError):
        return None


def parse_sid(raw_value: Any) -> list[int] | None:
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    matches = SID_TOKEN_PATTERN.findall(text)
    if not matches:
        return None
    return [int(index) for _, index in matches]


def load_transport_map(path: str | Path) -> pd.DataFrame:
    resolved = Path(path)
    if resolved.suffix.lower() == ".parquet":
        frame = pd.read_parquet(resolved)
    elif resolved.suffix.lower() == ".csv":
        frame = pd.read_csv(resolved)
    else:
        raise ValueError(f"Unsupported transport map format: {resolved}")

    required_columns = {"layer", "teacher_code", "target_code", "distance_sq", "usage_count"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"transport_map is missing columns: {sorted(missing_columns)}")
    return frame.copy()


def infer_layer_sizes_from_transport(transport_map: pd.DataFrame) -> tuple[list[int], list[int]]:
    old_sizes: list[int] = []
    new_sizes: list[int] = []
    for layer_index in sorted(int(value) for value in transport_map["layer"].unique()):
        layer_frame = transport_map.loc[transport_map["layer"] == layer_index]
        old_sizes.append(int(layer_frame["teacher_code"].max()) + 1)
        new_sizes.append(int(layer_frame["target_code"].max()) + 1)
    return old_sizes, new_sizes


def load_item_sid_map(path: str | Path) -> pd.DataFrame:
    resolved = Path(path)
    if resolved.suffix.lower() == ".parquet":
        frame = pd.read_parquet(resolved)
    elif resolved.suffix.lower() == ".csv":
        frame = pd.read_csv(resolved)
    else:
        raise ValueError(f"Unsupported SID map format: {resolved}")

    identifier_column = None
    for candidate in ("item_id", "pid", "parent_asin"):
        if candidate in frame.columns:
            identifier_column = candidate
            break
    if identifier_column is None:
        raise ValueError("item_sid_map must contain one of: item_id, pid, parent_asin")

    rows: list[dict[str, Any]] = []
    inferred_layers: int | None = None
    for row in frame.to_dict("records"):
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
        if len(codes) != inferred_layers:
            raise ValueError(f"Inconsistent number of SID layers for item {item_id}.")
        rows.append({"item_id": item_id, "codes": codes})

    if not rows:
        raise ValueError(f"No valid SID rows found in {resolved}.")
    return pd.DataFrame(rows)


def infer_layer_sizes_from_item_sid_map(path: str | Path) -> list[int]:
    frame = load_item_sid_map(path)
    stacked_codes = np.asarray(frame["codes"].tolist(), dtype=np.int64)
    return [int(stacked_codes[:, layer_index].max()) + 1 for layer_index in range(stacked_codes.shape[1])]


def itemic_token(layer_index: int, code_index: int) -> str:
    return SID_TOKEN_TEMPLATE.format(layer=layer_letter(layer_index), index=int(code_index))


def resolve_token_ids(tokenizer: Any, layer_sizes: list[int], layer_indices: Iterable[int] | None = None) -> dict[int, list[int]]:
    selected_layers = list(layer_indices) if layer_indices is not None else list(range(len(layer_sizes)))
    layer_token_ids: dict[int, list[int]] = {}
    for layer_index in selected_layers:
        token_ids: list[int] = []
        for code_index in range(int(layer_sizes[layer_index])):
            token = itemic_token(layer_index, code_index)
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is None or token_id == tokenizer.unk_token_id:
                raise ValueError(f"Failed to resolve token id for itemic token: {token}")
            token_ids.append(int(token_id))
        layer_token_ids[layer_index] = token_ids
    return layer_token_ids


def flatten_token_ids(layer_token_ids: dict[int, list[int]]) -> list[int]:
    flat_ids: list[int] = []
    for layer_index in sorted(layer_token_ids):
        flat_ids.extend(layer_token_ids[layer_index])
    return flat_ids


def select_layer_indices(raw_value: str, n_layers: int) -> list[int]:
    if raw_value.strip().lower() == "all":
        return list(range(n_layers))

    indices: list[int] = []
    for token in raw_value.split(","):
        piece = token.strip()
        if not piece:
            continue
        if len(piece) == 1 and piece.isalpha():
            indices.append(ord(piece.lower()) - ord("a"))
        else:
            indices.append(int(piece))

    normalized = sorted(set(indices))
    for index in normalized:
        if index < 0 or index >= n_layers:
            raise ValueError(f"Layer index {index} is out of range for n_layers={n_layers}.")
    return normalized


def parse_torch_dtype(raw_value: str) -> torch.dtype | str:
    mapping = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if raw_value not in mapping:
        raise ValueError(f"Unsupported torch dtype: {raw_value}")
    return mapping[raw_value]


def build_initialized_rows(
    old_rows: torch.Tensor,
    layer_transport: pd.DataFrame,
    new_size: int,
    weight_mode: str,
    usage_power: float,
    distance_temperature: float,
) -> torch.Tensor:
    old_rows_float = old_rows.detach().float().cpu()
    new_rows = torch.empty((new_size, old_rows_float.shape[1]), dtype=old_rows_float.dtype)
    global_mean = old_rows_float.mean(dim=0)

    for target_code in range(new_size):
        bucket = layer_transport.loc[layer_transport["target_code"] == target_code]
        if bucket.empty:
            fallback_index = min(target_code, old_rows_float.shape[0] - 1)
            new_rows[target_code] = old_rows_float[fallback_index]
            continue

        teacher_indices = bucket["teacher_code"].to_numpy(dtype=np.int64)
        teacher_vectors = old_rows_float[teacher_indices]

        if weight_mode == "uniform":
            weights = np.ones(len(bucket), dtype=np.float64)
        else:
            weights = np.maximum(bucket["usage_count"].to_numpy(dtype=np.float64), 1.0) ** float(usage_power)
            if weight_mode == "inverse_distance":
                weights = weights * np.exp(
                    -bucket["distance_sq"].to_numpy(dtype=np.float64) / max(distance_temperature, 1e-8)
                )
            elif weight_mode != "usage":
                raise ValueError(f"Unsupported weight_mode: {weight_mode}")

        if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
            new_rows[target_code] = global_mean
            continue

        normalized = torch.as_tensor(weights / weights.sum(), dtype=teacher_vectors.dtype).unsqueeze(1)
        new_rows[target_code] = (teacher_vectors * normalized).sum(dim=0)

    return new_rows.to(dtype=old_rows.dtype)


@dataclass
class RowTrainingHandles:
    handles: list[Any]
    selected_token_ids: list[int]
    effective_trainable_elements: int


def configure_row_only_training(
    model: Any,
    selected_token_ids: list[int],
    train_input_embeddings: bool,
    train_lm_head: bool,
) -> RowTrainingHandles:
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    handles: list[Any] = []
    tracked_params: set[int] = set()
    effective_trainable_elements = 0

    def _attach_mask(weight: torch.nn.Parameter) -> None:
        nonlocal effective_trainable_elements
        param_id = id(weight)
        if param_id in tracked_params:
            return
        tracked_params.add(param_id)
        weight.requires_grad_(True)
        mask = torch.zeros_like(weight, dtype=weight.dtype)
        mask[selected_token_ids] = 1
        handles.append(weight.register_hook(lambda grad, local_mask=mask: grad * local_mask))
        effective_trainable_elements += int(len(selected_token_ids) * weight.shape[1])

    if train_input_embeddings:
        _attach_mask(model.get_input_embeddings().weight)

    if train_lm_head and model.get_output_embeddings() is not None:
        _attach_mask(model.get_output_embeddings().weight)

    if not handles:
        raise ValueError("No trainable rows were selected. Enable input embeddings and/or lm_head training.")

    return RowTrainingHandles(
        handles=handles,
        selected_token_ids=list(selected_token_ids),
        effective_trainable_elements=effective_trainable_elements,
    )


def clear_training_handles(row_handles: RowTrainingHandles) -> None:
    for handle in row_handles.handles:
        handle.remove()


def extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = str(item.get("text") or "").strip()
                if text:
                    parts.append(text)
            elif item:
                parts.append(str(item).strip())
        return "\n".join(part for part in parts if part)
    if content is None:
        return ""
    return str(content)


def normalize_messages(raw_messages: Any) -> list[dict[str, str]]:
    messages = raw_messages
    if isinstance(messages, str):
        messages = json.loads(messages)
    if not isinstance(messages, list):
        raise ValueError("Expected `messages` to be a list.")

    normalized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip()
        if not role:
            continue
        normalized.append({"role": role, "content": extract_text_content(message.get("content"))})
    if not normalized:
        raise ValueError("No valid messages remain after normalization.")
    return normalized


def split_prompt_completion(raw_messages: Any) -> tuple[list[dict[str, str]], str]:
    messages = normalize_messages(raw_messages)

    assistant_index = -1
    for index in range(len(messages) - 1, -1, -1):
        if messages[index]["role"] == "assistant":
            assistant_index = index
            break
    if assistant_index <= 0:
        raise ValueError("Failed to locate a terminal assistant message.")

    prompt_messages = messages[:assistant_index]
    completion_text = messages[assistant_index]["content"].strip()
    if not completion_text:
        raise ValueError("Assistant completion is empty.")
    return prompt_messages, completion_text


def render_prompt(tokenizer: Any, prompt_messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def build_prompt_completion_records(frame: pd.DataFrame, tokenizer: Any) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for row in frame.to_dict("records"):
        if "messages" in row:
            prompt_messages, completion_text = split_prompt_completion(row["messages"])
            prompt_text = render_prompt(tokenizer, prompt_messages)
            completion = f"{completion_text}{IM_END}"
        elif "prompt" in row and "completion" in row:
            prompt_text = str(row["prompt"])
            completion = str(row["completion"])
        else:
            raise ValueError("Training parquet must contain either `messages` or (`prompt`, `completion`).")
        records.append({"prompt": prompt_text, "completion": completion})
    return records


def encode_prompt_completion(
    tokenizer: Any,
    prompt_text: str,
    completion_text: str,
    max_length: int,
) -> dict[str, list[int]]:
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    completion_ids = tokenizer(completion_text, add_special_tokens=False).input_ids
    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids

    if len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
        labels = labels[-max_length:]
        if all(label == -100 for label in labels):
            raise ValueError("Example was truncated so aggressively that no completion labels remain.")

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


class PromptCompletionDataset(Dataset):
    def __init__(self, encoded_examples: list[dict[str, list[int]]]) -> None:
        self.examples = encoded_examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


class PromptCompletionCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        if self.pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id or eos_token_id.")

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        batch_input_ids: list[list[int]] = []
        batch_attention_mask: list[list[int]] = []
        batch_labels: list[list[int]] = []

        for feature in features:
            pad_length = max_length - len(feature["input_ids"])
            batch_input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad_length)
            batch_attention_mask.append(feature["attention_mask"] + [0] * pad_length)
            batch_labels.append(feature["labels"] + [-100] * pad_length)

        return {
            "input_ids": torch.as_tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.as_tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.as_tensor(batch_labels, dtype=torch.long),
        }
