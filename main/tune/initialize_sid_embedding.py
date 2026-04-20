from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sid_alignment_utils import (
    build_initialized_rows,
    infer_layer_sizes_from_item_sid_map,
    infer_layer_sizes_from_transport,
    load_transport_map,
    parse_torch_dtype,
    resolve_token_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize SID token embedding rows for a distilled codebook using teacher-to-target transport.",
    )
    parser.add_argument("--model_path", required=True, help="Base OpenOneRec HuggingFace checkpoint.")
    parser.add_argument("--transport_map", required=True, help="transport_map.parquet from distill_codebook.")
    parser.add_argument("--item_sid_map", required=True, help="distilled_item_sid_map.parquet to infer new layer sizes.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the initialized checkpoint.")
    parser.add_argument(
        "--weight_mode",
        choices=["usage", "uniform", "inverse_distance"],
        default="usage",
        help="How old teacher token rows contribute to each new SID row.",
    )
    parser.add_argument("--usage_power", type=float, default=1.0, help="Exponent applied to usage_count weights.")
    parser.add_argument(
        "--distance_temperature",
        type=float,
        default=1.0,
        help="Temperature for inverse-distance weighting. Only used when weight_mode=inverse_distance.",
    )
    parser.add_argument(
        "--overwrite_lm_head",
        action="store_true",
        help="Also overwrite the lm_head rows when embeddings are not tied.",
    )
    parser.add_argument("--torch_dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=parse_torch_dtype(args.torch_dtype),
        trust_remote_code=True,
    )
    model.eval()

    transport_map = load_transport_map(args.transport_map)
    old_layer_sizes, inferred_new_layer_sizes = infer_layer_sizes_from_transport(transport_map)
    new_layer_sizes = infer_layer_sizes_from_item_sid_map(args.item_sid_map)
    if len(new_layer_sizes) != len(old_layer_sizes):
        raise ValueError(
            f"Layer count mismatch: transport has {len(old_layer_sizes)} layers but item_sid_map has {len(new_layer_sizes)}."
        )

    for layer_index, (transport_size, sid_size) in enumerate(zip(inferred_new_layer_sizes, new_layer_sizes, strict=True)):
        if transport_size != sid_size:
            raise ValueError(
                f"Layer {layer_index} size mismatch: transport infers {transport_size} but item_sid_map infers {sid_size}."
            )

    old_token_ids = resolve_token_ids(tokenizer, old_layer_sizes)
    new_token_ids = resolve_token_ids(tokenizer, new_layer_sizes)

    input_weight = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings()
    output_weight = None if output_embeddings is None else output_embeddings.weight.data

    metadata_layers: list[dict[str, int | str]] = []

    for layer_index in range(len(old_layer_sizes)):
        layer_transport = transport_map.loc[transport_map["layer"] == layer_index].copy()
        old_rows = input_weight[old_token_ids[layer_index]]
        initialized_rows = build_initialized_rows(
            old_rows=old_rows,
            layer_transport=layer_transport,
            new_size=new_layer_sizes[layer_index],
            weight_mode=args.weight_mode,
            usage_power=args.usage_power,
            distance_temperature=args.distance_temperature,
        ).to(device=input_weight.device, dtype=input_weight.dtype)

        input_weight[new_token_ids[layer_index]] = initialized_rows

        if args.overwrite_lm_head and output_weight is not None and output_weight.data_ptr() != input_weight.data_ptr():
            output_weight[new_token_ids[layer_index]] = initialized_rows.to(
                device=output_weight.device,
                dtype=output_weight.dtype,
            )

        metadata_layers.append(
            {
                "layer": layer_index,
                "old_size": old_layer_sizes[layer_index],
                "new_size": new_layer_sizes[layer_index],
                "num_rows_overwritten": len(new_token_ids[layer_index]),
            }
        )

    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)

    summary = {
        "base_model_path": str(Path(args.model_path).resolve()),
        "transport_map": str(Path(args.transport_map).resolve()),
        "item_sid_map": str(Path(args.item_sid_map).resolve()),
        "weight_mode": args.weight_mode,
        "usage_power": args.usage_power,
        "distance_temperature": args.distance_temperature,
        "overwrite_lm_head": bool(args.overwrite_lm_head),
        "layers": metadata_layers,
        "output_dir": str(output_dir.resolve()),
    }
    summary_path = output_dir / "sid_embedding_init_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
