#!/usr/bin/env python3
"""Run inference with a trained SFT + LoRA adapter for CLEF Task 2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

try:
    from evaluate_clef_task2 import evaluate_predictions_against_dataset
    from hf_quantization_utils import add_4bit_loading_args, build_4bit_quantization_config
    from task2_ranking_utils import (
        DEFAULT_MODEL_ID,
        build_prompt,
        generate_prediction,
        infer_label_space,
        parse_response,
        resolve_dtype,
        set_seed,
    )
    from task2_utils import infer_language_name, load_json_rows
except ImportError:  # pragma: no cover - import path fallback
    from scripts.evaluate_clef_task2 import evaluate_predictions_against_dataset
    from scripts.hf_quantization_utils import add_4bit_loading_args, build_4bit_quantization_config
    from scripts.task2_ranking_utils import (
        DEFAULT_MODEL_ID,
        build_prompt,
        generate_prediction,
        infer_label_space,
        parse_response,
        resolve_dtype,
        set_seed,
    )
    from scripts.task2_utils import infer_language_name, load_json_rows


def infer_output_language(dataset_path: Path, override: Optional[str]) -> str:
    if override:
        return override.strip().lower()
    return infer_language_name(dataset_path)


def main() -> None:
    from peft import PeftModel
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description="Run inference with a trained SFT + LoRA adapter.")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language-name", type=str, default=None)
    parser.add_argument("--variant-name", type=str, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-evidence-items", type=int, default=3)
    parser.add_argument("--max-evidence-chars", type=int, default=1200)
    parser.add_argument("--max-trace-chars", type=int, default=1200)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--attn-implementation", type=str, default="sdpa", choices=["auto", "sdpa", "eager"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--save-debug-fields", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--eval-k", type=int, default=5)
    parser.add_argument("--eval-output", type=Path, default=None)
    add_4bit_loading_args(parser)
    args = parser.parse_args()

    set_seed(args.seed)
    rows = load_json_rows(args.dataset_path)
    selected_rows = rows[args.start_index :]
    if args.limit is not None:
        selected_rows = selected_rows[: args.limit]

    label_space = infer_label_space(rows)
    language_name = infer_output_language(args.dataset_path, args.language_name)
    variant_name = args.variant_name or args.adapter_path.name

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    resolved_dtype = resolve_dtype(args.dtype)
    quantization_config = build_4bit_quantization_config(
        load_in_4bit=args.load_in_4bit,
        quant_type=args.bnb_4bit_quant_type,
        compute_dtype_name=args.bnb_4bit_compute_dtype,
        use_double_quant=args.bnb_4bit_use_double_quant,
        fallback_dtype=resolved_dtype,
    )

    model_kwargs = {
        "torch_dtype": resolved_dtype,
        "device_map": args.device_map,
        "low_cpu_mem_usage": True,
    }
    if args.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = args.attn_implementation
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    base_model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    model = PeftModel.from_pretrained(base_model, str(args.adapter_path))
    model.eval()

    predictions: List[dict] = []
    do_sample = args.do_sample or args.temperature > 0.0
    for local_offset, row in enumerate(tqdm(selected_rows, desc="Running SFT LoRA inference")):
        dataset_index = args.start_index + local_offset
        num_traces = len(row.get("Reasoning_traces", []) or [])

        prompt = build_prompt(
            row=row,
            label_space=label_space,
            max_evidence_items=args.max_evidence_items,
            max_evidence_chars=args.max_evidence_chars,
            max_trace_chars=args.max_trace_chars,
        )
        raw_output = generate_prediction(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        parsed = parse_response(raw_output, label_space=label_space, num_traces=num_traces)
        record = {
            "language": language_name,
            "dataset_index": dataset_index,
            "ranked_trace_indices": parsed["ranked_trace_indices"],
            "predicted_verdict": parsed["predicted_verdict"],
            "variant": variant_name,
            "group_id": dataset_index,
        }
        if args.save_debug_fields:
            record.update(
                {
                    "claim": row.get("claim", ""),
                    "raw_model_output": raw_output,
                    "json_found": parsed["json_found"],
                    "json_parse_error": parsed["json_parse_error"],
                }
            )
        predictions.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(
        json.dumps(
            {
                "saved_predictions": str(args.output),
                "adapter_path": str(args.adapter_path),
                "model_id": args.model_id,
                "dataset_path": str(args.dataset_path),
                "language_name": language_name,
                "variant_name": variant_name,
                "num_predictions": len(predictions),
                "start_index": args.start_index,
                "limit": args.limit,
                "load_in_4bit": args.load_in_4bit,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    if args.evaluate:
        metrics = evaluate_predictions_against_dataset(
            predictions_path=args.output,
            k=args.eval_k,
            dataset_path=args.dataset_path,
            language_name=language_name,
            start_index=args.start_index,
            limit=args.limit,
        )
        if args.eval_output is not None:
            args.eval_output.parent.mkdir(parents=True, exist_ok=True)
            with args.eval_output.open("w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
