#!/usr/bin/env python3
"""Prompted ranking baseline using a frozen instruction-tuned causal LM.

This script loads a single dataset JSON file, prompts the model to:
1) rank the candidate reasoning traces from best to worst, and
2) predict the final claim verdict.

Predictions are written in the same schema expected by `evaluate_clef_task2.py`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

try:
    from evaluate_clef_task2 import evaluate_predictions_against_dataset
    from task2_ranking_utils import (
        DEFAULT_MODEL_ID,
        build_prompt,
        derive_verdict_from_ranking,
        generate_pairwise_ranking,
        generate_prediction,
        infer_label_space,
        parse_response,
        resolve_dtype,
        set_seed,
    )
    from task2_utils import infer_language_name, load_json_rows
except ImportError:  # pragma: no cover - import path fallback
    from scripts.evaluate_clef_task2 import evaluate_predictions_against_dataset
    from scripts.task2_ranking_utils import (
        DEFAULT_MODEL_ID,
        build_prompt,
        derive_verdict_from_ranking,
        generate_pairwise_ranking,
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
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description="Prompted ranking baseline for CLEF Task 2.")
    parser.add_argument("--dataset-path", type=Path, required=True, help="Path to a dataset JSON file.")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--output", type=Path, required=True, help="Path to save predictions JSON.")
    parser.add_argument(
        "--language-name",
        type=str,
        default=None,
        help="Language identifier stored in prediction records. Defaults to an inferred name.",
    )
    parser.add_argument(
        "--variant-name",
        type=str,
        default=None,
        help="Optional variant name for metadata. Defaults to the dataset file stem.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Optional number of examples to run.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-evidence-items", type=int, default=3)
    parser.add_argument("--max-evidence-chars", type=int, default=1200)
    parser.add_argument("--max-trace-chars", type=int, default=1200)
    parser.add_argument(
        "--ranking-mode",
        choices=["listwise", "pairwise"],
        default="listwise",
        help="Ranking strategy. Pairwise compares all trace pairs and aggregates wins.",
    )
    parser.add_argument(
        "--pairwise-max-new-tokens",
        type=int,
        default=32,
        help="Max generated tokens for each pairwise comparison.",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=1,
        help="Number of prompts to generate per model call in pairwise mode.",
    )
    parser.add_argument(
        "--derive-verdict-from-ranking",
        action="store_true",
        help="Derive predicted_verdict from top-k ranked trace verdicts instead of model output.",
    )
    parser.add_argument(
        "--verdict-top-k",
        type=int,
        default=1,
        help="Number of top ranked traces used for derived verdict majority vote.",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--save-debug-fields", action="store_true")
    parser.add_argument("--evaluate", action="store_true", help="Run CLEF metric evaluation after prediction.")
    parser.add_argument("--eval-k", type=int, default=5)
    parser.add_argument("--eval-output", type=Path, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    rows = load_json_rows(args.dataset_path)
    selected_rows = rows[args.start_index :]
    if args.limit is not None:
        selected_rows = selected_rows[: args.limit]

    label_space = infer_label_space(rows)
    language_name = infer_output_language(args.dataset_path, args.language_name)
    variant_name = args.variant_name or args.dataset_path.stem

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=resolve_dtype(args.dtype),
        device_map=args.device_map,
    )
    model.eval()

    do_sample = args.do_sample or args.temperature > 0.0
    predictions: List[dict] = []

    for local_offset, row in enumerate(tqdm(selected_rows, desc="Running prompted baseline")):
        dataset_index = args.start_index + local_offset
        num_traces = len(row.get("Reasoning_traces", []) or [])

        raw_output = ""
        parsed = {
            "ranked_trace_indices": list(range(num_traces)),
            "predicted_verdict": "",
            "json_found": False,
            "json_parse_error": None,
        }
        pairwise_result = None
        if args.ranking_mode == "pairwise":
            pairwise_result = generate_pairwise_ranking(
                model=model,
                tokenizer=tokenizer,
                row=row,
                max_evidence_items=args.max_evidence_items,
                max_evidence_chars=args.max_evidence_chars,
                max_trace_chars=args.max_trace_chars,
                max_new_tokens=args.pairwise_max_new_tokens,
                do_sample=do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                keep_pairwise_details=args.save_debug_fields,
                batch_size=args.inference_batch_size,
            )
            ranked_trace_indices = pairwise_result["ranked_trace_indices"]
            predicted_verdict = derive_verdict_from_ranking(
                verdict_list=row.get("Verdict_list", []) or [],
                ranked_trace_indices=ranked_trace_indices,
                top_k=args.verdict_top_k,
                label_space=label_space,
            )
        else:
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
            ranked_trace_indices = parsed["ranked_trace_indices"]
            if args.derive_verdict_from_ranking or not parsed["predicted_verdict"]:
                predicted_verdict = derive_verdict_from_ranking(
                    verdict_list=row.get("Verdict_list", []) or [],
                    ranked_trace_indices=ranked_trace_indices,
                    top_k=args.verdict_top_k,
                    label_space=label_space,
                )
            else:
                predicted_verdict = parsed["predicted_verdict"]

        record = {
            "language": language_name,
            "dataset_index": dataset_index,
            "ranked_trace_indices": ranked_trace_indices,
            "predicted_verdict": predicted_verdict,
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
            if pairwise_result is not None:
                record.update(
                    {
                        "pairwise_scores": pairwise_result["pairwise_scores"],
                        "num_pairwise_comparisons": pairwise_result["num_pairwise_comparisons"],
                        "invalid_pairwise_outputs": pairwise_result["invalid_pairwise_outputs"],
                        "pairwise_details": pairwise_result.get("pairwise_details", []),
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
                "model_id": args.model_id,
                "dataset_path": str(args.dataset_path),
                "language_name": language_name,
                "variant_name": variant_name,
                "num_predictions": len(predictions),
                "start_index": args.start_index,
                "limit": args.limit,
                "ranking_mode": args.ranking_mode,
                "derive_verdict_from_ranking": args.derive_verdict_from_ranking or args.ranking_mode == "pairwise",
                "verdict_top_k": args.verdict_top_k,
                "inference_batch_size": args.inference_batch_size,
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
        eval_output = args.eval_output
        if eval_output is not None:
            eval_output.parent.mkdir(parents=True, exist_ok=True)
            with eval_output.open("w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
