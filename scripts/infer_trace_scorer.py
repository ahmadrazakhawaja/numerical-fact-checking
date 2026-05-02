#!/usr/bin/env python3
"""Run inference with a trained binary trace scorer adapter."""

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
    from numeric_embedding_utils import (
        add_numeric_embedding_args,
        ensure_numeric_token,
        resize_model_embeddings_if_needed,
        tokenizer_source_for_adapter,
        wrap_model_with_numeric_embeddings,
    )
    from task2_ranking_utils import DEFAULT_MODEL_ID, derive_verdict_from_ranking, get_model_input_device, resolve_dtype, set_seed
    from task2_utils import infer_language_name, load_json_rows
    from training_utils import add_reporting_args, configure_reporting, log_wandb_metrics
    from trace_scorer_utils import (
        TraceScorerDataCollator,
        build_trace_scorer_input_artifacts,
        logits_to_trace_scores,
        rank_trace_indices_from_scores,
        tokenize_trace_scorer_input,
    )
except ImportError:  # pragma: no cover - import path fallback
    from scripts.evaluate_clef_task2 import evaluate_predictions_against_dataset
    from scripts.hf_quantization_utils import add_4bit_loading_args, build_4bit_quantization_config
    from scripts.numeric_embedding_utils import (
        add_numeric_embedding_args,
        ensure_numeric_token,
        resize_model_embeddings_if_needed,
        tokenizer_source_for_adapter,
        wrap_model_with_numeric_embeddings,
    )
    from scripts.task2_ranking_utils import (
        DEFAULT_MODEL_ID,
        derive_verdict_from_ranking,
        get_model_input_device,
        resolve_dtype,
        set_seed,
    )
    from scripts.task2_utils import infer_language_name, load_json_rows
    from scripts.training_utils import add_reporting_args, configure_reporting, log_wandb_metrics
    from scripts.trace_scorer_utils import (
        TraceScorerDataCollator,
        build_trace_scorer_input_artifacts,
        logits_to_trace_scores,
        rank_trace_indices_from_scores,
        tokenize_trace_scorer_input,
    )


def infer_output_language(dataset_path: Path, override: Optional[str]) -> str:
    if override:
        return override.strip().lower()
    return infer_language_name(dataset_path)


def score_features_in_batches(model, collator, features: List[dict], batch_size: int) -> List[float]:
    scores: List[float] = []
    device = get_model_input_device(model)

    for start in range(0, len(features), batch_size):
        batch = collator(features[start : start + batch_size])
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        batch_scores = logits_to_trace_scores(outputs.logits.detach()).cpu().tolist()
        scores.extend(float(score) for score in batch_scores)
    return scores


def build_submission_record(
    row: dict,
    dataset_index: int,
    score_list: List[float],
    predicted_verdict: str,
    *,
    max_evidence_items: int,
    max_evidence_chars: int,
) -> dict:
    cleaned_traces = [
        build_trace_scorer_input_artifacts(
            row,
            trace_index,
            max_claim_chars=10_000,
            max_evidence_items=max_evidence_items,
            max_evidence_chars=max_evidence_chars,
            max_trace_chars=100_000,
            use_numeric_embedding=False,
            append_normalized_numbers=False,
        )["cleaned_trace"]
        for trace_index in range(len(row.get("Reasoning_traces", []) or []))
    ]
    return {
        "query_id": dataset_index,
        "Claim": row.get("claim", ""),
        "Verdict_BoN": predicted_verdict,
        "BoN_Verdict_list": row.get("Verdict_list", []) or [],
        "Reasoning_traces": cleaned_traces,
        "score_list": score_list,
    }


def main() -> None:
    from peft import PeftModel
    from tqdm import tqdm
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    parser = argparse.ArgumentParser(description="Run inference with a binary trace scorer adapter.")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--output-format",
        type=str,
        default="submission",
        choices=["submission", "internal"],
        help="Format for --output. 'submission' matches the CodaLab JSON schema; 'internal' keeps the local debug/eval schema.",
    )
    parser.add_argument(
        "--internal-output",
        type=Path,
        default=None,
        help="Optional path for the old internal prediction schema when --output-format=submission.",
    )
    parser.add_argument("--language-name", type=str, default=None)
    parser.add_argument("--variant-name", type=str, default=None)
    parser.add_argument(
        "--supervisor-output",
        type=Path,
        default=None,
        help="Deprecated alias for writing an additional submission-format JSON.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument(
        "--scorer-head",
        type=str,
        default="ce",
        choices=["ce", "bce"],
        help="Head used when training the adapter: 2-logit CE ('ce') or 1-logit BCE ('bce').",
    )
    parser.add_argument("--max-claim-chars", type=int, default=600)
    parser.add_argument("--max-evidence-items", type=int, default=2)
    parser.add_argument("--max-evidence-chars", type=int, default=512)
    parser.add_argument("--max-trace-chars", type=int, default=1800)
    parser.add_argument(
        "--append-normalized-numbers",
        action="store_true",
        help="Append compact plain-text normalized numeric hints to each scorer input.",
    )
    parser.add_argument(
        "--max-normalized-numbers",
        type=int,
        default=20,
        help="Maximum normalized numeric hints to append per claim/evidence/trace section.",
    )
    parser.add_argument("--scoring-batch-size", type=int, default=8)
    parser.add_argument("--verdict-top-k", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--attn-implementation", type=str, default="sdpa", choices=["auto", "sdpa", "eager"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-debug-fields", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--eval-k", type=int, default=5)
    parser.add_argument("--eval-output", type=Path, default=None)
    add_reporting_args(parser)
    add_4bit_loading_args(parser)
    add_numeric_embedding_args(parser)
    args = parser.parse_args()

    if args.scoring_batch_size <= 0:
        raise ValueError(f"scoring_batch_size must be > 0, got {args.scoring_batch_size}")

    set_seed(args.seed)
    configure_reporting(args, args.output.parent.resolve())
    rows = load_json_rows(args.dataset_path)
    selected_rows = rows[args.start_index :]
    if args.limit is not None:
        selected_rows = selected_rows[: args.limit]

    language_name = infer_output_language(args.dataset_path, args.language_name)
    variant_name = args.variant_name or args.adapter_path.name

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source_for_adapter(args.model_id, args.adapter_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.use_numeric_embedding:
        ensure_numeric_token(tokenizer)

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
        "num_labels": 1 if args.scorer_head == "bce" else 2,
    }
    if args.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = args.attn_implementation
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    base_model = AutoModelForSequenceClassification.from_pretrained(args.model_id, **model_kwargs)
    if args.scorer_head == "bce":
        base_model.config.problem_type = "multi_label_classification"
    resize_model_embeddings_if_needed(base_model, tokenizer)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base_model, str(args.adapter_path))
    model, _ = wrap_model_with_numeric_embeddings(
        model,
        tokenizer,
        artifact_path=args.adapter_path,
        enable_numeric_embedding=args.use_numeric_embedding,
        max_numeric_chars=args.max_numeric_chars,
        numeric_char_embedding_dim=args.numeric_char_embedding_dim,
        numeric_gru_hidden_size=args.numeric_gru_hidden_size,
        freeze_value_encoder=True,
    )
    model.eval()

    collator = TraceScorerDataCollator(pad_token_id=tokenizer.pad_token_id)
    internal_predictions: List[dict] = []
    submission_predictions: List[dict] = []
    num_token_id = tokenizer.convert_tokens_to_ids("<num>") if args.use_numeric_embedding else None

    for local_offset, row in enumerate(tqdm(selected_rows, desc="Running trace scorer inference")):
        dataset_index = args.start_index + local_offset
        traces = row.get("Reasoning_traces", []) or []
        encoded_features: List[dict] = []
        cleaned_traces: List[str] = []

        for trace_index in range(len(traces)):
            artifacts = build_trace_scorer_input_artifacts(
                row,
                trace_index,
                max_claim_chars=args.max_claim_chars,
                max_evidence_items=args.max_evidence_items,
                max_evidence_chars=args.max_evidence_chars,
                max_trace_chars=args.max_trace_chars,
                use_numeric_embedding=args.use_numeric_embedding,
                append_normalized_numbers=args.append_normalized_numbers,
                max_normalized_numbers=args.max_normalized_numbers,
            )
            cleaned_traces.append(str(artifacts["cleaned_trace"]))
            encoded = tokenize_trace_scorer_input(
                tokenizer=tokenizer,
                text=str(artifacts["text"]),
                label=None,
                max_length=args.max_length,
                numeric_canonicals=artifacts["numeric_canonicals"] if args.use_numeric_embedding else None,
                num_token_id=num_token_id,
                max_numeric_chars=args.max_numeric_chars,
            )
            encoded_features.append(encoded)

        score_list = score_features_in_batches(
            model=model,
            collator=collator,
            features=encoded_features,
            batch_size=args.scoring_batch_size,
        )
        ranked_trace_indices = rank_trace_indices_from_scores(score_list)
        predicted_verdict = derive_verdict_from_ranking(
            verdict_list=row.get("Verdict_list", []) or [],
            ranked_trace_indices=ranked_trace_indices,
            top_k=args.verdict_top_k,
            label_space=sorted({str(x) for x in (row.get("Verdict_list", []) or []) if str(x).strip()}),
        )

        internal_record = {
            "language": language_name,
            "dataset_index": dataset_index,
            "ranked_trace_indices": ranked_trace_indices,
            "predicted_verdict": predicted_verdict,
            "score_list": score_list,
            "variant": variant_name,
            "group_id": dataset_index,
        }
        if args.save_debug_fields:
            internal_record["claim"] = row.get("claim", "")
            internal_record["cleaned_reasoning_traces"] = cleaned_traces
        internal_predictions.append(internal_record)

        submission_predictions.append(
            build_submission_record(
                row=row,
                dataset_index=dataset_index,
                score_list=score_list,
                predicted_verdict=predicted_verdict,
                max_evidence_items=args.max_evidence_items,
                max_evidence_chars=args.max_evidence_chars,
            )
        )

    predictions = submission_predictions if args.output_format == "submission" else internal_predictions
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    if args.internal_output is not None:
        args.internal_output.parent.mkdir(parents=True, exist_ok=True)
        with args.internal_output.open("w", encoding="utf-8") as f:
            json.dump(internal_predictions, f, indent=2, ensure_ascii=False)

    if args.supervisor_output is not None:
        args.supervisor_output.parent.mkdir(parents=True, exist_ok=True)
        with args.supervisor_output.open("w", encoding="utf-8") as f:
            json.dump(submission_predictions, f, indent=2, ensure_ascii=False)

    print(
        json.dumps(
            {
                "saved_predictions": str(args.output),
                "output_format": args.output_format,
                "internal_output": None if args.internal_output is None else str(args.internal_output),
                "supervisor_output": None if args.supervisor_output is None else str(args.supervisor_output),
                "adapter_path": str(args.adapter_path),
                "model_id": args.model_id,
                "scorer_head": args.scorer_head,
                "dataset_path": str(args.dataset_path),
                "language_name": language_name,
                "variant_name": variant_name,
                "num_predictions": len(predictions),
                "start_index": args.start_index,
                "limit": args.limit,
                "max_evidence_items": args.max_evidence_items,
                "max_evidence_chars": args.max_evidence_chars,
                "append_normalized_numbers": args.append_normalized_numbers,
                "max_normalized_numbers": args.max_normalized_numbers,
                "load_in_4bit": args.load_in_4bit,
                "use_numeric_embedding": args.use_numeric_embedding,
                "verdict_top_k": args.verdict_top_k,
                "scoring_batch_size": args.scoring_batch_size,
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
        log_wandb_metrics(
            args,
            args.output.parent.resolve(),
            metrics,
            prefix=f"task2/{language_name}",
            config={
                "adapter_path": str(args.adapter_path),
                "model_id": args.model_id,
                "dataset_path": str(args.dataset_path),
                "scorer_head": args.scorer_head,
                "language_name": language_name,
                "variant_name": variant_name,
                "output": str(args.output),
                "eval_k": args.eval_k,
                "max_evidence_items": args.max_evidence_items,
                "max_evidence_chars": args.max_evidence_chars,
                "append_normalized_numbers": args.append_normalized_numbers,
                "max_normalized_numbers": args.max_normalized_numbers,
                "verdict_top_k": args.verdict_top_k,
                "scoring_batch_size": args.scoring_batch_size,
                "load_in_4bit": args.load_in_4bit,
                "use_numeric_embedding": args.use_numeric_embedding,
            },
        )
        print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
