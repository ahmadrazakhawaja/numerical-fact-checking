#!/usr/bin/env python3
"""Train a QLoRA trace scorer baseline for CLEF Task 2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

try:
    from hf_quantization_utils import (
        add_4bit_loading_args,
        build_4bit_quantization_config,
        prepare_model_for_4bit_training,
    )
    from numeric_embedding_utils import (
        add_numeric_embedding_args,
        ensure_numeric_token,
        resize_model_embeddings_if_needed,
        wrap_model_with_numeric_embeddings,
    )
    from task2_ranking_utils import DEFAULT_MODEL_ID, derive_verdict_from_ranking, resolve_dtype, set_seed
    from task2_utils import load_json_rows, normalize_label, safe_div
    from trace_scorer_utils import (
        TraceScorerDataCollator,
        TokenizedTraceScorerDataset,
        average_sequence_length,
        build_trace_scorer_features,
        stats_as_dict,
    )
    from training_utils import (
        build_training_arguments,
        add_reporting_args,
        configure_reporting,
        maybe_limit_rows,
        parse_target_modules,
        resolve_attn_implementation,
        resolve_device_map_arg,
        save_run_metadata,
    )
except ImportError:  # pragma: no cover - import path fallback
    from scripts.hf_quantization_utils import (
        add_4bit_loading_args,
        build_4bit_quantization_config,
        prepare_model_for_4bit_training,
    )
    from scripts.numeric_embedding_utils import (
        add_numeric_embedding_args,
        ensure_numeric_token,
        resize_model_embeddings_if_needed,
        wrap_model_with_numeric_embeddings,
    )
    from scripts.task2_ranking_utils import DEFAULT_MODEL_ID, derive_verdict_from_ranking, resolve_dtype, set_seed
    from scripts.task2_utils import load_json_rows, normalize_label, safe_div
    from scripts.trace_scorer_utils import (
        TraceScorerDataCollator,
        TokenizedTraceScorerDataset,
        average_sequence_length,
        build_trace_scorer_features,
        stats_as_dict,
    )
    from scripts.training_utils import (
        build_training_arguments,
        add_reporting_args,
        configure_reporting,
        maybe_limit_rows,
        parse_target_modules,
        resolve_attn_implementation,
        resolve_device_map_arg,
        save_run_metadata,
    )


def build_trainer(
    Trainer,
    model,
    training_args,
    train_dataset,
    eval_dataset,
    data_collator,
    tokenizer,
    callbacks=None,
    compute_metrics_fn=None,
):
    common_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics_fn or compute_binary_metrics,
    }
    if callbacks:
        common_kwargs["callbacks"] = callbacks
    try:
        return Trainer(processing_class=tokenizer, **common_kwargs)
    except TypeError:
        return Trainer(tokenizer=tokenizer, **common_kwargs)


def compute_binary_metrics(eval_prediction) -> dict[str, float]:
    logits = eval_prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    labels = eval_prediction.label_ids
    predicted = np.asarray(logits).argmax(axis=-1)
    labels = np.asarray(labels)

    accuracy = float((predicted == labels).mean()) if len(labels) else 0.0
    true_positive = int(((predicted == 1) & (labels == 1)).sum())
    false_positive = int(((predicted == 1) & (labels == 0)).sum())
    false_negative = int(((predicted == 0) & (labels == 1)).sum())
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "accuracy": accuracy,
        "positive_precision": precision,
        "positive_recall": recall,
        "positive_f1": f1,
    }


def logits_to_scores(logits) -> np.ndarray:
    logits = np.asarray(logits)
    if logits.ndim == 1:
        return logits
    if logits.shape[-1] == 1:
        return logits[:, 0]
    if logits.shape[-1] == 2:
        return logits[:, 1] - logits[:, 0]
    raise ValueError(f"Unsupported scorer logits shape: {tuple(logits.shape)}")


def f1_scores(y_true: list[str], y_pred: list[str], labels: list[str]) -> tuple[float, dict[str, float]]:
    classwise: dict[str, float] = {}
    for label in labels:
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred == label)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != label and pred == label)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred != label)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        classwise[label] = safe_div(2 * precision * recall, precision + recall)

    return safe_div(sum(classwise.values()), len(labels)), classwise


def recall_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for idx in ranked[:k] if idx in relevant) / len(relevant)


def build_trace_and_task2_metrics(eval_features: list[dict], eval_k: int, verdict_top_k: int):
    def compute_metrics(eval_prediction) -> dict[str, float]:
        metrics = compute_binary_metrics(eval_prediction)
        logits = eval_prediction.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        scores = logits_to_scores(logits)

        grouped: dict[int, list[tuple[int, float]]] = {}
        claim_metadata: dict[int, dict[str, object]] = {}
        for feature, score in zip(eval_features, scores):
            dataset_index = int(feature["dataset_index"])
            trace_index = int(feature["trace_index"])
            grouped.setdefault(dataset_index, []).append((trace_index, float(score)))
            claim_metadata.setdefault(
                dataset_index,
                {
                    "gold_label": normalize_label(feature.get("gold_label", "")),
                    "verdict_list": [normalize_label(v) for v in feature.get("verdict_list", [])],
                    "num_traces": int(feature.get("num_traces", 0)),
                },
            )

        ranking_recall: list[float] = []
        y_true: list[str] = []
        y_pred: list[str] = []
        for dataset_index, metadata in sorted(claim_metadata.items()):
            num_traces = int(metadata["num_traces"])
            verdict_list = list(metadata["verdict_list"])
            gold_label = str(metadata["gold_label"])
            scored = sorted(grouped.get(dataset_index, []), key=lambda item: (-item[1], item[0]))
            ranked = [trace_index for trace_index, _ in scored]
            seen = set(ranked)
            ranked.extend(idx for idx in range(num_traces) if idx not in seen)

            relevant = {idx for idx, verdict in enumerate(verdict_list) if verdict == gold_label}
            ranking_recall.append(recall_at_k(ranked, relevant, k=eval_k))
            y_true.append(gold_label)
            y_pred.append(
                normalize_label(
                    derive_verdict_from_ranking(
                        verdict_list=verdict_list,
                        ranked_trace_indices=ranked,
                        top_k=verdict_top_k,
                        label_space=sorted({label for label in verdict_list if label}),
                    )
                )
            )

        labels = sorted({label for label in y_true if label})
        macro_f1, classwise_f1 = f1_scores(y_true, y_pred, labels) if labels else (0.0, {})
        recall = safe_div(sum(ranking_recall), len(ranking_recall))
        metrics.update(
            {
                "macro_f1": macro_f1,
                "recall_at_k": recall,
                "macro_f1_recall_at_k_mean": 0.5 * (macro_f1 + recall),
            }
        )
        for label, score in classwise_f1.items():
            metrics[f"classwise_f1_{label}"] = score
        return metrics

    return compute_metrics


class SaveBestAdapterCallback:
    def __init__(
        self,
        *,
        output_dir: Path,
        tokenizer,
        metric_name: str,
        greater_is_better: bool,
        improvement_threshold: float = 0.0,
        adapter_dir_name: str = "final_adapter",
    ) -> None:
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.improvement_threshold = improvement_threshold
        self.adapter_dir_name = adapter_dir_name
        self.best_metric: float | None = None
        self.best_step: int | None = None
        self.saved_best_adapter = False
        self._missing_metric_reported = False

    def _metric_keys(self) -> list[str]:
        stripped = self.metric_name
        if stripped.startswith("eval_"):
            return [stripped, stripped.removeprefix("eval_")]
        return [f"eval_{stripped}", stripped]

    def _is_improved(self, value: float) -> bool:
        if self.best_metric is None:
            return True
        if self.greater_is_better:
            return value > self.best_metric + self.improvement_threshold
        return value < self.best_metric - self.improvement_threshold

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        if not getattr(state, "is_world_process_zero", True):
            return control
        if model is None or not metrics:
            return control

        metric_value = None
        metric_key = None
        for key in self._metric_keys():
            if key in metrics:
                metric_key = key
                metric_value = float(metrics[key])
                break

        if metric_value is None:
            if not self._missing_metric_reported:
                print(
                    f"Best-adapter save skipped: metric '{self.metric_name}' was not present in eval metrics. "
                    f"Available metrics: {sorted(metrics)}"
                )
                self._missing_metric_reported = True
            return control

        if not self._is_improved(metric_value):
            return control

        self.best_metric = metric_value
        self.best_step = int(getattr(state, "global_step", 0))
        adapter_dir = self.output_dir / self.adapter_dir_name
        model.save_pretrained(adapter_dir)
        self.tokenizer.save_pretrained(adapter_dir)
        payload = {
            "best_metric": self.best_metric,
            "best_metric_name": metric_key,
            "best_step": self.best_step,
            "epoch": getattr(state, "epoch", None),
        }
        with (adapter_dir / "best_adapter_state.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        self.saved_best_adapter = True
        print(
            f"Saved new best adapter to {adapter_dir} "
            f"({metric_key}={self.best_metric:.6f}, step={self.best_step})."
        )
        return control


def main() -> None:
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback, Trainer, TrainingArguments

    parser = argparse.ArgumentParser(description="Train a binary trace scorer with QLoRA.")
    parser.add_argument("--train-dataset", type=Path, default=Path("dataset/english/train_complete.json"))
    parser.add_argument("--validation-dataset", type=Path, default=Path("dataset/english/validation_complete.json"))
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-validation", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=1024)
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
    parser.add_argument("--eval-k", type=int, default=5, help="k for validation Recall@k in Task2-style metrics.")
    parser.add_argument(
        "--verdict-top-k",
        type=int,
        default=5,
        help="Top-k ranked trace verdicts used for validation Task2-style verdict voting.",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--device-map", type=str, default="none", choices=["none", "auto"])
    parser.add_argument("--attn-implementation", type=str, default="sdpa", choices=["auto", "sdpa", "eager"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--eval-strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help='Resume Trainer state from a checkpoint path, or use "latest" to auto-detect the latest checkpoint.',
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after this many evaluations without improvement. Set 0 to disable.",
    )
    parser.add_argument(
        "--early-stopping-threshold",
        type=float,
        default=0.0,
        help="Minimum metric improvement required to reset early-stopping patience.",
    )
    parser.add_argument(
        "--metric-for-best-model",
        type=str,
        default="macro_f1_recall_at_k_mean",
        help="Evaluation metric used for early stopping and best-checkpoint loading.",
    )
    parser.add_argument(
        "--load-best-model-at-end",
        action="store_true",
        help="Reload the best checkpoint before saving final_adapter. Enabled automatically with early stopping.",
    )
    parser.add_argument(
        "--save-best-adapter-at-each-eval",
        action="store_true",
        help="Update final_adapter immediately after each evaluation when metric_for_best_model improves.",
    )
    metric_direction = parser.add_mutually_exclusive_group()
    metric_direction.add_argument(
        "--greater-is-better",
        dest="greater_is_better",
        action="store_true",
        help="Treat larger metric values as better.",
    )
    metric_direction.add_argument(
        "--lower-is-better",
        dest="greater_is_better",
        action="store_false",
        help="Treat smaller metric values as better.",
    )
    ddp_find_unused_group = parser.add_mutually_exclusive_group()
    ddp_find_unused_group.add_argument(
        "--ddp-find-unused-parameters",
        dest="ddp_find_unused_parameters",
        action="store_true",
        help="Enable DDP unused-parameter detection. Usually slower; only use if DDP reports unused parameters.",
    )
    ddp_find_unused_group.add_argument(
        "--no-ddp-find-unused-parameters",
        dest="ddp_find_unused_parameters",
        action="store_false",
        help="Disable DDP unused-parameter detection for faster multi-GPU training.",
    )
    parser.add_argument("--gradient-checkpointing", dest="gradient_checkpointing", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        type=str,
        default="all-linear",
        help='Comma-separated module names or "all-linear" for QLoRA-style coverage.',
    )
    parser.add_argument("--optim", type=str, default="adamw_torch")
    add_reporting_args(parser)
    add_4bit_loading_args(parser)
    add_numeric_embedding_args(parser)
    parser.set_defaults(gradient_checkpointing=True, greater_is_better=True, ddp_find_unused_parameters=False)
    args = parser.parse_args()

    if args.early_stopping_patience < 0:
        raise ValueError(f"early_stopping_patience must be >= 0, got {args.early_stopping_patience}")
    if args.early_stopping_threshold < 0:
        raise ValueError(f"early_stopping_threshold must be >= 0, got {args.early_stopping_threshold}")
    if args.early_stopping_patience > 0:
        args.load_best_model_at_end = True
    if args.load_best_model_at_end:
        if args.eval_strategy == "no":
            raise ValueError("Best-checkpoint loading/early stopping requires --eval-strategy steps or epoch.")
        if args.save_strategy == "no":
            raise ValueError("Best-checkpoint loading/early stopping requires --save-strategy steps or epoch.")
        if args.eval_strategy != args.save_strategy:
            raise ValueError(
                "Best-checkpoint loading/early stopping requires matching --eval-strategy and --save-strategy."
            )

    set_seed(args.seed)
    output_dir = args.output_dir.resolve()
    configure_reporting(args, output_dir)
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.is_available() and world_size > 1 and local_rank >= 0:
        torch.cuda.set_device(local_rank)

    train_rows = maybe_limit_rows(load_json_rows(args.train_dataset), args.limit_train)
    validation_rows = maybe_limit_rows(load_json_rows(args.validation_dataset), args.limit_validation)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.use_numeric_embedding:
        ensure_numeric_token(tokenizer)

    train_features, train_stats = build_trace_scorer_features(
        train_rows,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_claim_chars=args.max_claim_chars,
        max_evidence_items=args.max_evidence_items,
        max_evidence_chars=args.max_evidence_chars,
        max_trace_chars=args.max_trace_chars,
        use_numeric_embedding=args.use_numeric_embedding,
        max_numeric_chars=args.max_numeric_chars,
        append_normalized_numbers=args.append_normalized_numbers,
        max_normalized_numbers=args.max_normalized_numbers,
    )
    validation_features, validation_stats = build_trace_scorer_features(
        validation_rows,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_claim_chars=args.max_claim_chars,
        max_evidence_items=args.max_evidence_items,
        max_evidence_chars=args.max_evidence_chars,
        max_trace_chars=args.max_trace_chars,
        use_numeric_embedding=args.use_numeric_embedding,
        max_numeric_chars=args.max_numeric_chars,
        append_normalized_numbers=args.append_normalized_numbers,
        max_normalized_numbers=args.max_normalized_numbers,
    )

    if not train_features:
        raise ValueError("No scorer training examples remained after preprocessing.")
    if not validation_features and args.eval_strategy != "no":
        raise ValueError("No scorer validation examples remained after preprocessing.")

    resolved_dtype = resolve_dtype(args.dtype)
    target_modules = parse_target_modules(args.target_modules)
    quantization_config = build_4bit_quantization_config(
        load_in_4bit=args.load_in_4bit,
        quant_type=args.bnb_4bit_quant_type,
        compute_dtype_name=args.bnb_4bit_compute_dtype,
        use_double_quant=args.bnb_4bit_use_double_quant,
        fallback_dtype=resolved_dtype,
    )

    model_kwargs = {
        "torch_dtype": resolved_dtype,
        "low_cpu_mem_usage": True,
        "num_labels": 2,
    }
    resolved_device_map = resolve_device_map_arg(args.device_map)
    if resolved_device_map is not None:
        model_kwargs["device_map"] = resolved_device_map
    elif args.load_in_4bit and world_size > 1 and local_rank >= 0:
        # In torchrun, each process must load the quantized model on its own GPU.
        model_kwargs["device_map"] = {"": local_rank}
    resolved_attn_implementation = resolve_attn_implementation(args.attn_implementation)
    if resolved_attn_implementation is not None:
        model_kwargs["attn_implementation"] = resolved_attn_implementation
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForSequenceClassification.from_pretrained(args.model_id, **model_kwargs)
    resize_model_embeddings_if_needed(model, tokenizer)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    if args.load_in_4bit:
        model = prepare_model_for_4bit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )
    elif args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
        modules_to_save=["score"],
    )
    model = get_peft_model(model, peft_config)
    model, numeric_embedding_config = wrap_model_with_numeric_embeddings(
        model,
        tokenizer,
        artifact_path=None,
        enable_numeric_embedding=args.use_numeric_embedding,
        max_numeric_chars=args.max_numeric_chars,
        numeric_char_embedding_dim=args.numeric_char_embedding_dim,
        numeric_gru_hidden_size=args.numeric_gru_hidden_size,
        freeze_value_encoder=False,
    )

    collator = TraceScorerDataCollator(pad_token_id=tokenizer.pad_token_id)
    training_args = build_training_arguments(TrainingArguments, args=args, output_dir=output_dir)
    callbacks = []
    best_adapter_callback = None
    if args.save_best_adapter_at_each_eval:
        if args.eval_strategy == "no":
            raise ValueError("--save-best-adapter-at-each-eval requires --eval-strategy steps or epoch.")
        best_adapter_callback = SaveBestAdapterCallback(
            output_dir=output_dir,
            tokenizer=tokenizer,
            metric_name=args.metric_for_best_model,
            greater_is_better=args.greater_is_better,
            improvement_threshold=args.early_stopping_threshold,
        )
        callbacks.append(best_adapter_callback)
    if args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )
    trainer = build_trainer(
        Trainer,
        model=model,
        training_args=training_args,
        train_dataset=TokenizedTraceScorerDataset(train_features),
        eval_dataset=TokenizedTraceScorerDataset(validation_features) if validation_features else None,
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
        compute_metrics_fn=build_trace_and_task2_metrics(
            eval_features=validation_features,
            eval_k=args.eval_k,
            verdict_top_k=args.verdict_top_k,
        ),
    )

    resume_from_checkpoint = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint.strip().lower() in {"1", "true", "yes", "latest", "auto"}:
            resume_from_checkpoint = True
        else:
            resume_from_checkpoint = args.resume_from_checkpoint

    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    final_adapter_dir = output_dir / "final_adapter"
    skip_final_save = (
        best_adapter_callback is not None
        and best_adapter_callback.saved_best_adapter
        and not args.load_best_model_at_end
    )
    if skip_final_save:
        print(
            f"Keeping best adapter already saved at {final_adapter_dir}; "
            "skipping final last-model adapter overwrite."
        )
    else:
        trainer.model.save_pretrained(final_adapter_dir)
        tokenizer.save_pretrained(final_adapter_dir)

    metrics = dict(train_result.metrics)
    if validation_features:
        metrics.update({f"final_{k}": v for k, v in trainer.evaluate().items()})
    best_metric = getattr(trainer.state, "best_metric", None)
    best_model_checkpoint = getattr(trainer.state, "best_model_checkpoint", None)
    if best_metric is not None:
        metrics["best_metric"] = best_metric
    if best_model_checkpoint is not None:
        metrics["best_model_checkpoint"] = best_model_checkpoint
    with (output_dir / "train_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    run_metadata = {
        "model_id": args.model_id,
        "train_dataset": str(args.train_dataset),
        "validation_dataset": str(args.validation_dataset),
        "load_in_4bit": args.load_in_4bit,
        "use_numeric_embedding": args.use_numeric_embedding,
        "training_args": {
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "optim": args.optim,
            "resume_from_checkpoint": args.resume_from_checkpoint,
            "max_length": args.max_length,
            "max_claim_chars": args.max_claim_chars,
            "max_evidence_items": args.max_evidence_items,
            "max_evidence_chars": args.max_evidence_chars,
            "max_trace_chars": args.max_trace_chars,
            "append_normalized_numbers": args.append_normalized_numbers,
            "max_normalized_numbers": args.max_normalized_numbers,
            "eval_k": args.eval_k,
            "verdict_top_k": args.verdict_top_k,
            "target_modules": args.target_modules,
            "report_to": args.report_to,
            "run_name": args.run_name,
            "wandb_project": args.wandb_project,
            "wandb_entity": args.wandb_entity,
            "wandb_group": args.wandb_group,
            "wandb_tags": args.wandb_tags,
            "wandb_mode": args.wandb_mode,
            "wandb_log_model": args.wandb_log_model,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_threshold": args.early_stopping_threshold,
            "load_best_model_at_end": args.load_best_model_at_end,
            "save_best_adapter_at_each_eval": args.save_best_adapter_at_each_eval,
            "metric_for_best_model": args.metric_for_best_model,
            "greater_is_better": args.greater_is_better,
            "ddp_find_unused_parameters": args.ddp_find_unused_parameters,
        },
        "train_preprocess": {
            **stats_as_dict(train_stats),
            "average_sequence_length": average_sequence_length(train_stats),
        },
        "validation_preprocess": {
            **stats_as_dict(validation_stats),
            "average_sequence_length": average_sequence_length(validation_stats),
        },
        "numeric_embedding_config": None if numeric_embedding_config is None else {
            "enabled": numeric_embedding_config.enabled,
            "num_token": numeric_embedding_config.num_token,
            "num_token_id": numeric_embedding_config.num_token_id,
            "max_numeric_chars": numeric_embedding_config.max_numeric_chars,
            "numeric_char_embedding_dim": numeric_embedding_config.numeric_char_embedding_dim,
            "numeric_gru_hidden_size": numeric_embedding_config.numeric_gru_hidden_size,
            "hidden_size": numeric_embedding_config.hidden_size,
        },
        "metrics": metrics,
    }
    save_run_metadata(output_dir, run_metadata)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "final_adapter_dir": str(final_adapter_dir),
                "train_examples": len(train_features),
                "validation_examples": len(validation_features),
                "max_evidence_items": args.max_evidence_items,
                "max_evidence_chars": args.max_evidence_chars,
                "append_normalized_numbers": args.append_normalized_numbers,
                "max_normalized_numbers": args.max_normalized_numbers,
                "load_in_4bit": args.load_in_4bit,
                "use_numeric_embedding": args.use_numeric_embedding,
                "early_stopping_patience": args.early_stopping_patience,
                "best_metric": best_metric,
                "best_model_checkpoint": best_model_checkpoint,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
