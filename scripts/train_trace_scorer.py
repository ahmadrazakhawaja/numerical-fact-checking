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
    from task2_ranking_utils import DEFAULT_MODEL_ID, resolve_dtype, set_seed
    from task2_utils import load_json_rows
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
    from scripts.task2_ranking_utils import DEFAULT_MODEL_ID, resolve_dtype, set_seed
    from scripts.task2_utils import load_json_rows
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


def build_trainer(Trainer, model, training_args, train_dataset, eval_dataset, data_collator, tokenizer, callbacks=None):
    common_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "compute_metrics": compute_binary_metrics,
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
    parser.add_argument("--max-trace-chars", type=int, default=1800)
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
        default="positive_f1",
        help="Evaluation metric used for early stopping and best-checkpoint loading.",
    )
    parser.add_argument(
        "--load-best-model-at-end",
        action="store_true",
        help="Reload the best checkpoint before saving final_adapter. Enabled automatically with early stopping.",
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
        max_trace_chars=args.max_trace_chars,
        use_numeric_embedding=args.use_numeric_embedding,
        max_numeric_chars=args.max_numeric_chars,
    )
    validation_features, validation_stats = build_trace_scorer_features(
        validation_rows,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_claim_chars=args.max_claim_chars,
        max_trace_chars=args.max_trace_chars,
        use_numeric_embedding=args.use_numeric_embedding,
        max_numeric_chars=args.max_numeric_chars,
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
    )

    resume_from_checkpoint = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint.strip().lower() in {"1", "true", "yes", "latest", "auto"}:
            resume_from_checkpoint = True
        else:
            resume_from_checkpoint = args.resume_from_checkpoint

    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    final_adapter_dir = output_dir / "final_adapter"
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
            "max_trace_chars": args.max_trace_chars,
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
