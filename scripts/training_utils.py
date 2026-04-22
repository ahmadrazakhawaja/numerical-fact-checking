#!/usr/bin/env python3
"""Shared trainer CLI helpers for LoRA/QLoRA scripts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Sequence


def maybe_limit_rows(rows: Sequence[dict], limit: Optional[int]) -> list[dict]:
    if limit is None:
        return list(rows)
    if limit < 0:
        raise ValueError(f"limit must be >= 0, got {limit}")
    return list(rows[:limit])


def parse_target_modules(value: str):
    lowered = value.strip().lower()
    if lowered == "all-linear":
        return "all-linear"

    modules = [item.strip() for item in value.split(",") if item.strip()]
    if not modules:
        raise ValueError("At least one LoRA target module is required.")
    return modules


def resolve_device_map_arg(value: str):
    lowered = value.strip().lower()
    if lowered == "none":
        return None
    if lowered == "auto":
        return "auto"
    raise ValueError(f"Unsupported device_map value: {value}")


def resolve_attn_implementation(value: str) -> Optional[str]:
    lowered = value.strip().lower()
    if lowered == "auto":
        return None
    if lowered in {"sdpa", "eager"}:
        return lowered
    raise ValueError(f"Unsupported attention implementation: {value}")


def save_run_metadata(output_dir: Path, payload: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_config.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def add_reporting_args(parser) -> None:
    parser.add_argument(
        "--report-to",
        type=str,
        default="none",
        help='Trainer reporting backend. Use "wandb" to enable Weights & Biases logging.',
    )
    parser.add_argument("--run-name", type=str, default=None, help="Optional Trainer/W&B run name.")
    parser.add_argument("--wandb-project", type=str, default=None, help="Optional W&B project name.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Optional W&B team/entity.")
    parser.add_argument("--wandb-group", type=str, default=None, help="Optional W&B run group.")
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=None, help="Optional W&B tags.")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default=None,
        help="Optional W&B mode. Use offline when the cluster has no network during training.",
    )
    parser.add_argument(
        "--wandb-log-model",
        type=str,
        choices=["false", "checkpoint", "end"],
        default=None,
        help="Optional W&B model artifact logging mode.",
    )


def normalize_report_to(value: object) -> object:
    if value is None:
        return "none"
    if isinstance(value, (list, tuple)):
        return list(value)
    text = str(value).strip()
    if not text:
        return "none"
    if "," in text:
        return [item.strip() for item in text.split(",") if item.strip()]
    return text


def reporting_uses_wandb(value: object) -> bool:
    normalized = normalize_report_to(value)
    if isinstance(normalized, list):
        return any(str(item).strip().lower() == "wandb" for item in normalized)
    return str(normalized).strip().lower() == "wandb"


def configure_reporting(args, output_dir: Path) -> None:
    if not reporting_uses_wandb(getattr(args, "report_to", "none")):
        return

    wandb_project = getattr(args, "wandb_project", None)
    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
    else:
        os.environ.setdefault("WANDB_PROJECT", "numerical-fact-checking")
    if getattr(args, "wandb_entity", None):
        os.environ["WANDB_ENTITY"] = args.wandb_entity
    if getattr(args, "wandb_group", None):
        os.environ["WANDB_RUN_GROUP"] = args.wandb_group
    if getattr(args, "run_name", None):
        os.environ["WANDB_NAME"] = args.run_name
    else:
        os.environ.setdefault("WANDB_NAME", output_dir.name)
    if getattr(args, "wandb_tags", None):
        os.environ["WANDB_TAGS"] = ",".join(str(tag) for tag in args.wandb_tags if str(tag).strip())
    if getattr(args, "wandb_mode", None):
        os.environ["WANDB_MODE"] = args.wandb_mode
    if getattr(args, "wandb_log_model", None):
        os.environ["WANDB_LOG_MODEL"] = args.wandb_log_model


def flatten_numeric_metrics(metrics: dict[str, object], prefix: str = "") -> dict[str, float]:
    flattened: dict[str, float] = {}
    for key, value in metrics.items():
        metric_name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_numeric_metrics(value, metric_name))
        elif isinstance(value, bool):
            flattened[metric_name] = float(value)
        elif isinstance(value, (int, float)):
            flattened[metric_name] = float(value)
    return flattened


def log_wandb_metrics(
    args,
    output_dir: Path,
    metrics: dict[str, object],
    *,
    prefix: str = "",
    config: Optional[dict[str, object]] = None,
) -> None:
    if not reporting_uses_wandb(getattr(args, "report_to", "none")):
        return

    configure_reporting(args, output_dir)
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("W&B logging was requested, but wandb is not installed. Run `uv add wandb`.") from exc

    tags = getattr(args, "wandb_tags", None)
    if not tags and os.environ.get("WANDB_TAGS"):
        tags = [tag.strip() for tag in os.environ["WANDB_TAGS"].split(",") if tag.strip()]

    init_kwargs = {
        "project": os.environ.get("WANDB_PROJECT"),
        "entity": os.environ.get("WANDB_ENTITY"),
        "name": os.environ.get("WANDB_NAME"),
        "group": os.environ.get("WANDB_RUN_GROUP"),
        "tags": tags,
        "mode": os.environ.get("WANDB_MODE"),
        "config": config,
    }
    run = wandb.init(**{key: value for key, value in init_kwargs.items() if value is not None})
    flattened = flatten_numeric_metrics(metrics, prefix=prefix)
    if flattened:
        wandb.log(flattened)
        for key, value in flattened.items():
            wandb.summary[key] = value
    run.finish()


def build_training_arguments(TrainingArguments, args, output_dir: Path):
    load_best_model_at_end = bool(getattr(args, "load_best_model_at_end", False))
    common_kwargs = {
        "output_dir": str(output_dir),
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_strategy": args.save_strategy,
        "save_total_limit": args.save_total_limit,
        "bf16": args.dtype == "bfloat16",
        "fp16": args.dtype == "float16",
        "remove_unused_columns": False,
        "report_to": normalize_report_to(getattr(args, "report_to", "none")),
        "optim": args.optim,
    }
    if getattr(args, "run_name", None):
        common_kwargs["run_name"] = args.run_name

    if load_best_model_at_end:
        common_kwargs["load_best_model_at_end"] = True
        metric_for_best_model = getattr(args, "metric_for_best_model", None)
        if metric_for_best_model:
            common_kwargs["metric_for_best_model"] = metric_for_best_model
        greater_is_better = getattr(args, "greater_is_better", None)
        if greater_is_better is not None:
            common_kwargs["greater_is_better"] = greater_is_better

    try:
        return TrainingArguments(evaluation_strategy=args.eval_strategy, **common_kwargs)
    except TypeError:
        return TrainingArguments(eval_strategy=args.eval_strategy, **common_kwargs)
