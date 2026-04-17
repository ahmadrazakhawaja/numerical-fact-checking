#!/usr/bin/env python3
"""Shared trainer CLI helpers for LoRA/QLoRA scripts."""

from __future__ import annotations

import json
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


def build_training_arguments(TrainingArguments, args, output_dir: Path):
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
        "report_to": "none",
        "optim": args.optim,
    }

    try:
        return TrainingArguments(evaluation_strategy=args.eval_strategy, **common_kwargs)
    except TypeError:
        return TrainingArguments(eval_strategy=args.eval_strategy, **common_kwargs)
