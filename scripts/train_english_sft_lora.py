#!/usr/bin/env python3
"""Train an English-only SFT + LoRA model for CLEF Task 2 ranking/verdict generation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

try:
    from task2_ranking_utils import (
        DEFAULT_MODEL_ID,
        build_prompt,
        build_sft_completion,
        infer_label_space,
        resolve_dtype,
        set_seed,
        tokenize_supervised_example,
    )
    from task2_utils import load_json_rows
except ImportError:  # pragma: no cover - import path fallback
    from scripts.task2_ranking_utils import (
        DEFAULT_MODEL_ID,
        build_prompt,
        build_sft_completion,
        infer_label_space,
        resolve_dtype,
        set_seed,
        tokenize_supervised_example,
    )
    from scripts.task2_utils import load_json_rows


@dataclass
class PreprocessStats:
    num_rows_seen: int = 0
    num_examples_built: int = 0
    num_examples_skipped_overlength: int = 0


class TokenizedRankingDataset(Dataset):
    def __init__(self, features: Sequence[Dict[str, List[int]]]) -> None:
        self.features = list(features)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.features[idx]


class SupervisedDataCollator:
    def __init__(self, pad_token_id: int, label_pad_id: int = -100) -> None:
        self.pad_token_id = pad_token_id
        self.label_pad_id = label_pad_id

    def __call__(self, features: Sequence[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []

        for feature in features:
            pad_len = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            labels.append(feature["labels"] + [self.label_pad_id] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def build_tokenized_features(
    rows: Sequence[dict],
    tokenizer,
    label_space: Sequence[str],
    max_length: int,
    max_evidence_items: int,
    max_evidence_chars: int,
    max_trace_chars: int,
) -> tuple[list[Dict[str, List[int]]], PreprocessStats]:
    features: List[Dict[str, List[int]]] = []
    stats = PreprocessStats(num_rows_seen=len(rows))

    for row in rows:
        prompt = build_prompt(
            row=row,
            label_space=label_space,
            max_evidence_items=max_evidence_items,
            max_evidence_chars=max_evidence_chars,
            max_trace_chars=max_trace_chars,
        )
        completion = build_sft_completion(row)
        encoded = tokenize_supervised_example(
            tokenizer=tokenizer,
            prompt=prompt,
            assistant_response=completion,
            max_length=max_length,
        )
        if encoded is None:
            stats.num_examples_skipped_overlength += 1
            continue
        features.append(encoded)
        stats.num_examples_built += 1

    return features, stats


def maybe_limit_rows(rows: Sequence[dict], limit: Optional[int]) -> List[dict]:
    if limit is None:
        return list(rows)
    if limit < 0:
        raise ValueError(f"limit must be >= 0, got {limit}")
    return list(rows[:limit])


def parse_target_modules(value: str) -> List[str]:
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


def save_run_metadata(output_dir: Path, payload: Dict[str, object]) -> None:
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


def build_trainer(Trainer, model, training_args, train_dataset, eval_dataset, data_collator, tokenizer):
    common_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }
    try:
        return Trainer(processing_class=tokenizer, **common_kwargs)
    except TypeError:
        return Trainer(tokenizer=tokenizer, **common_kwargs)


def main() -> None:
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    parser = argparse.ArgumentParser(description="Train English-only SFT + LoRA for CLEF Task 2.")
    parser.add_argument("--train-dataset", type=Path, default=Path("dataset/english/train.json"))
    parser.add_argument("--validation-dataset", type=Path, default=Path("dataset/english/validation.json"))
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-validation", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-evidence-items", type=int, default=3)
    parser.add_argument("--max-evidence-chars", type=int, default=900)
    parser.add_argument("--max-trace-chars", type=int, default=320)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--device-map", type=str, default="none", choices=["none", "auto"])
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
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", type=str, default="q_proj,v_proj")
    parser.add_argument("--optim", type=str, default="adamw_torch")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = args.output_dir.resolve()

    train_rows = maybe_limit_rows(load_json_rows(args.train_dataset), args.limit_train)
    validation_rows = maybe_limit_rows(load_json_rows(args.validation_dataset), args.limit_validation)
    label_space = infer_label_space(train_rows + validation_rows)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_features, train_stats = build_tokenized_features(
        rows=train_rows,
        tokenizer=tokenizer,
        label_space=label_space,
        max_length=args.max_length,
        max_evidence_items=args.max_evidence_items,
        max_evidence_chars=args.max_evidence_chars,
        max_trace_chars=args.max_trace_chars,
    )
    validation_features, validation_stats = build_tokenized_features(
        rows=validation_rows,
        tokenizer=tokenizer,
        label_space=label_space,
        max_length=args.max_length,
        max_evidence_items=args.max_evidence_items,
        max_evidence_chars=args.max_evidence_chars,
        max_trace_chars=args.max_trace_chars,
    )

    if not train_features:
        raise ValueError("No train examples remained after preprocessing. Adjust max length or truncation settings.")
    if not validation_features and args.eval_strategy != "no":
        raise ValueError("No validation examples remained after preprocessing. Adjust validation limits or truncation.")

    model_kwargs = {
        "torch_dtype": resolve_dtype(args.dtype),
    }
    resolved_device_map = resolve_device_map_arg(args.device_map)
    if resolved_device_map is not None:
        model_kwargs["device_map"] = resolved_device_map

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=parse_target_modules(args.target_modules),
    )
    model = get_peft_model(model, peft_config)

    collator = SupervisedDataCollator(pad_token_id=tokenizer.pad_token_id)

    training_args = build_training_arguments(TrainingArguments, args=args, output_dir=output_dir)

    trainer = build_trainer(
        Trainer,
        model=model,
        training_args=training_args,
        train_dataset=TokenizedRankingDataset(train_features),
        eval_dataset=TokenizedRankingDataset(validation_features) if validation_features else None,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    train_result = trainer.train()

    final_adapter_dir = output_dir / "final_adapter"
    trainer.model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)

    metrics = dict(train_result.metrics)
    metrics_path = output_dir / "train_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    if validation_features and args.eval_strategy != "no":
        eval_metrics = trainer.evaluate()
        eval_metrics_path = output_dir / "eval_metrics.json"
        with eval_metrics_path.open("w", encoding="utf-8") as f:
            json.dump(eval_metrics, f, indent=2, ensure_ascii=False)

    save_run_metadata(
        output_dir=output_dir,
        payload={
            "model_id": args.model_id,
            "train_dataset": str(args.train_dataset),
            "validation_dataset": str(args.validation_dataset),
            "label_space": list(label_space),
            "train_preprocess": asdict(train_stats),
            "validation_preprocess": asdict(validation_stats),
            "training_args": {
                "max_length": args.max_length,
                "max_evidence_items": args.max_evidence_items,
                "max_evidence_chars": args.max_evidence_chars,
                "max_trace_chars": args.max_trace_chars,
                "learning_rate": args.learning_rate,
                "num_train_epochs": args.num_train_epochs,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "seed": args.seed,
                "dtype": args.dtype,
                "device_map": args.device_map,
                "gradient_checkpointing": args.gradient_checkpointing,
            },
            "lora_config": {
                "r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "target_modules": parse_target_modules(args.target_modules),
            },
            "artifacts": {
                "final_adapter_dir": str(final_adapter_dir),
                "train_metrics": str(metrics_path),
            },
        },
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "final_adapter_dir": str(final_adapter_dir),
                "train_examples": len(train_features),
                "validation_examples": len(validation_features),
                "train_skipped_overlength": train_stats.num_examples_skipped_overlength,
                "validation_skipped_overlength": validation_stats.num_examples_skipped_overlength,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
