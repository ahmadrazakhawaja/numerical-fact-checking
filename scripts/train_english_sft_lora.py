#!/usr/bin/env python3
"""Train an English-only SFT + LoRA model for CLEF Task 2 ranking/verdict generation."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.utils.data import Dataset

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
    from task2_ranking_utils import (
        DEFAULT_MODEL_ID,
        build_prompt_artifacts,
        build_sft_completion,
        infer_label_space,
        resolve_dtype,
        set_seed,
        tokenize_supervised_example,
    )
    from training_utils import (
        add_reporting_args,
        build_training_arguments,
        configure_reporting,
        maybe_limit_rows,
        parse_target_modules,
        resolve_attn_implementation,
        resolve_device_map_arg,
        save_run_metadata,
    )
    from task2_utils import load_json_rows
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
    from scripts.task2_ranking_utils import (
        DEFAULT_MODEL_ID,
        build_prompt_artifacts,
        build_sft_completion,
        infer_label_space,
        resolve_dtype,
        set_seed,
        tokenize_supervised_example,
    )
    from scripts.training_utils import (
        add_reporting_args,
        build_training_arguments,
        configure_reporting,
        maybe_limit_rows,
        parse_target_modules,
        resolve_attn_implementation,
        resolve_device_map_arg,
        save_run_metadata,
    )
    from scripts.task2_utils import load_json_rows


@dataclass
class PreprocessStats:
    num_rows_seen: int = 0
    num_examples_built: int = 0
    num_examples_skipped_overlength: int = 0
    total_sequence_length: int = 0
    max_sequence_length: int = 0


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
        numeric_mask = []
        numeric_char_ids = []
        has_numeric = "numeric_mask" in features[0] and "numeric_char_ids" in features[0]
        max_numeric_chars = len(features[0]["numeric_char_ids"][0]) if has_numeric and features[0]["numeric_char_ids"] else 0

        for feature in features:
            pad_len = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            labels.append(feature["labels"] + [self.label_pad_id] * pad_len)
            if has_numeric:
                numeric_mask.append(feature["numeric_mask"] + [0] * pad_len)
                numeric_char_ids.append(
                    feature["numeric_char_ids"] + ([[0] * max_numeric_chars] * pad_len)
                )

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        if has_numeric:
            batch["numeric_mask"] = torch.tensor(numeric_mask, dtype=torch.bool)
            batch["numeric_char_ids"] = torch.tensor(numeric_char_ids, dtype=torch.long)
        return batch


def build_tokenized_features(
    rows: Sequence[dict],
    tokenizer,
    label_space: Sequence[str],
    max_length: int,
    max_evidence_items: int,
    max_evidence_chars: int,
    max_trace_chars: int,
    use_numeric_embedding: bool,
    max_numeric_chars: int,
) -> tuple[list[Dict[str, List[int]]], PreprocessStats]:
    features: List[Dict[str, List[int]]] = []
    stats = PreprocessStats(num_rows_seen=len(rows))
    num_token_id = tokenizer.convert_tokens_to_ids("<num>") if use_numeric_embedding else None

    for row in rows:
        prompt_artifacts = build_prompt_artifacts(
            row=row,
            label_space=label_space,
            max_evidence_items=max_evidence_items,
            max_evidence_chars=max_evidence_chars,
            max_trace_chars=max_trace_chars,
            use_numeric_embedding=use_numeric_embedding,
        )
        completion = build_sft_completion(row)
        encoded = tokenize_supervised_example(
            tokenizer=tokenizer,
            prompt=prompt_artifacts["prompt"],
            assistant_response=completion,
            max_length=max_length,
            prompt_numeric_canonicals=prompt_artifacts["prompt_numeric_canonicals"] if use_numeric_embedding else None,
            num_token_id=num_token_id,
            max_numeric_chars=max_numeric_chars,
        )
        if encoded is None:
            stats.num_examples_skipped_overlength += 1
            continue
        sequence_length = len(encoded["input_ids"])
        stats.total_sequence_length += sequence_length
        stats.max_sequence_length = max(stats.max_sequence_length, sequence_length)
        features.append(encoded)
        stats.num_examples_built += 1

    return features, stats


def average_sequence_length(stats: PreprocessStats) -> float:
    if stats.num_examples_built == 0:
        return 0.0
    return stats.total_sequence_length / stats.num_examples_built


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
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--max-evidence-items", type=int, default=2)
    parser.add_argument("--max-evidence-chars", type=int, default=512)
    parser.add_argument("--max-trace-chars", type=int, default=160)
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
    parser.add_argument("--gradient-checkpointing", dest="gradient_checkpointing", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        type=str,
        default="q_proj,v_proj",
        help='Comma-separated module names or "all-linear" for QLoRA-style coverage.',
    )
    parser.add_argument("--optim", type=str, default="adamw_torch")
    add_reporting_args(parser)
    add_4bit_loading_args(parser)
    add_numeric_embedding_args(parser)
    parser.set_defaults(gradient_checkpointing=True)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = args.output_dir.resolve()
    configure_reporting(args, output_dir)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and args.device_map == "none":
        print(
            f"Detected {torch.cuda.device_count()} visible CUDA devices. "
            "Use --device-map auto if you want Hugging Face to shard model loading."
        )

    train_rows = maybe_limit_rows(load_json_rows(args.train_dataset), args.limit_train)
    validation_rows = maybe_limit_rows(load_json_rows(args.validation_dataset), args.limit_validation)
    label_space = infer_label_space(train_rows + validation_rows)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.use_numeric_embedding:
        ensure_numeric_token(tokenizer)

    train_features, train_stats = build_tokenized_features(
        rows=train_rows,
        tokenizer=tokenizer,
        label_space=label_space,
        max_length=args.max_length,
        max_evidence_items=args.max_evidence_items,
        max_evidence_chars=args.max_evidence_chars,
        max_trace_chars=args.max_trace_chars,
        use_numeric_embedding=args.use_numeric_embedding,
        max_numeric_chars=args.max_numeric_chars,
    )
    validation_features, validation_stats = build_tokenized_features(
        rows=validation_rows,
        tokenizer=tokenizer,
        label_space=label_space,
        max_length=args.max_length,
        max_evidence_items=args.max_evidence_items,
        max_evidence_chars=args.max_evidence_chars,
        max_trace_chars=args.max_trace_chars,
        use_numeric_embedding=args.use_numeric_embedding,
        max_numeric_chars=args.max_numeric_chars,
    )

    if not train_features:
        raise ValueError("No train examples remained after preprocessing. Adjust max length or truncation settings.")
    if not validation_features and args.eval_strategy != "no":
        raise ValueError("No validation examples remained after preprocessing. Adjust validation limits or truncation.")

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
    }
    resolved_device_map = resolve_device_map_arg(args.device_map)
    if resolved_device_map is not None:
        model_kwargs["device_map"] = resolved_device_map
    resolved_attn_implementation = resolve_attn_implementation(args.attn_implementation)
    if resolved_attn_implementation is not None:
        model_kwargs["attn_implementation"] = resolved_attn_implementation
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    resize_model_embeddings_if_needed(model, tokenizer)
    model.config.use_cache = False
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
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
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
                "attn_implementation": args.attn_implementation,
                "gradient_checkpointing": args.gradient_checkpointing,
                "report_to": args.report_to,
                "run_name": args.run_name,
                "wandb_project": args.wandb_project,
                "wandb_entity": args.wandb_entity,
                "wandb_group": args.wandb_group,
                "wandb_tags": args.wandb_tags,
                "wandb_mode": args.wandb_mode,
                "wandb_log_model": args.wandb_log_model,
            },
            "lora_config": {
                "r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "target_modules": target_modules,
            },
            "quantization": {
                "load_in_4bit": args.load_in_4bit,
                "bnb_4bit_quant_type": args.bnb_4bit_quant_type,
                "bnb_4bit_compute_dtype": args.bnb_4bit_compute_dtype,
                "bnb_4bit_use_double_quant": args.bnb_4bit_use_double_quant,
            },
            "numeric_embedding": {
                "enabled": args.use_numeric_embedding,
                "config": asdict(numeric_embedding_config) if numeric_embedding_config is not None else None,
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
                "train_avg_sequence_length": average_sequence_length(train_stats),
                "validation_avg_sequence_length": average_sequence_length(validation_stats),
                "train_max_sequence_length": train_stats.max_sequence_length,
                "validation_max_sequence_length": validation_stats.max_sequence_length,
                "load_in_4bit": args.load_in_4bit,
                "use_numeric_embedding": args.use_numeric_embedding,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
