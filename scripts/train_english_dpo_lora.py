#!/usr/bin/env python3
"""Train an English-only DPO + LoRA model for CLEF Task 2."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
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
        tokenizer_source_for_adapter,
        wrap_model_with_numeric_embeddings,
    )
    from task2_ranking_utils import (
        DEFAULT_MODEL_ID,
        build_dpo_pairs,
        build_prompt_artifacts,
        infer_label_space,
        resolve_dtype,
        set_seed,
        tokenize_supervised_example,
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
        tokenizer_source_for_adapter,
        wrap_model_with_numeric_embeddings,
    )
    from scripts.task2_ranking_utils import (
        DEFAULT_MODEL_ID,
        build_dpo_pairs,
        build_prompt_artifacts,
        infer_label_space,
        resolve_dtype,
        set_seed,
        tokenize_supervised_example,
    )
    from scripts.task2_utils import load_json_rows


@dataclass
class DPOPreprocessStats:
    num_rows_seen: int = 0
    num_pairs_built: int = 0
    num_pairs_skipped_overlength: int = 0
    total_chosen_length: int = 0
    total_rejected_length: int = 0
    max_chosen_length: int = 0
    max_rejected_length: int = 0


class TokenizedDPODataset(Dataset):
    def __init__(self, features: Sequence[Dict[str, List[int]]]) -> None:
        self.features = list(features)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.features[idx]


class DPODataCollator:
    def __init__(self, pad_token_id: int, label_pad_id: int = -100) -> None:
        self.pad_token_id = pad_token_id
        self.label_pad_id = label_pad_id

    def _pad_block(
        self,
        features: Sequence[Dict[str, List[int]]],
        prefix: str,
    ) -> Dict[str, torch.Tensor]:
        input_key = f"{prefix}_input_ids"
        attention_key = f"{prefix}_attention_mask"
        labels_key = f"{prefix}_labels"
        max_length = max(len(feature[input_key]) for feature in features)

        input_ids = []
        attention_mask = []
        labels = []
        numeric_mask = []
        numeric_char_ids = []
        has_numeric = f"{prefix}_numeric_mask" in features[0] and f"{prefix}_numeric_char_ids" in features[0]
        max_numeric_chars = (
            len(features[0][f"{prefix}_numeric_char_ids"][0])
            if has_numeric and features[0][f"{prefix}_numeric_char_ids"]
            else 0
        )
        for feature in features:
            pad_len = max_length - len(feature[input_key])
            input_ids.append(feature[input_key] + [self.pad_token_id] * pad_len)
            attention_mask.append(feature[attention_key] + [0] * pad_len)
            labels.append(feature[labels_key] + [self.label_pad_id] * pad_len)
            if has_numeric:
                numeric_mask.append(feature[f"{prefix}_numeric_mask"] + [0] * pad_len)
                numeric_char_ids.append(
                    feature[f"{prefix}_numeric_char_ids"] + ([[0] * max_numeric_chars] * pad_len)
                )

        output = {
            input_key: torch.tensor(input_ids, dtype=torch.long),
            attention_key: torch.tensor(attention_mask, dtype=torch.long),
            labels_key: torch.tensor(labels, dtype=torch.long),
        }
        if has_numeric:
            output[f"{prefix}_numeric_mask"] = torch.tensor(numeric_mask, dtype=torch.bool)
            output[f"{prefix}_numeric_char_ids"] = torch.tensor(numeric_char_ids, dtype=torch.long)
        return output

    def __call__(self, features: Sequence[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        batch = {}
        batch.update(self._pad_block(features, "chosen"))
        batch.update(self._pad_block(features, "rejected"))
        return batch


def maybe_limit_rows(rows: Sequence[dict], limit: Optional[int]) -> List[dict]:
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


def save_run_metadata(output_dir: Path, payload: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_config.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def average_length(total: int, count: int) -> float:
    if count == 0:
        return 0.0
    return total / count


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


def build_trainer(trainer_cls, model, ref_model, beta, training_args, train_dataset, eval_dataset, data_collator, tokenizer):
    common_kwargs = {
        "model": model,
        "ref_model": ref_model,
        "beta": beta,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }
    try:
        return trainer_cls(processing_class=tokenizer, **common_kwargs)
    except TypeError:
        return trainer_cls(tokenizer=tokenizer, **common_kwargs)


def build_tokenized_dpo_features(
    rows: Sequence[dict],
    tokenizer,
    label_space: Sequence[str],
    max_length: int,
    max_evidence_items: int,
    max_evidence_chars: int,
    max_trace_chars: int,
    max_rejected_per_prompt: int,
    use_numeric_embedding: bool,
    max_numeric_chars: int,
) -> tuple[list[Dict[str, List[int]]], DPOPreprocessStats]:
    features: List[Dict[str, List[int]]] = []
    stats = DPOPreprocessStats(num_rows_seen=len(rows))
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
        for pair in build_dpo_pairs(
            row=row,
            label_space=label_space,
            max_rejected_per_prompt=max_rejected_per_prompt,
        ):
            chosen = tokenize_supervised_example(
                tokenizer=tokenizer,
                prompt=prompt_artifacts["prompt"],
                assistant_response=pair["chosen"],
                max_length=max_length,
                prompt_numeric_canonicals=prompt_artifacts["prompt_numeric_canonicals"] if use_numeric_embedding else None,
                num_token_id=num_token_id,
                max_numeric_chars=max_numeric_chars,
            )
            rejected = tokenize_supervised_example(
                tokenizer=tokenizer,
                prompt=prompt_artifacts["prompt"],
                assistant_response=pair["rejected"],
                max_length=max_length,
                prompt_numeric_canonicals=prompt_artifacts["prompt_numeric_canonicals"] if use_numeric_embedding else None,
                num_token_id=num_token_id,
                max_numeric_chars=max_numeric_chars,
            )
            if chosen is None or rejected is None:
                stats.num_pairs_skipped_overlength += 1
                continue

            chosen_len = len(chosen["input_ids"])
            rejected_len = len(rejected["input_ids"])
            stats.total_chosen_length += chosen_len
            stats.total_rejected_length += rejected_len
            stats.max_chosen_length = max(stats.max_chosen_length, chosen_len)
            stats.max_rejected_length = max(stats.max_rejected_length, rejected_len)
            stats.num_pairs_built += 1

            features.append(
                {
                    "chosen_input_ids": chosen["input_ids"],
                    "chosen_attention_mask": chosen["attention_mask"],
                    "chosen_labels": chosen["labels"],
                    "rejected_input_ids": rejected["input_ids"],
                    "rejected_attention_mask": rejected["attention_mask"],
                    "rejected_labels": rejected["labels"],
                }
            )
            if use_numeric_embedding:
                features[-1]["chosen_numeric_mask"] = chosen["numeric_mask"]
                features[-1]["chosen_numeric_char_ids"] = chosen["numeric_char_ids"]
                features[-1]["rejected_numeric_mask"] = rejected["numeric_mask"]
                features[-1]["rejected_numeric_char_ids"] = rejected["numeric_char_ids"]

    return features, stats


def mark_adapter_trainable(model) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "lora_" in name or "modules_to_save" in name


def maybe_enable_gradient_checkpointing(model, enabled: bool) -> None:
    model.config.use_cache = False
    if not enabled:
        return
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()


def build_model_kwargs(args) -> Dict[str, object]:
    resolved_dtype = resolve_dtype(args.dtype)
    quantization_config = build_4bit_quantization_config(
        load_in_4bit=args.load_in_4bit,
        quant_type=args.bnb_4bit_quant_type,
        compute_dtype_name=args.bnb_4bit_compute_dtype,
        use_double_quant=args.bnb_4bit_use_double_quant,
        fallback_dtype=resolved_dtype,
    )

    model_kwargs: Dict[str, object] = {
        "torch_dtype": resolved_dtype,
        "low_cpu_mem_usage": True,
    }
    resolved_device_map = resolve_device_map_arg(args.device_map)
    if resolved_device_map is not None:
        model_kwargs["device_map"] = resolved_device_map
    resolved_attn = resolve_attn_implementation(args.attn_implementation)
    if resolved_attn is not None:
        model_kwargs["attn_implementation"] = resolved_attn
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    return model_kwargs


def load_policy_model(args, model_kwargs):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base_model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    resize_model_embeddings_if_needed(base_model, args.tokenizer)
    if args.load_in_4bit:
        base_model = prepare_model_for_4bit_training(
            base_model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )
    else:
        maybe_enable_gradient_checkpointing(base_model, args.gradient_checkpointing)

    try:
        model = PeftModel.from_pretrained(base_model, str(args.sft_adapter_path), is_trainable=True)
    except TypeError:
        model = PeftModel.from_pretrained(base_model, str(args.sft_adapter_path))
        mark_adapter_trainable(model)
    model, _ = wrap_model_with_numeric_embeddings(
        model,
        args.tokenizer,
        artifact_path=args.sft_adapter_path,
        enable_numeric_embedding=args.use_numeric_embedding,
        max_numeric_chars=args.max_numeric_chars,
        numeric_char_embedding_dim=args.numeric_char_embedding_dim,
        numeric_gru_hidden_size=args.numeric_gru_hidden_size,
        freeze_value_encoder=False,
    )
    model.train()
    return model


def load_reference_model(args, model_kwargs):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    ref_base_model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    resize_model_embeddings_if_needed(ref_base_model, args.tokenizer)
    ref_base_model.config.use_cache = False
    ref_model = PeftModel.from_pretrained(ref_base_model, str(args.sft_adapter_path))
    ref_model, _ = wrap_model_with_numeric_embeddings(
        ref_model,
        args.tokenizer,
        artifact_path=args.sft_adapter_path,
        enable_numeric_embedding=args.use_numeric_embedding,
        max_numeric_chars=args.max_numeric_chars,
        numeric_char_embedding_dim=args.numeric_char_embedding_dim,
        numeric_gru_hidden_size=args.numeric_gru_hidden_size,
        freeze_value_encoder=True,
    )
    ref_model.eval()
    for parameter in ref_model.parameters():
        parameter.requires_grad = False
    return ref_model


class DPOLoraTrainerMixin:
    @staticmethod
    def _sequence_logps(model, input_ids, attention_mask, labels, numeric_mask=None, numeric_char_ids=None):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            numeric_mask=numeric_mask,
            numeric_char_ids=numeric_char_ids,
            use_cache=False,
        )
        logits = outputs.logits

        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:].clone()
        loss_mask = shift_labels.ne(-100)
        shift_labels[~loss_mask] = 0

        token_logps = F.log_softmax(shift_logits, dim=-1).gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)
        return (token_logps * loss_mask).sum(dim=-1)


def build_dpo_trainer_class(Trainer):
    class DPOLoraTrainer(DPOLoraTrainerMixin, Trainer):
        def __init__(self, *args, ref_model, beta: float, **kwargs):
            super().__init__(*args, **kwargs)
            self.ref_model = ref_model
            self.beta = beta
            if getattr(self.ref_model, "hf_device_map", None) is None:
                self._move_model_to_device(self.ref_model, self.args.device)
            self.ref_model.eval()
            for parameter in self.ref_model.parameters():
                parameter.requires_grad = False

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            chosen_input_ids = inputs["chosen_input_ids"]
            chosen_attention_mask = inputs["chosen_attention_mask"]
            chosen_labels = inputs["chosen_labels"]
            rejected_input_ids = inputs["rejected_input_ids"]
            rejected_attention_mask = inputs["rejected_attention_mask"]
            rejected_labels = inputs["rejected_labels"]
            chosen_numeric_mask = inputs.get("chosen_numeric_mask")
            chosen_numeric_char_ids = inputs.get("chosen_numeric_char_ids")
            rejected_numeric_mask = inputs.get("rejected_numeric_mask")
            rejected_numeric_char_ids = inputs.get("rejected_numeric_char_ids")

            policy_chosen_logps = self._sequence_logps(
                model,
                chosen_input_ids,
                chosen_attention_mask,
                chosen_labels,
                chosen_numeric_mask,
                chosen_numeric_char_ids,
            )
            policy_rejected_logps = self._sequence_logps(
                model,
                rejected_input_ids,
                rejected_attention_mask,
                rejected_labels,
                rejected_numeric_mask,
                rejected_numeric_char_ids,
            )

            with torch.no_grad():
                ref_chosen_logps = self._sequence_logps(
                    self.ref_model,
                    chosen_input_ids,
                    chosen_attention_mask,
                    chosen_labels,
                    chosen_numeric_mask,
                    chosen_numeric_char_ids,
                )
                ref_rejected_logps = self._sequence_logps(
                    self.ref_model,
                    rejected_input_ids,
                    rejected_attention_mask,
                    rejected_labels,
                    rejected_numeric_mask,
                    rejected_numeric_char_ids,
                )

            logits = self.beta * (
                (policy_chosen_logps - policy_rejected_logps)
                - (ref_chosen_logps - ref_rejected_logps)
            )
            losses = -F.logsigmoid(logits)
            loss = losses.mean()

            if not return_outputs:
                return loss

            outputs = {
                "losses": losses.detach(),
                "policy_margin": (policy_chosen_logps - policy_rejected_logps).detach(),
                "reference_margin": (ref_chosen_logps - ref_rejected_logps).detach(),
            }
            return loss, outputs

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            with torch.no_grad():
                loss = self.compute_loss(model, inputs)
            return loss.detach(), None, None

    return DPOLoraTrainer


def main() -> None:
    from transformers import Trainer, TrainingArguments

    parser = argparse.ArgumentParser(description="Train English-only DPO + LoRA for CLEF Task 2.")
    parser.add_argument("--train-dataset", type=Path, default=Path("dataset/english/train_complete.json"))
    parser.add_argument("--validation-dataset", type=Path, default=Path("dataset/english/validation_complete.json"))
    parser.add_argument("--sft-adapter-path", type=Path, required=True)
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-validation", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--max-evidence-items", type=int, default=2)
    parser.add_argument("--max-evidence-chars", type=int, default=512)
    parser.add_argument("--max-trace-chars", type=int, default=160)
    parser.add_argument("--max-rejected-per-prompt", type=int, default=2)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--device-map", type=str, default="none", choices=["none", "auto"])
    parser.add_argument("--attn-implementation", type=str, default="sdpa", choices=["auto", "sdpa", "eager"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
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
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument(
        "--target-modules",
        type=str,
        default="all-linear",
        help='Stored for metadata only. DPO continues training the existing SFT adapter; "all-linear" is the recommended QLoRA setting.',
    )
    parser.add_argument("--optim", type=str, default="adamw_torch")
    add_4bit_loading_args(parser)
    add_numeric_embedding_args(parser)
    parser.set_defaults(gradient_checkpointing=True)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = args.output_dir.resolve()

    train_rows = maybe_limit_rows(load_json_rows(args.train_dataset), args.limit_train)
    validation_rows = maybe_limit_rows(load_json_rows(args.validation_dataset), args.limit_validation)
    label_space = infer_label_space(train_rows + validation_rows)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and args.device_map == "none":
        print(
            f"Detected {torch.cuda.device_count()} visible CUDA devices. "
            "Use --device-map auto if you want Hugging Face to shard model loading."
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source_for_adapter(args.model_id, args.sft_adapter_path)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.use_numeric_embedding:
        ensure_numeric_token(tokenizer)
    args.tokenizer = tokenizer

    train_features, train_stats = build_tokenized_dpo_features(
        rows=train_rows,
        tokenizer=tokenizer,
        label_space=label_space,
        max_length=args.max_length,
        max_evidence_items=args.max_evidence_items,
        max_evidence_chars=args.max_evidence_chars,
        max_trace_chars=args.max_trace_chars,
        max_rejected_per_prompt=args.max_rejected_per_prompt,
        use_numeric_embedding=args.use_numeric_embedding,
        max_numeric_chars=args.max_numeric_chars,
    )
    validation_features, validation_stats = build_tokenized_dpo_features(
        rows=validation_rows,
        tokenizer=tokenizer,
        label_space=label_space,
        max_length=args.max_length,
        max_evidence_items=args.max_evidence_items,
        max_evidence_chars=args.max_evidence_chars,
        max_trace_chars=args.max_trace_chars,
        max_rejected_per_prompt=args.max_rejected_per_prompt,
        use_numeric_embedding=args.use_numeric_embedding,
        max_numeric_chars=args.max_numeric_chars,
    )

    if not train_features:
        raise ValueError("No train DPO pairs remained after preprocessing. Adjust max length or truncation settings.")
    if not validation_features and args.eval_strategy != "no":
        raise ValueError("No validation DPO pairs remained after preprocessing. Adjust validation limits or truncation.")

    model_kwargs = build_model_kwargs(args)
    policy_model = load_policy_model(args, model_kwargs)
    reference_model = load_reference_model(args, model_kwargs)

    training_args = build_training_arguments(TrainingArguments, args=args, output_dir=output_dir)
    data_collator = DPODataCollator(pad_token_id=tokenizer.pad_token_id)
    DPOLoraTrainer = build_dpo_trainer_class(Trainer)
    trainer = build_trainer(
        trainer_cls=DPOLoraTrainer,
        model=policy_model,
        ref_model=reference_model,
        beta=args.beta,
        training_args=training_args,
        train_dataset=TokenizedDPODataset(train_features),
        eval_dataset=TokenizedDPODataset(validation_features) if validation_features else None,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    train_result = trainer.train()

    final_adapter_dir = output_dir / "final_adapter"
    trainer.model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)

    metrics = dict(train_result.metrics)
    train_metrics_path = output_dir / "train_metrics.json"
    with train_metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    eval_metrics_path = output_dir / "eval_metrics.json"
    if validation_features and args.eval_strategy != "no":
        eval_metrics = trainer.evaluate()
        with eval_metrics_path.open("w", encoding="utf-8") as f:
            json.dump(eval_metrics, f, indent=2, ensure_ascii=False)

    save_run_metadata(
        output_dir=output_dir,
        payload={
            "model_id": args.model_id,
            "sft_adapter_path": str(args.sft_adapter_path),
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
                "max_rejected_per_prompt": args.max_rejected_per_prompt,
                "learning_rate": args.learning_rate,
                "num_train_epochs": args.num_train_epochs,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "seed": args.seed,
                "dtype": args.dtype,
                "device_map": args.device_map,
                "attn_implementation": args.attn_implementation,
                "gradient_checkpointing": args.gradient_checkpointing,
                "beta": args.beta,
            },
            "lora_config": {
                "target_modules": parse_target_modules(args.target_modules),
            },
            "quantization": {
                "load_in_4bit": args.load_in_4bit,
                "bnb_4bit_quant_type": args.bnb_4bit_quant_type,
                "bnb_4bit_compute_dtype": args.bnb_4bit_compute_dtype,
                "bnb_4bit_use_double_quant": args.bnb_4bit_use_double_quant,
            },
            "numeric_embedding": {
                "enabled": args.use_numeric_embedding,
            },
            "artifacts": {
                "final_adapter_dir": str(final_adapter_dir),
                "train_metrics": str(train_metrics_path),
                "eval_metrics": str(eval_metrics_path) if validation_features and args.eval_strategy != "no" else None,
            },
        },
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "final_adapter_dir": str(final_adapter_dir),
                "train_pairs": len(train_features),
                "validation_pairs": len(validation_features),
                "train_skipped_overlength": train_stats.num_pairs_skipped_overlength,
                "validation_skipped_overlength": validation_stats.num_pairs_skipped_overlength,
                "train_avg_chosen_length": average_length(train_stats.total_chosen_length, train_stats.num_pairs_built),
                "train_avg_rejected_length": average_length(train_stats.total_rejected_length, train_stats.num_pairs_built),
                "validation_avg_chosen_length": average_length(validation_stats.total_chosen_length, validation_stats.num_pairs_built),
                "validation_avg_rejected_length": average_length(validation_stats.total_rejected_length, validation_stats.num_pairs_built),
                "train_max_chosen_length": train_stats.max_chosen_length,
                "train_max_rejected_length": train_stats.max_rejected_length,
                "validation_max_chosen_length": validation_stats.max_chosen_length,
                "validation_max_rejected_length": validation_stats.max_rejected_length,
                "use_numeric_embedding": args.use_numeric_embedding,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
