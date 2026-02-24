#!/usr/bin/env python3
"""Train mini verifier experiments on 5 claims.

Modes:
- ntp: Next-token prediction on verdict token(s).
- cls_frozen: Linear classification head, frozen backbone.
- cls_unfrozen: Linear classification head, unfrozen backbone.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))

from mini_factcheck_common import (
    TraceClassifier,
    TraceExample,
    build_backbone_model,
    build_tokenizer,
    build_trace_examples,
    load_json_rows,
    resolve_base_model_id,
    save_metadata,
    select_rows,
    to_metadata_dict,
)


DEFAULT_MODES = ("ntp", "cls_frozen", "cls_unfrozen")


class NTPDataset(Dataset):
    def __init__(self, examples: Sequence[TraceExample], tokenizer, max_length: int) -> None:
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        prompt = ex.packed_input + " "
        full_text = prompt + ex.label_text

        full_enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        prompt_enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )

        input_ids = list(full_enc["input_ids"])
        attention_mask = list(full_enc["attention_mask"])
        labels = list(input_ids)

        prompt_len = min(len(prompt_enc["input_ids"]), len(labels))
        for i in range(prompt_len):
            labels[i] = -100
        if all(x == -100 for x in labels):
            labels[-1] = input_ids[-1]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class ClassifierDataset(Dataset):
    def __init__(self, examples: Sequence[TraceExample], tokenizer, max_length: int) -> None:
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex.packed_input,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        return {
            "input_ids": list(enc["input_ids"]),
            "attention_mask": list(enc["attention_mask"]),
            "labels": ex.label_id,
        }


def pad_2d(seqs: Sequence[Sequence[int]], pad_value: int) -> torch.Tensor:
    max_len = max(len(x) for x in seqs)
    out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    for row_idx, row in enumerate(seqs):
        out[row_idx, : len(row)] = torch.tensor(row, dtype=torch.long)
    return out


def ntp_collate(batch: Sequence[dict], pad_token_id: int) -> dict:
    input_ids = pad_2d([x["input_ids"] for x in batch], pad_value=pad_token_id)
    attention_mask = pad_2d([x["attention_mask"] for x in batch], pad_value=0)
    labels = pad_2d([x["labels"] for x in batch], pad_value=-100)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def cls_collate(batch: Sequence[dict], pad_token_id: int) -> dict:
    input_ids = pad_2d([x["input_ids"] for x in batch], pad_value=pad_token_id)
    attention_mask = pad_2d([x["attention_mask"] for x in batch], pad_value=0)
    labels = torch.tensor([x["labels"] for x in batch], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def num_trainable_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def num_total_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def train_ntp(
    *,
    examples: Sequence[TraceExample],
    model_id: str,
    adapter_id: Optional[str],
    output_dir: Path,
    tokenizer,
    epochs: int,
    batch_size: int,
    lr: float,
    max_length: int,
    device: torch.device,
) -> Dict[str, object]:
    dataset = NTPDataset(examples=examples, tokenizer=tokenizer, max_length=max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: ntp_collate(batch, pad_token_id=tokenizer.pad_token_id),
    )

    model = build_backbone_model(base_model_id=model_id, adapter_id=adapter_id, dtype=torch.float32)
    model.to(device)

    optim = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    epoch_losses: List[float] = []
    for epoch in range(epochs):
        total_loss = 0.0
        steps = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = out.loss
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            total_loss += float(loss.item())
            steps += 1
        epoch_losses.append(total_loss / max(1, steps))

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    stats = {
        "mode": "ntp",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "epoch_loss": epoch_losses,
        "trainable_parameters": num_trainable_parameters(model),
        "total_parameters": num_total_parameters(model),
    }

    del model, loader, dataset, optim
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return stats


def train_classifier(
    *,
    examples: Sequence[TraceExample],
    model_id: str,
    adapter_id: Optional[str],
    output_dir: Path,
    tokenizer,
    epochs: int,
    batch_size: int,
    lr: float,
    max_length: int,
    device: torch.device,
    freeze_backbone: bool,
    num_labels: int,
) -> Dict[str, object]:
    dataset = ClassifierDataset(examples=examples, tokenizer=tokenizer, max_length=max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: cls_collate(batch, pad_token_id=tokenizer.pad_token_id),
    )

    backbone = build_backbone_model(base_model_id=model_id, adapter_id=adapter_id, dtype=torch.float32)
    for p in backbone.parameters():
        p.requires_grad = not freeze_backbone

    model = TraceClassifier(backbone=backbone, num_labels=num_labels)
    model.to(device)

    optim = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    epoch_losses: List[float] = []
    for epoch in range(epochs):
        total_loss = 0.0
        steps = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = out["loss"]
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            total_loss += float(loss.item())
            steps += 1
        epoch_losses.append(total_loss / max(1, steps))

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "classifier.pt")
    tokenizer.save_pretrained(output_dir)

    mode_name = "cls_frozen" if freeze_backbone else "cls_unfrozen"
    stats = {
        "mode": mode_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "epoch_loss": epoch_losses,
        "trainable_parameters": num_trainable_parameters(model),
        "total_parameters": num_total_parameters(model),
    }

    del model, backbone, loader, dataset, optim
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train mini verifier experiments.")
    parser.add_argument("--dataset-path", type=Path, default=Path("dataset/english/train.json"))
    parser.add_argument("--num-claims", type=int, default=5)
    parser.add_argument("--max-traces-per-claim", type=int, default=20)
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--adapter-id", type=str, default="IlyaGusev/saiga_mistral_7b_lora")
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lr-ntp", type=float, default=2e-5)
    parser.add_argument("--lr-cls", type=float, default=1e-4)
    parser.add_argument("--output-root", type=Path, default=Path("checkpoints/mini_factcheck"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def validate_modes(modes: Sequence[str]) -> List[str]:
    clean = [m.strip().lower() for m in modes]
    bad = [m for m in clean if m not in DEFAULT_MODES]
    if bad:
        raise ValueError(f"Unsupported modes: {bad}. Allowed: {DEFAULT_MODES}")
    return clean


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    modes = validate_modes(args.modes)
    device = choose_device(args.device)
    adapter_id = args.adapter_id.strip() if args.adapter_id else None
    if adapter_id == "":
        adapter_id = None

    rows = load_json_rows(args.dataset_path)
    selected_rows = select_rows(rows=rows, num_claims=args.num_claims)
    examples, label_to_id = build_trace_examples(
        rows=selected_rows,
        max_traces_per_claim=args.max_traces_per_claim,
    )

    model_id = resolve_base_model_id(explicit_base_model_id=args.model_id, adapter_id=adapter_id)
    tokenizer = build_tokenizer(base_model_id=model_id, adapter_id=adapter_id)

    args.output_root.mkdir(parents=True, exist_ok=True)
    save_metadata(
        args.output_root / "dataset_snapshot.json",
        {
            "dataset_path": str(args.dataset_path),
            "num_claims": args.num_claims,
            "num_examples": len(examples),
            "modes": modes,
        },
    )

    train_summaries: List[Dict[str, object]] = []
    for mode in modes:
        mode_dir = args.output_root / mode
        metadata = to_metadata_dict(
            model_id=model_id,
            adapter_id=adapter_id,
            label_to_id=label_to_id,
            num_claims=args.num_claims,
            num_examples=len(examples),
            max_traces_per_claim=args.max_traces_per_claim,
        )
        metadata["mode"] = mode

        if mode == "ntp":
            stats = train_ntp(
                examples=examples,
                model_id=model_id,
                adapter_id=adapter_id,
                output_dir=mode_dir,
                tokenizer=tokenizer,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr_ntp,
                max_length=args.max_length,
                device=device,
            )
        elif mode == "cls_frozen":
            stats = train_classifier(
                examples=examples,
                model_id=model_id,
                adapter_id=adapter_id,
                output_dir=mode_dir,
                tokenizer=tokenizer,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr_cls,
                max_length=args.max_length,
                device=device,
                freeze_backbone=True,
                num_labels=len(label_to_id),
            )
        else:
            stats = train_classifier(
                examples=examples,
                model_id=model_id,
                adapter_id=adapter_id,
                output_dir=mode_dir,
                tokenizer=tokenizer,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr_cls,
                max_length=args.max_length,
                device=device,
                freeze_backbone=False,
                num_labels=len(label_to_id),
            )

        save_metadata(mode_dir / "metadata.json", metadata)
        save_metadata(mode_dir / "train_stats.json", stats)
        train_summaries.append(stats)

    summary = {
        "checkpoint_root": str(args.output_root),
        "model_id": model_id,
        "adapter_id": adapter_id,
        "modes": modes,
        "device": str(device),
        "num_claims": args.num_claims,
        "num_examples": len(examples),
        "label_space": sorted(label_to_id.keys()),
        "train_summaries": train_summaries,
    }
    with (args.output_root / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
