#!/usr/bin/env python3
"""Shared helpers for mini fact-checking verifier experiments."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftConfig, PeftModel
except ImportError:  # pragma: no cover - runtime dependency check.
    PeftConfig = None
    PeftModel = None


VERDICT_SYNONYMS = {
    "supports": "true",
    "supported": "true",
    "entails": "true",
    "true": "true",
    "refutes": "false",
    "refuted": "false",
    "false": "false",
    "conflicting": "conflicting",
    "conflict": "conflicting",
    "partially true": "conflicting",
    "partially false": "conflicting",
    "mixed": "conflicting",
}


@dataclass
class TraceExample:
    claim_idx: int
    trace_idx: int
    packed_input: str
    label_text: str
    label_id: int
    claim_label: str


def normalize_verdict(label: str) -> str:
    clean = re.sub(r"\s+", " ", str(label).strip().lower())
    return VERDICT_SYNONYMS.get(clean, clean)


def load_json_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def select_rows(rows: Sequence[dict], num_claims: int) -> List[dict]:
    if num_claims <= 0:
        raise ValueError("--num-claims must be > 0")
    if num_claims > len(rows):
        raise ValueError(f"Requested {num_claims} claims, but dataset has {len(rows)}")
    return list(rows[:num_claims])


def _shorten(text: str, max_chars: int) -> str:
    clean = " ".join(str(text).split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max(0, max_chars - 3)].rstrip() + "..."


def pack_trace_input(
    claim: str,
    evidences: Iterable[str],
    trace: str,
    max_claim_chars: int = 600,
    max_evidence_chars: int = 1200,
    max_trace_chars: int = 1200,
) -> str:
    claim_norm = _shorten(claim, max_claim_chars)
    evidence_norm = _shorten(" ".join([str(e) for e in evidences if e]), max_evidence_chars)
    trace_norm = _shorten(trace, max_trace_chars)
    return (
        f"[CLAIM] {claim_norm}\n"
        f"[EVIDENCE] {evidence_norm}\n"
        f"[TRACE] {trace_norm}\n"
        f"[VERDICT]"
    )


def build_trace_examples(rows: Sequence[dict], max_traces_per_claim: Optional[int]) -> Tuple[List[TraceExample], Dict[str, int]]:
    raw_labels: List[str] = []
    packed_rows: List[Tuple[int, int, str, str, str]] = []

    for claim_idx, row in enumerate(rows):
        claim = str(row.get("claim", ""))
        evidences = row.get("evidences", []) or []
        traces = row.get("Reasoning_traces", []) or []
        verdict_list = row.get("Verdict_list", []) or []
        claim_label = normalize_verdict(str(row.get("label", "")))

        limit = len(traces) if max_traces_per_claim is None else min(len(traces), max_traces_per_claim)
        for trace_idx in range(limit):
            trace = traces[trace_idx]
            trace_verdict = verdict_list[trace_idx] if trace_idx < len(verdict_list) else claim_label
            label_text = normalize_verdict(str(trace_verdict))
            packed_rows.append(
                (
                    claim_idx,
                    trace_idx,
                    pack_trace_input(claim=claim, evidences=evidences, trace=str(trace)),
                    label_text,
                    claim_label,
                )
            )
            raw_labels.append(label_text)

    if not packed_rows:
        raise ValueError("No trace examples found in selected rows.")

    label_space = sorted(set(raw_labels))
    label_to_id = {label: idx for idx, label in enumerate(label_space)}
    examples = [
        TraceExample(
            claim_idx=claim_idx,
            trace_idx=trace_idx,
            packed_input=packed_input,
            label_text=label_text,
            label_id=label_to_id[label_text],
            claim_label=claim_label,
        )
        for claim_idx, trace_idx, packed_input, label_text, claim_label in packed_rows
    ]
    return examples, label_to_id


def require_dependencies_for_lora(adapter_id: Optional[str]) -> None:
    if adapter_id and (PeftConfig is None or PeftModel is None):
        raise ImportError(
            "LoRA adapter requested but `peft` is not installed. "
            "Install dependencies from requirements.txt."
        )


def resolve_base_model_id(explicit_base_model_id: Optional[str], adapter_id: Optional[str]) -> str:
    if explicit_base_model_id:
        return explicit_base_model_id
    if not adapter_id:
        raise ValueError("Provide --model-id or --adapter-id.")
    require_dependencies_for_lora(adapter_id=adapter_id)
    peft_cfg = PeftConfig.from_pretrained(adapter_id)
    if not peft_cfg.base_model_name_or_path:
        raise ValueError(f"Could not resolve base model for adapter: {adapter_id}")
    return peft_cfg.base_model_name_or_path


def build_tokenizer(base_model_id: str, adapter_id: Optional[str]) -> AutoTokenizer:
    candidates = [adapter_id, base_model_id] if adapter_id else [base_model_id]
    last_exc: Optional[Exception] = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            tokenizer = AutoTokenizer.from_pretrained(candidate, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        except Exception as exc:  # pragma: no cover - external dependency.
            last_exc = exc
    raise RuntimeError(f"Failed to load tokenizer from {candidates}: {last_exc}")


def build_backbone_model(
    base_model_id: str,
    adapter_id: Optional[str],
    dtype: torch.dtype,
) -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=dtype)
    if adapter_id:
        require_dependencies_for_lora(adapter_id=adapter_id)
        model = PeftModel.from_pretrained(model, adapter_id, is_trainable=True)
    return model


class TraceClassifier(nn.Module):
    """Linear head on top of a CausalLM backbone."""

    def __init__(self, backbone: nn.Module, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.backbone = backbone
        hidden_size = getattr(backbone.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Backbone model does not expose config.hidden_size")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = outputs.hidden_states[-1]  # [batch, seq, hidden]
        last_positions = attention_mask.long().sum(dim=1) - 1
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        pooled = hidden[batch_idx, last_positions]
        logits = self.classifier(self.dropout(pooled))

        out: Dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
            out["loss"] = loss
        return out


def to_metadata_dict(
    *,
    model_id: str,
    adapter_id: Optional[str],
    label_to_id: Dict[str, int],
    num_claims: int,
    num_examples: int,
    max_traces_per_claim: Optional[int],
) -> Dict[str, object]:
    return {
        "model_id": model_id,
        "adapter_id": adapter_id,
        "label_to_id": label_to_id,
        "id_to_label": {str(v): k for k, v in label_to_id.items()},
        "num_claims": num_claims,
        "num_examples": num_examples,
        "max_traces_per_claim": max_traces_per_claim,
    }


def save_metadata(path: Path, metadata: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def load_metadata(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def trace_examples_to_jsonable(examples: Sequence[TraceExample]) -> List[dict]:
    return [asdict(x) for x in examples]


def parse_generated_label(generated_text: str, label_to_id: Dict[str, int]) -> str:
    clean = normalize_verdict(generated_text)
    if clean in label_to_id:
        return clean
    for label in label_to_id:
        if label in clean:
            return label
    # fallback to first label for stability
    return next(iter(label_to_id.keys()))
