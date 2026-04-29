#!/usr/bin/env python3
"""Utilities for trace-level binary scorer training and inference."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

try:
    from numeric_embedding_utils import (
        NUMERIC_PATTERN,
        annotate_numeric_text,
        build_numeric_dense_features,
        canonicalize_numeric_surface,
    )
    from task2_ranking_utils import render_list_block, truncate_text
    from task2_utils import normalize_label
except ImportError:  # pragma: no cover - import path fallback
    from scripts.numeric_embedding_utils import (
        NUMERIC_PATTERN,
        annotate_numeric_text,
        build_numeric_dense_features,
        canonicalize_numeric_surface,
    )
    from scripts.task2_ranking_utils import render_list_block, truncate_text
    from scripts.task2_utils import normalize_label


TRACE_LABEL_PATTERN = re.compile(
    r"(?:#+\s*)?\*{0,2}\[?\s*Label\s*\]?\s*:\s*\*{0,2}\s*"
    r"(true|false|conflicting|unknown|supports|refutes|not enough information)\b\*{0,2}",
    flags=re.IGNORECASE,
)
TRACE_JUSTIFICATION_MARKER_PATTERN = re.compile(
    r"\[?\s*Justification\s*\]?:\s*",
    flags=re.IGNORECASE,
)
TRACE_LABEL_MARKER_PATTERN = re.compile(
    r"(?:#+\s*)?\*{0,2}(?:Final\s+)?\[?\s*Label\s*\]?\*{0,2}\s*:\s*",
    flags=re.IGNORECASE,
)
TRACE_WHITESPACE_PATTERN = re.compile(r"\s+")
TRACE_SCORER_TEMPLATE = "Claim: {claim}\n{body}"


@dataclass
class TraceScorerPreprocessStats:
    num_claims_seen: int = 0
    num_examples_built: int = 0
    num_examples_skipped_empty: int = 0
    num_positive_examples: int = 0
    num_negative_examples: int = 0
    total_sequence_length: int = 0
    max_sequence_length: int = 0


class TokenizedTraceScorerDataset(Dataset):
    def __init__(self, features: Sequence[Dict[str, object]]) -> None:
        self.features = list(features)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return self.features[idx]


class TraceScorerDataCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: Sequence[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []
        numeric_mask = []
        numeric_char_ids = []
        has_labels = "labels" in features[0]
        has_numeric = "numeric_mask" in features[0] and "numeric_char_ids" in features[0]
        max_numeric_chars = (
            len(features[0]["numeric_char_ids"][0])
            if has_numeric and features[0]["numeric_char_ids"]
            else 0
        )

        for feature in features:
            pad_len = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            if has_labels:
                labels.append(int(feature["labels"]))
            if has_numeric:
                numeric_mask.append(feature["numeric_mask"] + [0] * pad_len)
                numeric_char_ids.append(
                    feature["numeric_char_ids"] + ([[0] * max_numeric_chars] * pad_len)
                )

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
        if has_labels:
            batch["labels"] = torch.tensor(labels, dtype=torch.long)
        if has_numeric:
            batch["numeric_mask"] = torch.tensor(numeric_mask, dtype=torch.bool)
            batch["numeric_char_ids"] = torch.tensor(numeric_char_ids, dtype=torch.long)
        return batch


def clean_reasoning_trace(trace: str) -> str:
    cleaned = TRACE_LABEL_PATTERN.sub("", str(trace))
    cleaned = TRACE_LABEL_MARKER_PATTERN.sub("", cleaned)
    cleaned = TRACE_JUSTIFICATION_MARKER_PATTERN.sub("", cleaned)
    return TRACE_WHITESPACE_PATTERN.sub(" ", cleaned).strip()


def extract_normalized_number_hints(text: str, max_numbers: int) -> List[str]:
    hints: List[str] = []
    seen = set()
    text = str(text)
    if max_numbers <= 0:
        return hints

    for match in NUMERIC_PATTERN.finditer(text):
        surface = match.group(0).strip()
        canonical = canonicalize_numeric_surface(surface)
        if canonical is None:
            continue
        if is_probable_list_marker(text, match, canonical):
            continue
        hint = canonical if surface == canonical else f"{surface}={canonical}"
        if hint in seen:
            continue
        seen.add(hint)
        hints.append(hint)
        if len(hints) >= max_numbers:
            break

    return hints


def is_probable_list_marker(text: str, match: re.Match[str], canonical: str) -> bool:
    try:
        value = float(canonical)
    except ValueError:
        return False
    if not value.is_integer() or not 1 <= value <= 30:
        return False

    end = match.end()
    next_char = text[end : end + 1]
    if next_char not in {".", ")", ":"}:
        return False

    start = match.start()
    previous = text[max(0, start - 4) : start]
    return start == 0 or previous.endswith((" ", "\n", "\t", "- "))


def render_normalized_number_hints(
    *,
    claim: str,
    evidence_items: Sequence[str],
    justification: str,
    max_numbers: int,
) -> str:
    sections = [
        ("Claim", claim),
        ("Evidence", " ".join(str(item) for item in evidence_items)),
        ("Trace", justification),
    ]
    lines: List[str] = []
    for section_name, section_text in sections:
        hints = extract_normalized_number_hints(section_text, max_numbers=max_numbers)
        if hints:
            lines.append(f"{section_name}: {', '.join(hints)}")
    if not lines:
        return ""
    return "Normalized numbers:\n" + "\n".join(lines)


def build_trace_scorer_input_artifacts(
    row: dict,
    trace_index: int,
    *,
    max_claim_chars: int,
    max_evidence_items: int,
    max_evidence_chars: int,
    max_trace_chars: int,
    use_numeric_embedding: bool,
    append_normalized_numbers: bool = False,
    max_normalized_numbers: int = 20,
) -> Dict[str, object]:
    evidences = row.get("evidences", []) or []
    verdict_list = row.get("Verdict_list", []) or []
    reasoning_traces = row.get("Reasoning_traces", []) or []
    claim = truncate_text(str(row.get("claim", "")).strip(), max_claim_chars)
    verdict = str(verdict_list[trace_index] if trace_index < len(verdict_list) else "").strip()
    justification = truncate_text(
        clean_reasoning_trace(reasoning_traces[trace_index] if trace_index < len(reasoning_traces) else ""),
        max_trace_chars,
    )
    capped_evidences = list(evidences[:max_evidence_items]) if max_evidence_items > 0 else []
    numeric_canonicals: List[str] = []

    def maybe_annotate(text: str) -> str:
        if not use_numeric_embedding:
            return text
        annotated, canonicals = annotate_numeric_text(text)
        numeric_canonicals.extend(canonicals)
        return annotated

    raw_evidence_items = [truncate_text(str(item), max_evidence_chars) for item in capped_evidences]
    evidence_items = [maybe_annotate(item) for item in raw_evidence_items]
    body_parts: List[str] = []
    if evidence_items:
        body_parts.append(
            render_list_block(
                items=evidence_items,
                max_items=0,
                max_chars=max_evidence_chars,
                header="Evidence snippets:",
                truncate_items=False,
            )
        )
    if append_normalized_numbers:
        normalized_number_hints = render_normalized_number_hints(
            claim=claim,
            evidence_items=raw_evidence_items,
            justification=justification,
            max_numbers=max_normalized_numbers,
        )
        if normalized_number_hints:
            body_parts.append(normalized_number_hints)
    body_parts.append(f"Verdict: {maybe_annotate(verdict)}")
    body_parts.append(f"Justification: {maybe_annotate(justification)}")
    text = TRACE_SCORER_TEMPLATE.format(
        claim=maybe_annotate(claim),
        body="\n".join(body_parts),
    )
    label = int(normalize_label(verdict) == normalize_label(row.get("label", "")))
    return {
        "text": text,
        "label": label,
        "trace_index": trace_index,
        "trace_verdict": verdict,
        "cleaned_trace": justification,
        "numeric_canonicals": numeric_canonicals,
    }


def tokenize_trace_scorer_input(
    *,
    tokenizer,
    text: str,
    label: Optional[int],
    max_length: int,
    numeric_canonicals: Optional[Sequence[str]],
    num_token_id: Optional[int],
    max_numeric_chars: int,
) -> Dict[str, object]:
    input_ids = tokenizer(text, add_special_tokens=True, truncation=False).input_ids
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]

    output: Dict[str, object] = {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
    }
    if label is not None:
        output["labels"] = int(label)

    if numeric_canonicals is not None and num_token_id is not None:
        num_positions = [idx for idx, token_id in enumerate(input_ids) if token_id == num_token_id]
        if len(num_positions) > len(numeric_canonicals):
            raise ValueError(
                "Tokenized scorer input contains more <num> tokens than parsed numeric values. "
                f"Found {len(num_positions)} tokens but {len(numeric_canonicals)} values."
            )
        kept_canonicals = list(numeric_canonicals[: len(num_positions)])
        numeric_mask, numeric_char_ids = build_numeric_dense_features(
            sequence_length=len(input_ids),
            positions=num_positions,
            canonicals=kept_canonicals,
            max_numeric_chars=max_numeric_chars,
        )
        output["numeric_mask"] = numeric_mask
        output["numeric_char_ids"] = numeric_char_ids

    return output


def build_trace_scorer_features(
    rows: Sequence[dict],
    *,
    tokenizer,
    max_length: int,
    max_claim_chars: int,
    max_evidence_items: int,
    max_evidence_chars: int,
    max_trace_chars: int,
    use_numeric_embedding: bool,
    max_numeric_chars: int,
    append_normalized_numbers: bool = False,
    max_normalized_numbers: int = 20,
    skip_short_justifications: bool = True,
) -> tuple[list[Dict[str, object]], TraceScorerPreprocessStats]:
    features: List[Dict[str, object]] = []
    stats = TraceScorerPreprocessStats(num_claims_seen=len(rows))
    num_token_id = tokenizer.convert_tokens_to_ids("<num>") if use_numeric_embedding else None

    for dataset_index, row in enumerate(rows):
        traces = row.get("Reasoning_traces", []) or []
        for trace_index in range(len(traces)):
            artifacts = build_trace_scorer_input_artifacts(
                row,
                trace_index,
                max_claim_chars=max_claim_chars,
                max_evidence_items=max_evidence_items,
                max_evidence_chars=max_evidence_chars,
                max_trace_chars=max_trace_chars,
                use_numeric_embedding=use_numeric_embedding,
                append_normalized_numbers=append_normalized_numbers,
                max_normalized_numbers=max_normalized_numbers,
            )
            if skip_short_justifications and len(str(artifacts["cleaned_trace"]).split()) < 3:
                stats.num_examples_skipped_empty += 1
                continue

            encoded = tokenize_trace_scorer_input(
                tokenizer=tokenizer,
                text=str(artifacts["text"]),
                label=int(artifacts["label"]),
                max_length=max_length,
                numeric_canonicals=artifacts["numeric_canonicals"] if use_numeric_embedding else None,
                num_token_id=num_token_id,
                max_numeric_chars=max_numeric_chars,
            )
            encoded["trace_index"] = trace_index
            encoded["dataset_index"] = dataset_index
            encoded["trace_verdict"] = artifacts["trace_verdict"]
            encoded["gold_label"] = row.get("label", "")
            encoded["verdict_list"] = [str(verdict) for verdict in (row.get("Verdict_list", []) or [])]
            encoded["num_traces"] = len(traces)
            features.append(encoded)
            stats.num_examples_built += 1
            stats.num_positive_examples += int(artifacts["label"])
            stats.num_negative_examples += int(not artifacts["label"])
            stats.total_sequence_length += len(encoded["input_ids"])
            stats.max_sequence_length = max(stats.max_sequence_length, len(encoded["input_ids"]))

    return features, stats


def average_sequence_length(stats: TraceScorerPreprocessStats) -> float:
    if stats.num_examples_built == 0:
        return 0.0
    return stats.total_sequence_length / stats.num_examples_built


def logits_to_trace_scores(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1:
        return logits
    if logits.shape[-1] == 1:
        return logits[:, 0]
    if logits.shape[-1] == 2:
        return logits[:, 1] - logits[:, 0]
    raise ValueError(f"Unsupported scorer logits shape: {tuple(logits.shape)}")


def rank_trace_indices_from_scores(score_list: Sequence[float]) -> List[int]:
    return sorted(range(len(score_list)), key=lambda idx: (-float(score_list[idx]), idx))


def stats_as_dict(stats: TraceScorerPreprocessStats) -> Dict[str, object]:
    return asdict(stats)
