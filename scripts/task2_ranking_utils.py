#!/usr/bin/env python3
"""Shared formatting and parsing helpers for CLEF Task 2 ranking generation."""

from __future__ import annotations

import ast
import json
import random
import re
from typing import Dict, List, Optional, Sequence

import torch

try:
    from task2_utils import normalize_label
except ImportError:  # pragma: no cover - import path fallback
    from scripts.task2_utils import normalize_label


DEFAULT_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_SYSTEM_PROMPT = "You are a careful multilingual fact-checking assistant. Return strict JSON only."


def resolve_dtype(name: str):
    lowered = name.strip().lower()
    if lowered == "auto":
        return "auto"
    if lowered == "bfloat16":
        return torch.bfloat16
    if lowered == "float16":
        return torch.float16
    if lowered == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def truncate_text(text: str, max_chars: int) -> str:
    text = str(text).strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def render_list_block(items: Sequence[str], max_items: int, max_chars: int, header: str) -> str:
    if not items:
        return f"{header}\n- None"

    capped_items = list(items[:max_items]) if max_items > 0 else list(items)
    lines = [header]
    for idx, item in enumerate(capped_items):
        lines.append(f"[{idx}] {truncate_text(str(item), max_chars)}")
    if max_items > 0 and len(items) > max_items:
        lines.append(f"... {len(items) - max_items} more omitted")
    return "\n".join(lines)


def infer_label_space(rows: Sequence[dict]) -> List[str]:
    labels = sorted({str(row.get("label", "")).strip() for row in rows if str(row.get("label", "")).strip()})
    return labels


def build_prompt(
    row: dict,
    label_space: Sequence[str],
    max_evidence_items: int,
    max_evidence_chars: int,
    max_trace_chars: int,
) -> str:
    claim = str(row.get("claim", "")).strip()
    evidences = row.get("evidences", []) or []
    traces = row.get("Reasoning_traces", []) or []

    evidence_block = render_list_block(
        items=[str(x) for x in evidences],
        max_items=max_evidence_items,
        max_chars=max_evidence_chars,
        header="Evidence snippets:",
    )
    trace_block = render_list_block(
        items=[str(x) for x in traces],
        max_items=0,
        max_chars=max_trace_chars,
        header="Candidate reasoning traces:",
    )
    allowed_labels = ", ".join(label_space)

    return (
        "You are ranking candidate reasoning traces for multilingual fact-checking.\n"
        "Use the claim and the evidence snippets to judge which reasoning traces are most reliable.\n"
        "Rank all traces from best to worst.\n"
        "Then predict the final verdict for the claim.\n\n"
        f"Allowed verdict labels: {allowed_labels}\n\n"
        "Return strict JSON only in this exact schema:\n"
        '{"ranked_trace_indices":[0,1,2],"predicted_verdict":"False"}\n'
        "Do not include markdown. Do not include any explanation.\n\n"
        f"Claim:\n{claim}\n\n"
        f"{evidence_block}\n\n"
        f"{trace_block}\n"
    )


def build_prompt_messages(prompt: str) -> List[dict]:
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def render_chat_prompt(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        build_prompt_messages(prompt),
        tokenize=False,
        add_generation_prompt=True,
    )


def render_chat_transcript(tokenizer, prompt: str, assistant_response: str) -> str:
    return tokenizer.apply_chat_template(
        build_prompt_messages(prompt) + [{"role": "assistant", "content": assistant_response}],
        tokenize=False,
        add_generation_prompt=False,
    )


def tokenize_supervised_example(
    tokenizer,
    prompt: str,
    assistant_response: str,
    max_length: int,
) -> Optional[Dict[str, List[int]]]:
    prompt_text = render_chat_prompt(tokenizer, prompt)
    full_text = render_chat_transcript(tokenizer, prompt, assistant_response)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids

    if not full_ids:
        return None

    prompt_len = len(prompt_ids)
    if len(full_ids) > max_length:
        overflow = len(full_ids) - max_length
        if overflow >= prompt_len:
            return None
        full_ids = full_ids[overflow:]
        prompt_len -= overflow

    attention_mask = [1] * len(full_ids)
    labels = list(full_ids)
    for idx in range(min(prompt_len, len(labels))):
        labels[idx] = -100

    return {
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def extract_balanced_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def parse_ranked_trace_indices(raw_value, num_traces: int) -> List[int]:
    ints: List[int] = []

    if isinstance(raw_value, (list, tuple)):
        for item in raw_value:
            try:
                ints.append(int(item))
            except (TypeError, ValueError):
                continue
    elif isinstance(raw_value, str):
        ints = [int(x) for x in re.findall(r"-?\d+", raw_value)]

    seen = set()
    ranked: List[int] = []
    for idx in ints:
        if 0 <= idx < num_traces and idx not in seen:
            ranked.append(idx)
            seen.add(idx)

    for idx in range(num_traces):
        if idx not in seen:
            ranked.append(idx)

    return ranked


def parse_predicted_verdict(raw_value, label_space: Sequence[str]) -> str:
    canonical = {normalize_label(label): label for label in label_space}
    if raw_value is not None:
        label = canonical.get(normalize_label(raw_value))
        if label is not None:
            return label
    return ""


def parse_response(text: str, label_space: Sequence[str], num_traces: int) -> Dict[str, object]:
    parsed: Dict[str, object] = {}
    json_blob = extract_balanced_json_object(text)
    parse_error = None

    if json_blob is not None:
        try:
            parsed = json.loads(json_blob)
        except json.JSONDecodeError as exc:
            parse_error = str(exc)
            try:
                value = ast.literal_eval(json_blob)
                if isinstance(value, dict):
                    parsed = value
            except (SyntaxError, ValueError) as inner_exc:
                parse_error = str(inner_exc)

    ranked = parse_ranked_trace_indices(parsed.get("ranked_trace_indices"), num_traces)
    verdict = parse_predicted_verdict(parsed.get("predicted_verdict"), label_space)

    if not verdict:
        lowered_response = normalize_label(text)
        for label in label_space:
            if normalize_label(label) in lowered_response:
                verdict = label
                break

    return {
        "ranked_trace_indices": ranked,
        "predicted_verdict": verdict,
        "json_found": json_blob is not None,
        "json_parse_error": parse_error,
    }


def render_prediction_json(ranked_trace_indices: Sequence[int], predicted_verdict: str) -> str:
    payload = {
        "ranked_trace_indices": [int(idx) for idx in ranked_trace_indices],
        "predicted_verdict": str(predicted_verdict),
    }
    return json.dumps(payload, ensure_ascii=False)


def build_proxy_target(row: dict) -> Dict[str, object]:
    gold_label = str(row.get("label", "")).strip()
    normalized_gold = normalize_label(gold_label)
    verdict_list = row.get("Verdict_list", []) or []

    positives = [idx for idx, verdict in enumerate(verdict_list) if normalize_label(verdict) == normalized_gold]
    negatives = [idx for idx, verdict in enumerate(verdict_list) if normalize_label(verdict) != normalized_gold]
    ranked = positives + negatives

    if not ranked:
        ranked = list(range(len(row.get("Reasoning_traces", []) or [])))

    return {
        "ranked_trace_indices": ranked,
        "predicted_verdict": gold_label,
    }


def choose_alternative_label(
    gold_label: str,
    label_space: Sequence[str],
    preferred_label: Optional[str] = None,
) -> str:
    canonical = {normalize_label(label): label for label in label_space}
    normalized_gold = normalize_label(gold_label)

    if preferred_label is not None:
        preferred = canonical.get(normalize_label(preferred_label))
        if preferred is not None and normalize_label(preferred) != normalized_gold:
            return preferred

    for label in label_space:
        if normalize_label(label) != normalized_gold:
            return label

    return gold_label


def build_rejected_targets(
    row: dict,
    label_space: Sequence[str],
    max_rejected_per_prompt: int = 2,
) -> List[Dict[str, object]]:
    chosen = build_proxy_target(row)
    chosen_ranking = list(chosen["ranked_trace_indices"])
    chosen_verdict = str(chosen["predicted_verdict"]).strip()
    normalized_gold = normalize_label(chosen_verdict)
    verdict_list = row.get("Verdict_list", []) or []
    num_traces = len(row.get("Reasoning_traces", []) or [])

    positives = [idx for idx, verdict in enumerate(verdict_list) if normalize_label(verdict) == normalized_gold]
    negatives = [idx for idx, verdict in enumerate(verdict_list) if normalize_label(verdict) != normalized_gold]
    fallback_wrong_label = choose_alternative_label(
        gold_label=chosen_verdict,
        label_space=label_space,
        preferred_label=row.get("verdict"),
    )

    candidates: List[Dict[str, object]] = []

    if negatives and chosen_ranking:
        negative_first = negatives[0]
        ranking = [negative_first] + [idx for idx in chosen_ranking if idx != negative_first]
        candidates.append(
            {
                "ranked_trace_indices": ranking,
                "predicted_verdict": chosen_verdict,
            }
        )

    if num_traces > 1 and len(chosen_ranking) > 1:
        swapped = list(chosen_ranking)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        candidates.append(
            {
                "ranked_trace_indices": swapped,
                "predicted_verdict": chosen_verdict,
            }
        )

    if fallback_wrong_label != chosen_verdict:
        candidates.append(
            {
                "ranked_trace_indices": list(chosen_ranking),
                "predicted_verdict": fallback_wrong_label,
            }
        )

    if negatives and fallback_wrong_label != chosen_verdict and chosen_ranking:
        negative_first = negatives[0]
        ranking = [negative_first] + [idx for idx in chosen_ranking if idx != negative_first]
        candidates.append(
            {
                "ranked_trace_indices": ranking,
                "predicted_verdict": fallback_wrong_label,
            }
        )

    if len(candidates) == 0 and len(chosen_ranking) > 1:
        candidates.append(
            {
                "ranked_trace_indices": list(reversed(chosen_ranking)),
                "predicted_verdict": chosen_verdict,
            }
        )

    if len(candidates) == 0:
        candidates.append(
            {
                "ranked_trace_indices": list(range(num_traces)),
                "predicted_verdict": fallback_wrong_label,
            }
        )

    deduped: List[Dict[str, object]] = []
    seen = set()
    for candidate in candidates:
        ranking_key = tuple(int(idx) for idx in candidate["ranked_trace_indices"])
        key = (ranking_key, normalize_label(candidate["predicted_verdict"]))
        if ranking_key == tuple(chosen_ranking) and normalize_label(candidate["predicted_verdict"]) == normalized_gold:
            continue
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
        if len(deduped) >= max_rejected_per_prompt:
            break

    return deduped


def build_dpo_pairs(
    row: dict,
    label_space: Sequence[str],
    max_rejected_per_prompt: int = 2,
) -> List[Dict[str, str]]:
    chosen = build_proxy_target(row)
    chosen_text = render_prediction_json(
        ranked_trace_indices=chosen["ranked_trace_indices"],
        predicted_verdict=chosen["predicted_verdict"],
    )

    pairs: List[Dict[str, str]] = []
    for rejected in build_rejected_targets(
        row=row,
        label_space=label_space,
        max_rejected_per_prompt=max_rejected_per_prompt,
    ):
        rejected_text = render_prediction_json(
            ranked_trace_indices=rejected["ranked_trace_indices"],
            predicted_verdict=rejected["predicted_verdict"],
        )
        pairs.append(
            {
                "chosen": chosen_text,
                "rejected": rejected_text,
            }
        )

    return pairs


def build_sft_completion(row: dict) -> str:
    target = build_proxy_target(row)
    return render_prediction_json(
        ranked_trace_indices=target["ranked_trace_indices"],
        predicted_verdict=target["predicted_verdict"],
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model_input_device(model) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device


def generate_prediction(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    messages = build_prompt_messages(prompt)
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer(rendered, return_tensors="pt")
    model_inputs = model_inputs.to(get_model_input_device(model))

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.inference_mode():
        outputs = model.generate(**model_inputs, **generation_kwargs)

    generated = outputs[0][model_inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()
