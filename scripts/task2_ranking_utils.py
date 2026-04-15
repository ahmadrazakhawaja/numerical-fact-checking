#!/usr/bin/env python3
"""Shared formatting and parsing helpers for CLEF Task 2 ranking generation."""

from __future__ import annotations

import ast
import json
import random
import re
from typing import Dict, List, Optional, Sequence, Tuple

import torch

try:
    from numeric_embedding_utils import annotate_numeric_text, build_numeric_prompt_features
    from task2_utils import normalize_label
except ImportError:  # pragma: no cover - import path fallback
    from scripts.numeric_embedding_utils import annotate_numeric_text, build_numeric_prompt_features
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


def render_list_block(
    items: Sequence[str],
    max_items: int,
    max_chars: int,
    header: str,
    truncate_items: bool = True,
) -> str:
    if not items:
        return f"{header}\n- None"

    capped_items = list(items[:max_items]) if max_items > 0 else list(items)
    lines = [header]
    for idx, item in enumerate(capped_items):
        rendered_item = truncate_text(str(item), max_chars) if truncate_items else str(item)
        lines.append(f"[{idx}] {rendered_item}")
    if max_items > 0 and len(items) > max_items:
        lines.append(f"... {len(items) - max_items} more omitted")
    return "\n".join(lines)


def infer_label_space(rows: Sequence[dict]) -> List[str]:
    labels = sorted({str(row.get("label", "")).strip() for row in rows if str(row.get("label", "")).strip()})
    return labels


def build_prompt_artifacts(
    row: dict,
    label_space: Sequence[str],
    max_evidence_items: int,
    max_evidence_chars: int,
    max_trace_chars: int,
    use_numeric_embedding: bool = False,
) -> Dict[str, object]:
    claim = str(row.get("claim", "")).strip()
    evidences = row.get("evidences", []) or []
    traces = row.get("Reasoning_traces", []) or []
    numeric_canonicals: List[str] = []

    def maybe_annotate(text: str, use_numeric_embedding: bool) -> str:
        nonlocal numeric_canonicals
        if not use_numeric_embedding:
            return str(text)
        annotated, canonicals = annotate_numeric_text(str(text))
        numeric_canonicals.extend(canonicals)
        return annotated

    claim_text = maybe_annotate(claim, use_numeric_embedding)
    capped_evidences = list(evidences[:max_evidence_items]) if max_evidence_items > 0 else list(evidences)
    evidence_display_items = [truncate_text(str(x), max_evidence_chars) for x in capped_evidences]
    trace_display_items = [truncate_text(str(x), max_trace_chars) for x in traces]
    evidence_items = [maybe_annotate(x, use_numeric_embedding) for x in evidence_display_items]
    trace_items = [maybe_annotate(x, use_numeric_embedding) for x in trace_display_items]
    num_traces = len(trace_items)
    valid_indices = ", ".join(str(idx) for idx in range(num_traces))

    evidence_block = render_list_block(
        items=evidence_items,
        max_items=max_evidence_items,
        max_chars=max_evidence_chars,
        header="Evidence snippets:",
        truncate_items=False,
    )
    trace_block = render_list_block(
        items=trace_items,
        max_items=0,
        max_chars=max_trace_chars,
        header="Candidate reasoning traces:",
        truncate_items=False,
    )
    allowed_labels = ", ".join(label_space)

    prompt = (
        "You are ranking candidate reasoning traces for multilingual fact-checking.\n"
        "Use the claim and the evidence snippets to judge which reasoning traces are most reliable.\n"
        "Rank all candidate traces from best to worst. The input order is arbitrary and must not be copied unless it is truly the best ranking.\n"
        "A better trace is one that is better supported by the evidence, checks the claim more directly, and reaches a more reliable verdict.\n\n"
        f"There are {num_traces} traces indexed 0 through {max(num_traces - 1, 0)}.\n"
        f"The ranked_trace_indices value must be a permutation containing each of these indices exactly once: [{valid_indices}].\n"
        "Do not omit indices. Do not repeat indices. Do not use placeholder indices.\n"
        "Then predict the final verdict for the claim.\n\n"
        f"Allowed verdict labels: {allowed_labels}\n\n"
        "Return strict JSON only with this schema:\n"
        '{"ranked_trace_indices":[...],"predicted_verdict":"<one allowed label>"}\n'
        "Do not include markdown. Do not include any explanation.\n\n"
        f"Claim:\n{claim_text}\n\n"
        f"{evidence_block}\n\n"
        f"{trace_block}\n"
    )
    return {
        "prompt": prompt,
        "prompt_numeric_canonicals": numeric_canonicals,
    }


def build_prompt(
    row: dict,
    label_space: Sequence[str],
    max_evidence_items: int,
    max_evidence_chars: int,
    max_trace_chars: int,
    use_numeric_embedding: bool = False,
) -> str:
    return build_prompt_artifacts(
        row=row,
        label_space=label_space,
        max_evidence_items=max_evidence_items,
        max_evidence_chars=max_evidence_chars,
        max_trace_chars=max_trace_chars,
        use_numeric_embedding=use_numeric_embedding,
    )["prompt"]


def build_pairwise_prompt_artifacts(
    row: dict,
    max_evidence_items: int,
    max_evidence_chars: int,
    max_trace_chars: int,
    left_index: int,
    right_index: int,
    use_numeric_embedding: bool = False,
) -> Dict[str, object]:
    claim = str(row.get("claim", "")).strip()
    evidences = row.get("evidences", []) or []
    traces = row.get("Reasoning_traces", []) or []
    numeric_canonicals: List[str] = []

    def maybe_annotate(text: str) -> str:
        nonlocal numeric_canonicals
        if not use_numeric_embedding:
            return str(text)
        annotated, canonicals = annotate_numeric_text(str(text))
        numeric_canonicals.extend(canonicals)
        return annotated

    claim_text = maybe_annotate(claim)
    capped_evidences = list(evidences[:max_evidence_items]) if max_evidence_items > 0 else list(evidences)
    evidence_items = [
        maybe_annotate(truncate_text(str(item), max_evidence_chars))
        for item in capped_evidences
    ]
    left_trace = maybe_annotate(truncate_text(str(traces[left_index]), max_trace_chars))
    right_trace = maybe_annotate(truncate_text(str(traces[right_index]), max_trace_chars))
    evidence_block = render_list_block(
        items=evidence_items,
        max_items=max_evidence_items,
        max_chars=max_evidence_chars,
        header="Evidence snippets:",
        truncate_items=False,
    )

    prompt = (
        "You are comparing two candidate reasoning traces for multilingual fact-checking.\n"
        "Use the claim and evidence snippets to decide which trace is more reliable.\n"
        "Choose exactly one trace: A or B.\n\n"
        "Return strict JSON only in this exact schema:\n"
        '{"preferred_trace":"A"}\n'
        "Allowed values for preferred_trace: A, B.\n"
        "Do not include markdown. Do not include any explanation.\n\n"
        f"Claim:\n{claim_text}\n\n"
        f"{evidence_block}\n\n"
        f"Trace A (original index {left_index}):\n{left_trace}\n\n"
        f"Trace B (original index {right_index}):\n{right_trace}\n"
    )
    return {
        "prompt": prompt,
        "prompt_numeric_canonicals": numeric_canonicals,
    }


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
    prompt_numeric_canonicals: Optional[Sequence[str]] = None,
    num_token_id: Optional[int] = None,
    max_numeric_chars: int = 24,
) -> Optional[Dict[str, List[int]]]:
    prompt_text = render_chat_prompt(tokenizer, prompt)
    full_text = render_chat_transcript(tokenizer, prompt, assistant_response)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids

    if not full_ids:
        return None

    prompt_len = len(prompt_ids)
    overflow = 0
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

    output = {
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    if prompt_numeric_canonicals is not None and num_token_id is not None:
        numeric_mask, numeric_char_ids = build_numeric_prompt_features(
            prompt_ids=prompt_ids,
            prompt_numeric_canonicals=prompt_numeric_canonicals,
            num_token_id=num_token_id,
            max_numeric_chars=max_numeric_chars,
            overflow=overflow,
            sequence_length=len(full_ids),
        )
        output["numeric_mask"] = numeric_mask
        output["numeric_char_ids"] = numeric_char_ids
    return output


def tokenize_prompt_for_generation(
    tokenizer,
    prompt: str,
    prompt_numeric_canonicals: Optional[Sequence[str]] = None,
    num_token_id: Optional[int] = None,
    max_numeric_chars: int = 24,
    max_length: Optional[int] = None,
) -> Dict[str, List[int]]:
    prompt_text = render_chat_prompt(tokenizer, prompt)
    input_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    overflow = 0
    if max_length is not None and len(input_ids) > max_length:
        overflow = len(input_ids) - max_length
        input_ids = input_ids[overflow:]

    output = {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
    }
    if prompt_numeric_canonicals is not None and num_token_id is not None:
        numeric_mask, numeric_char_ids = build_numeric_prompt_features(
            prompt_ids=tokenizer(prompt_text, add_special_tokens=False).input_ids,
            prompt_numeric_canonicals=prompt_numeric_canonicals,
            num_token_id=num_token_id,
            max_numeric_chars=max_numeric_chars,
            overflow=overflow,
            sequence_length=len(input_ids),
        )
        output["numeric_mask"] = numeric_mask
        output["numeric_char_ids"] = numeric_char_ids
    return output


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


def parse_pairwise_preference(text: str) -> Optional[str]:
    parsed: Dict[str, object] = {}
    json_blob = extract_balanced_json_object(text)
    if json_blob is not None:
        try:
            value = json.loads(json_blob)
            if isinstance(value, dict):
                parsed = value
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(json_blob)
                if isinstance(value, dict):
                    parsed = value
            except (SyntaxError, ValueError):
                parsed = {}

    raw_value = None
    for key in ("preferred_trace", "preferred", "choice", "winner"):
        if key in parsed:
            raw_value = parsed[key]
            break

    normalized = normalize_label(raw_value if raw_value is not None else text)
    normalized = re.sub(r"[^a-z0-9 ]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if normalized in {"a", "0", "first", "trace a", "candidate a", "option a"}:
        return "A"
    if normalized in {"b", "1", "second", "trace b", "candidate b", "option b"}:
        return "B"

    if re.search(r"\b(?:choose|prefer|preferred|select|winner is)\s+(?:trace\s+)?a\b", normalized):
        return "A"
    if re.search(r"\b(?:choose|prefer|preferred|select|winner is)\s+(?:trace\s+)?b\b", normalized):
        return "B"

    mentions_a = re.search(r"\btrace\s+a\b", normalized) is not None
    mentions_b = re.search(r"\btrace\s+b\b", normalized) is not None
    if mentions_a and not mentions_b:
        return "A"
    if mentions_b and not mentions_a:
        return "B"
    return None


def derive_verdict_from_ranking(
    verdict_list: Sequence[str],
    ranked_trace_indices: Sequence[int],
    top_k: int,
    label_space: Optional[Sequence[str]] = None,
) -> str:
    top_k = max(1, int(top_k))
    display_by_normalized = {
        normalize_label(label): str(label).strip()
        for label in (label_space or [])
        if str(label).strip()
    }
    picked: List[str] = []
    for index in ranked_trace_indices[:top_k]:
        try:
            trace_index = int(index)
        except (TypeError, ValueError):
            continue
        if 0 <= trace_index < len(verdict_list):
            normalized = normalize_label(verdict_list[trace_index])
            if normalized:
                picked.append(normalized)

    if not picked:
        return ""

    counts: Dict[str, int] = {}
    for label in picked:
        counts[label] = counts.get(label, 0) + 1

    # Majority vote over top-k. Ties are resolved by earliest occurrence in the ranking.
    best_label = picked[0]
    best_count = counts[best_label]
    for label in picked:
        count = counts[label]
        if count > best_count:
            best_label = label
            best_count = count

    return display_by_normalized.get(best_label, best_label)


def generate_pairwise_ranking(
    model,
    tokenizer,
    row: dict,
    max_evidence_items: int,
    max_evidence_chars: int,
    max_trace_chars: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    use_numeric_embedding: bool = False,
    num_token_id: Optional[int] = None,
    max_numeric_chars: int = 24,
    keep_pairwise_details: bool = False,
    batch_size: int = 1,
    pairwise_orientation: str = "original",
) -> Dict[str, object]:
    traces = row.get("Reasoning_traces", []) or []
    num_traces = len(traces)
    scores = [0.0] * num_traces
    pairwise_details: List[Dict[str, object]] = []
    invalid_outputs = 0
    comparison_count = 0
    comparisons: List[Dict[str, object]] = []

    for left_index in range(num_traces):
        for right_index in range(left_index + 1, num_traces):
            if pairwise_orientation == "original":
                trace_a_index = left_index
                trace_b_index = right_index
            elif pairwise_orientation == "balanced":
                if left_index % 2 == 0:
                    trace_a_index = left_index
                    trace_b_index = right_index
                else:
                    trace_a_index = right_index
                    trace_b_index = left_index
            elif pairwise_orientation == "bidirectional":
                trace_a_index = left_index
                trace_b_index = right_index
            else:
                raise ValueError(f"Unsupported pairwise_orientation: {pairwise_orientation}")

            orientations = [(trace_a_index, trace_b_index)]
            if pairwise_orientation == "bidirectional":
                orientations.append((right_index, left_index))

            for orientation_index, (oriented_a_index, oriented_b_index) in enumerate(orientations):
                prompt_artifacts = build_pairwise_prompt_artifacts(
                    row=row,
                    max_evidence_items=max_evidence_items,
                    max_evidence_chars=max_evidence_chars,
                    max_trace_chars=max_trace_chars,
                    left_index=oriented_a_index,
                    right_index=oriented_b_index,
                    use_numeric_embedding=use_numeric_embedding,
                )
                comparisons.append(
                    {
                        "left_index": left_index,
                        "right_index": right_index,
                        "trace_a_index": oriented_a_index,
                        "trace_b_index": oriented_b_index,
                        "orientation_index": orientation_index,
                        "prompt": prompt_artifacts["prompt"],
                        "prompt_numeric_canonicals": prompt_artifacts["prompt_numeric_canonicals"],
                    }
                )

    batch_size = max(1, int(batch_size))
    bidirectional_votes: Dict[Tuple[int, int], List[Optional[int]]] = {}
    bidirectional_details: Dict[Tuple[int, int], List[Dict[str, object]]] = {}
    for start in range(0, len(comparisons), batch_size):
        batch = comparisons[start : start + batch_size]
        raw_outputs = generate_predictions_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=[str(item["prompt"]) for item in batch],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            prompt_numeric_canonicals_list=(
                [item["prompt_numeric_canonicals"] for item in batch] if use_numeric_embedding else None
            ),
            num_token_id=num_token_id if use_numeric_embedding else None,
            max_numeric_chars=max_numeric_chars,
        )
        for comparison, raw_output in zip(batch, raw_outputs):
            preferred = parse_pairwise_preference(raw_output)
            comparison_count += 1
            left_index = int(comparison["left_index"])
            right_index = int(comparison["right_index"])
            trace_a_index = int(comparison["trace_a_index"])
            trace_b_index = int(comparison["trace_b_index"])
            if preferred == "A":
                winner = trace_a_index
            elif preferred == "B":
                winner = trace_b_index
            else:
                invalid_outputs += 1
                winner = None

            if pairwise_orientation == "bidirectional":
                pair_key = (left_index, right_index)
                bidirectional_votes.setdefault(pair_key, []).append(winner)
                if keep_pairwise_details:
                    bidirectional_details.setdefault(pair_key, []).append(
                        {
                            "left_index": left_index,
                            "right_index": right_index,
                            "trace_a_index": trace_a_index,
                            "trace_b_index": trace_b_index,
                            "orientation_index": int(comparison["orientation_index"]),
                            "preferred": preferred,
                            "winner_index": winner,
                            "raw_model_output": raw_output,
                        }
                    )
                continue

            if winner is None:
                scores[left_index] += 0.5
                scores[right_index] += 0.5
            else:
                scores[winner] += 1.0

            if keep_pairwise_details:
                pairwise_details.append(
                    {
                        "left_index": left_index,
                        "right_index": right_index,
                        "trace_a_index": trace_a_index,
                        "trace_b_index": trace_b_index,
                        "orientation_index": int(comparison["orientation_index"]),
                        "preferred": preferred,
                        "winner_index": winner,
                        "raw_model_output": raw_output,
                    }
                )

    if pairwise_orientation == "bidirectional":
        for (left_index, right_index), winners in bidirectional_votes.items():
            non_null_winners = [winner for winner in winners if winner is not None]
            if len(non_null_winners) == 2 and non_null_winners[0] == non_null_winners[1]:
                scores[non_null_winners[0]] += 1.0
                outcome = "agree"
                final_winner: Optional[int] = non_null_winners[0]
            else:
                scores[left_index] += 0.5
                scores[right_index] += 0.5
                outcome = "tie"
                final_winner = None

            if keep_pairwise_details:
                pairwise_details.append(
                    {
                        "left_index": left_index,
                        "right_index": right_index,
                        "bidirectional_outcome": outcome,
                        "winner_index": final_winner,
                        "orientations": bidirectional_details.get((left_index, right_index), []),
                    }
                )

    ranked = sorted(range(num_traces), key=lambda idx: (-scores[idx], idx))
    result: Dict[str, object] = {
        "ranked_trace_indices": ranked,
        "pairwise_scores": scores,
        "num_pairwise_comparisons": comparison_count,
        "invalid_pairwise_outputs": invalid_outputs,
    }
    if keep_pairwise_details:
        result["pairwise_details"] = pairwise_details
    return result


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


def _sample_next_token(logits: torch.Tensor, do_sample: bool, temperature: float, top_p: float) -> torch.Tensor:
    if not do_sample:
        return torch.argmax(logits, dim=-1)

    adjusted = logits / max(temperature, 1e-5)
    probs = torch.softmax(adjusted, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(sorted_mask, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        sampled = torch.multinomial(sorted_probs, num_samples=1)
        return sorted_indices.gather(dim=-1, index=sampled).squeeze(-1)

    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _left_pad_generation_features(
    features: Sequence[Dict[str, List[int]]],
    pad_token_id: int,
    max_numeric_chars: int,
) -> Dict[str, torch.Tensor]:
    max_length = max(len(feature["input_ids"]) for feature in features)
    input_ids: List[List[int]] = []
    attention_mask: List[List[int]] = []
    numeric_mask: List[List[int]] = []
    numeric_char_ids: List[List[List[int]]] = []
    has_numeric = "numeric_mask" in features[0] and "numeric_char_ids" in features[0]

    for feature in features:
        pad_len = max_length - len(feature["input_ids"])
        input_ids.append([pad_token_id] * pad_len + feature["input_ids"])
        attention_mask.append([0] * pad_len + feature["attention_mask"])
        if has_numeric:
            numeric_mask.append([0] * pad_len + feature["numeric_mask"])
            numeric_char_ids.append(
                ([[0] * max_numeric_chars] * pad_len) + feature["numeric_char_ids"]
            )

    output = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }
    if has_numeric:
        output["numeric_mask"] = torch.tensor(numeric_mask, dtype=torch.bool)
        output["numeric_char_ids"] = torch.tensor(numeric_char_ids, dtype=torch.long)
    return output


def generate_predictions_batch(
    model,
    tokenizer,
    prompts: Sequence[str],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    prompt_numeric_canonicals_list: Optional[Sequence[Sequence[str]]] = None,
    num_token_id: Optional[int] = None,
    max_numeric_chars: int = 24,
) -> List[str]:
    if not prompts:
        return []

    numeric_lists: Sequence[Optional[Sequence[str]]]
    if prompt_numeric_canonicals_list is None:
        numeric_lists = [None] * len(prompts)
    else:
        if len(prompt_numeric_canonicals_list) != len(prompts):
            raise ValueError("prompt_numeric_canonicals_list must match prompts length.")
        numeric_lists = list(prompt_numeric_canonicals_list)

    features = [
        tokenize_prompt_for_generation(
            tokenizer=tokenizer,
            prompt=prompt,
            prompt_numeric_canonicals=numeric_lists[idx],
            num_token_id=num_token_id,
            max_numeric_chars=max_numeric_chars,
        )
        for idx, prompt in enumerate(prompts)
    ]
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    padded = _left_pad_generation_features(
        features=features,
        pad_token_id=int(pad_token_id),
        max_numeric_chars=max_numeric_chars,
    )
    device = get_model_input_device(model)
    input_ids = padded["input_ids"].to(device)
    attention_mask = padded["attention_mask"].to(device)
    numeric_mask = padded.get("numeric_mask")
    numeric_char_ids = padded.get("numeric_char_ids")
    if numeric_mask is not None:
        numeric_mask = numeric_mask.to(device)
        numeric_char_ids = numeric_char_ids.to(device)

    eos_token_id = tokenizer.eos_token_id
    finished = torch.zeros((len(prompts),), dtype=torch.bool, device=device)
    generated_tokens: List[List[int]] = [[] for _ in prompts]

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            numeric_mask=numeric_mask,
            numeric_char_ids=numeric_char_ids,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token = _sample_next_token(
            outputs.logits[:, -1, :],
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )

        for _ in range(max_new_tokens):
            if eos_token_id is not None:
                just_finished = next_token.eq(eos_token_id)
            else:
                just_finished = torch.zeros_like(finished)

            for row_idx, token_id in enumerate(next_token.tolist()):
                if not bool(finished[row_idx]) and not bool(just_finished[row_idx]):
                    generated_tokens[row_idx].append(int(token_id))

            finished = finished | just_finished
            if bool(finished.all()):
                break

            step_input_ids = next_token.clone()
            if eos_token_id is not None:
                step_input_ids = step_input_ids.masked_fill(finished, int(eos_token_id))
            else:
                step_input_ids = step_input_ids.masked_fill(finished, int(pad_token_id))
            step_input_ids = step_input_ids.view(len(prompts), 1).to(device)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=device),
                ],
                dim=1,
            )
            outputs = model(
                input_ids=step_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token = _sample_next_token(
                outputs.logits[:, -1, :],
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )

    return [
        tokenizer.decode(tokens, skip_special_tokens=True).strip()
        for tokens in generated_tokens
    ]


def generate_prediction(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    prompt_numeric_canonicals: Optional[Sequence[str]] = None,
    num_token_id: Optional[int] = None,
    max_numeric_chars: int = 24,
) -> str:
    tokenized = tokenize_prompt_for_generation(
        tokenizer=tokenizer,
        prompt=prompt,
        prompt_numeric_canonicals=prompt_numeric_canonicals,
        num_token_id=num_token_id,
        max_numeric_chars=max_numeric_chars,
    )
    device = get_model_input_device(model)
    input_ids = torch.tensor([tokenized["input_ids"]], dtype=torch.long, device=device)
    attention_mask = torch.tensor([tokenized["attention_mask"]], dtype=torch.long, device=device)
    numeric_mask = None
    numeric_char_ids = None
    if "numeric_mask" in tokenized:
        numeric_mask = torch.tensor([tokenized["numeric_mask"]], dtype=torch.bool, device=device)
        numeric_char_ids = torch.tensor([tokenized["numeric_char_ids"]], dtype=torch.long, device=device)

    eos_token_id = tokenizer.eos_token_id
    generated_tokens: List[int] = []

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            numeric_mask=numeric_mask,
            numeric_char_ids=numeric_char_ids,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token = _sample_next_token(outputs.logits[:, -1, :], do_sample=do_sample, temperature=temperature, top_p=top_p)

        for _ in range(max_new_tokens):
            token_id = int(next_token.item())
            if eos_token_id is not None and token_id == eos_token_id:
                break
            generated_tokens.append(token_id)

            step_input_ids = next_token.view(1, 1).to(device)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=device)],
                dim=1,
            )
            outputs = model(
                input_ids=step_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token = _sample_next_token(outputs.logits[:, -1, :], do_sample=do_sample, temperature=temperature, top_p=top_p)

    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
