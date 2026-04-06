#!/usr/bin/env python3
"""Prompted ranking baseline using a frozen instruction-tuned causal LM.

This script loads a single dataset JSON file, prompts the model to:
1) rank the candidate reasoning traces from best to worst, and
2) predict the final claim verdict.

Predictions are written in the same schema expected by `evaluate_clef_task2.py`.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_clef_task2 import evaluate_predictions_against_dataset, infer_language_name


DEFAULT_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"


def normalize_label(label: str) -> str:
    return str(label).strip().lower()


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


def render_list_block(items: List[str], max_items: int, max_chars: int, header: str) -> str:
    if not items:
        return f"{header}\n- None"

    capped_items = items[:max_items] if max_items > 0 else items
    lines = [header]
    for idx, item in enumerate(capped_items):
        lines.append(f"[{idx}] {truncate_text(str(item), max_chars)}")
    if max_items > 0 and len(items) > max_items:
        lines.append(f"... {len(items) - max_items} more omitted")
    return "\n".join(lines)


def build_prompt(
    row: dict,
    label_space: List[str],
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


def parse_predicted_verdict(raw_value, label_space: List[str]) -> str:
    canonical = {normalize_label(label): label for label in label_space}
    if raw_value is not None:
        label = canonical.get(normalize_label(raw_value))
        if label is not None:
            return label
    return ""


def parse_response(text: str, label_space: List[str], num_traces: int) -> Dict[str, object]:
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
            except (SyntaxError, ValueError) as exc:
                parse_error = str(exc)

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


def infer_label_space(rows: List[dict]) -> List[str]:
    labels = sorted({str(row.get("label", "")).strip() for row in rows if str(row.get("label", "")).strip()})
    return labels


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
    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful multilingual fact-checking assistant. "
                "Return strict JSON only."
            ),
        },
        {"role": "user", "content": prompt},
    ]
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


def infer_output_language(dataset_path: Path, override: Optional[str]) -> str:
    if override:
        return override.strip().lower()
    return infer_language_name(dataset_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompted ranking baseline for CLEF Task 2.")
    parser.add_argument("--dataset-path", type=Path, required=True, help="Path to a dataset JSON file.")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--output", type=Path, required=True, help="Path to save predictions JSON.")
    parser.add_argument(
        "--language-name",
        type=str,
        default=None,
        help="Language identifier stored in prediction records. Defaults to an inferred name.",
    )
    parser.add_argument(
        "--variant-name",
        type=str,
        default=None,
        help="Optional variant name for metadata. Defaults to the dataset file stem.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Optional number of examples to run.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-evidence-items", type=int, default=3)
    parser.add_argument("--max-evidence-chars", type=int, default=1200)
    parser.add_argument("--max-trace-chars", type=int, default=1200)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--save-debug-fields", action="store_true")
    parser.add_argument("--evaluate", action="store_true", help="Run CLEF metric evaluation after prediction.")
    parser.add_argument("--eval-k", type=int, default=5)
    parser.add_argument("--eval-output", type=Path, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    rows = json.load(args.dataset_path.open("r", encoding="utf-8"))
    selected_rows = rows[args.start_index :]
    if args.limit is not None:
        selected_rows = selected_rows[: args.limit]

    label_space = infer_label_space(rows)
    language_name = infer_output_language(args.dataset_path, args.language_name)
    variant_name = args.variant_name or args.dataset_path.stem

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=resolve_dtype(args.dtype),
        device_map=args.device_map,
    )
    model.eval()

    do_sample = args.do_sample or args.temperature > 0.0
    predictions: List[dict] = []

    for local_offset, row in enumerate(tqdm(selected_rows, desc="Running prompted baseline")):
        dataset_index = args.start_index + local_offset
        num_traces = len(row.get("Reasoning_traces", []) or [])

        prompt = build_prompt(
            row=row,
            label_space=label_space,
            max_evidence_items=args.max_evidence_items,
            max_evidence_chars=args.max_evidence_chars,
            max_trace_chars=args.max_trace_chars,
        )
        raw_output = generate_prediction(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        parsed = parse_response(raw_output, label_space=label_space, num_traces=num_traces)

        record = {
            "language": language_name,
            "dataset_index": dataset_index,
            "ranked_trace_indices": parsed["ranked_trace_indices"],
            "predicted_verdict": parsed["predicted_verdict"],
            "variant": variant_name,
            "group_id": dataset_index,
        }
        if args.save_debug_fields:
            record.update(
                {
                    "claim": row.get("claim", ""),
                    "raw_model_output": raw_output,
                    "json_found": parsed["json_found"],
                    "json_parse_error": parsed["json_parse_error"],
                }
            )
        predictions.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(
        json.dumps(
            {
                "saved_predictions": str(args.output),
                "model_id": args.model_id,
                "dataset_path": str(args.dataset_path),
                "language_name": language_name,
                "variant_name": variant_name,
                "num_predictions": len(predictions),
                "start_index": args.start_index,
                "limit": args.limit,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    if args.evaluate:
        metrics = evaluate_predictions_against_dataset(
            predictions_path=args.output,
            k=args.eval_k,
            dataset_path=args.dataset_path,
            language_name=language_name,
        )
        eval_output = args.eval_output
        if eval_output is not None:
            eval_output.parent.mkdir(parents=True, exist_ok=True)
            with eval_output.open("w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
