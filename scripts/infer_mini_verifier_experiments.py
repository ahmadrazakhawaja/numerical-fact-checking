#!/usr/bin/env python3
"""Run inference for mini verifier checkpoints."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))

from mini_factcheck_common import (
    TraceClassifier,
    TraceExample,
    build_backbone_model,
    build_tokenizer,
    build_trace_examples,
    login_hf_from_env,
    load_json_rows,
    load_metadata,
    normalize_verdict,
    select_rows,
)


DEFAULT_MODES = ("ntp", "cls_frozen", "cls_unfrozen")


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_modes(modes: Sequence[str]) -> List[str]:
    clean = [m.strip().lower() for m in modes]
    bad = [m for m in clean if m not in DEFAULT_MODES]
    if bad:
        raise ValueError(f"Unsupported modes: {bad}. Allowed: {DEFAULT_MODES}")
    return clean


def _group_examples_by_claim(examples: Sequence[TraceExample]) -> Dict[int, List[TraceExample]]:
    grouped: Dict[int, List[TraceExample]] = defaultdict(list)
    for ex in examples:
        grouped[ex.claim_idx].append(ex)
    return grouped


def _majority_top_k(labels: Sequence[str], k: int) -> str:
    top = list(labels[: max(1, k)])
    counts = Counter(top)
    best_label = top[0]
    best_count = counts[best_label]
    for label in top:
        if counts[label] > best_count:
            best_label = label
            best_count = counts[label]
    return best_label


def _load_label_to_id(metadata: Dict[str, object]) -> Dict[str, int]:
    raw = metadata["label_to_id"]
    return {str(k): int(v) for k, v in dict(raw).items()}


def infer_ntp_mode(
    *,
    mode_dir: Path,
    metadata: Dict[str, object],
    examples: Sequence[TraceExample],
    max_length: int,
    device: torch.device,
) -> List[dict]:
    label_to_id = _load_label_to_id(metadata)
    labels = list(label_to_id.keys())

    tokenizer = build_tokenizer(
        base_model_id=str(metadata["model_id"]),
        adapter_id=str(mode_dir) if (mode_dir / "adapter_config.json").exists() else None,
    )

    if (mode_dir / "adapter_config.json").exists():
        model = build_backbone_model(
            base_model_id=str(metadata["model_id"]),
            adapter_id=str(mode_dir),
            dtype=torch.float32,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(mode_dir, torch_dtype=torch.float32)
    model.to(device)
    model.eval()

    label_first_token_ids: Dict[str, int] = {}
    for label in labels:
        tok = tokenizer(label, add_special_tokens=False)["input_ids"]
        if not tok:
            continue
        label_first_token_ids[label] = int(tok[0])
    if not label_first_token_ids:
        raise ValueError("Could not build label token ids for NTP inference.")

    grouped = _group_examples_by_claim(examples)
    results: List[dict] = []
    with torch.no_grad():
        for claim_idx, claim_examples in sorted(grouped.items(), key=lambda x: x[0]):
            trace_preds = []
            for ex in claim_examples:
                prompt = ex.packed_input + " "
                enc = tokenizer(
                    prompt,
                    truncation=True,
                    max_length=max_length,
                    add_special_tokens=True,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                out = model(**enc)
                next_logits = out.logits[0, -1]

                cand_labels = list(label_first_token_ids.keys())
                cand_token_ids = [label_first_token_ids[l] for l in cand_labels]
                cand_scores = next_logits[cand_token_ids]
                probs = torch.softmax(cand_scores, dim=0)
                best_idx = int(torch.argmax(probs).item())
                pred_label = cand_labels[best_idx]

                trace_preds.append(
                    {
                        "trace_index": ex.trace_idx,
                        "predicted_verdict": pred_label,
                        "gold_trace_verdict": normalize_verdict(ex.label_text),
                        "score": float(probs[best_idx].item()),
                    }
                )

            trace_preds.sort(key=lambda x: x["score"], reverse=True)
            ranked_labels = [x["predicted_verdict"] for x in trace_preds]
            results.append(
                {
                    "claim_index": claim_idx,
                    "predicted_verdict_top1": ranked_labels[0],
                    "predicted_verdict_majority_top5": _majority_top_k(ranked_labels, k=5),
                    "ranked_trace_indices": [x["trace_index"] for x in trace_preds],
                    "trace_predictions": trace_preds,
                }
            )

    return results


def infer_classifier_mode(
    *,
    mode_dir: Path,
    metadata: Dict[str, object],
    examples: Sequence[TraceExample],
    max_length: int,
    device: torch.device,
) -> List[dict]:
    label_to_id = _load_label_to_id(metadata)
    id_to_label = {v: k for k, v in label_to_id.items()}

    tokenizer = build_tokenizer(
        base_model_id=str(metadata["model_id"]),
        adapter_id=str(metadata.get("adapter_id")) if metadata.get("adapter_id") else None,
    )
    backbone = build_backbone_model(
        base_model_id=str(metadata["model_id"]),
        adapter_id=str(metadata.get("adapter_id")) if metadata.get("adapter_id") else None,
        dtype=torch.float32,
    )
    model = TraceClassifier(backbone=backbone, num_labels=len(label_to_id))
    state = torch.load(mode_dir / "classifier.pt", map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    grouped = _group_examples_by_claim(examples)
    results: List[dict] = []
    with torch.no_grad():
        for claim_idx, claim_examples in sorted(grouped.items(), key=lambda x: x[0]):
            trace_preds = []
            for ex in claim_examples:
                enc = tokenizer(
                    ex.packed_input,
                    truncation=True,
                    max_length=max_length,
                    add_special_tokens=True,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
                probs = torch.softmax(out["logits"], dim=-1)[0]
                pred_id = int(torch.argmax(probs).item())
                pred_label = id_to_label[pred_id]

                trace_preds.append(
                    {
                        "trace_index": ex.trace_idx,
                        "predicted_verdict": pred_label,
                        "gold_trace_verdict": normalize_verdict(ex.label_text),
                        "score": float(probs[pred_id].item()),
                    }
                )

            trace_preds.sort(key=lambda x: x["score"], reverse=True)
            ranked_labels = [x["predicted_verdict"] for x in trace_preds]
            results.append(
                {
                    "claim_index": claim_idx,
                    "predicted_verdict_top1": ranked_labels[0],
                    "predicted_verdict_majority_top5": _majority_top_k(ranked_labels, k=5),
                    "ranked_trace_indices": [x["trace_index"] for x in trace_preds],
                    "trace_predictions": trace_preds,
                }
            )

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for mini verifier experiments.")
    parser.add_argument("--dataset-path", type=Path, default=Path("dataset/english/train.json"))
    parser.add_argument("--num-claims", type=int, default=5)
    parser.add_argument("--max-traces-per-claim", type=int, default=20)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints/mini_factcheck"))
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=Path("results/mini_factcheck_inference.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = parse_modes(args.modes)
    device = choose_device(args.device)
    if login_hf_from_env():
        print("Authenticated to Hugging Face Hub using HUGGINGFACE_HUB_TOKEN.")
    else:
        print("No HUGGINGFACE_HUB_TOKEN found in .env/env; proceeding unauthenticated.")

    rows = load_json_rows(args.dataset_path)
    selected_rows = select_rows(rows=rows, num_claims=args.num_claims)
    examples, _ = build_trace_examples(
        rows=selected_rows,
        max_traces_per_claim=args.max_traces_per_claim,
    )

    all_outputs: Dict[str, object] = {
        "dataset_path": str(args.dataset_path),
        "num_claims": args.num_claims,
        "max_traces_per_claim": args.max_traces_per_claim,
        "checkpoint_root": str(args.checkpoint_root),
        "device": str(device),
        "modes": {},
    }

    for mode in modes:
        mode_dir = args.checkpoint_root / mode
        metadata = load_metadata(mode_dir / "metadata.json")
        if mode == "ntp":
            results = infer_ntp_mode(
                mode_dir=mode_dir,
                metadata=metadata,
                examples=examples,
                max_length=args.max_length,
                device=device,
            )
        else:
            results = infer_classifier_mode(
                mode_dir=mode_dir,
                metadata=metadata,
                examples=examples,
                max_length=args.max_length,
                device=device,
            )
        all_outputs["modes"][mode] = results

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(all_outputs, f, indent=2, ensure_ascii=False)

    print(json.dumps(all_outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
