#!/usr/bin/env python3
"""Evaluate CLEF Task 2 predictions (ranking + verdict classification).

Expected prediction JSON format (list of objects):
[
  {
    "language": "english",
    "dataset_index": 0,
    "ranked_trace_indices": [3, 1, 2, ...],
    "predicted_verdict": "False"
  },
  ...
]

`ranked_trace_indices` must be a permutation (or subset) of 0-based indices into
`Reasoning_traces` for the corresponding claim. Metrics:
- Recall@k on relevant traces (relevant iff Verdict_list[i] == gold claim verdict)
- MRR@k on relevant traces
- macro F1 and classwise F1 for claim-level verdict prediction
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


DEFAULT_LANGS = ("english", "spanish", "arabic")


def normalize_label(label: str) -> str:
    return str(label).strip().lower()


@dataclass
class ClaimRecord:
    language: str
    dataset_index: int
    gold_verdict: str
    verdict_list: List[str]
    num_traces: int


@dataclass
class PredictionRecord:
    language: str
    dataset_index: int
    ranked_trace_indices: List[int]
    predicted_verdict: str


def load_claims(dataset_dir: Path, split: str, languages: Iterable[str]) -> Dict[Tuple[str, int], ClaimRecord]:
    claims: Dict[Tuple[str, int], ClaimRecord] = {}
    for language in languages:
        path = dataset_dir / language / f"{split}.json"
        with path.open("r", encoding="utf-8") as f:
            rows = json.load(f)
        for idx, row in enumerate(rows):
            verdict_list = [normalize_label(v) for v in row.get("Verdict_list", [])]
            gold_verdict = normalize_label(row.get("label", ""))
            num_traces = len(row.get("Reasoning_traces", []))
            claims[(language, idx)] = ClaimRecord(
                language=language,
                dataset_index=idx,
                gold_verdict=gold_verdict,
                verdict_list=verdict_list,
                num_traces=num_traces,
            )
    return claims


def load_predictions(path: Path) -> Dict[Tuple[str, int], PredictionRecord]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    preds: Dict[Tuple[str, int], PredictionRecord] = {}
    for row in rows:
        language = str(row["language"]).strip().lower()
        dataset_index = int(row["dataset_index"])
        ranked = [int(i) for i in row.get("ranked_trace_indices", [])]
        pred_verdict = normalize_label(row.get("predicted_verdict", ""))
        key = (language, dataset_index)
        preds[key] = PredictionRecord(
            language=language,
            dataset_index=dataset_index,
            ranked_trace_indices=ranked,
            predicted_verdict=pred_verdict,
        )
    return preds


def recall_at_k(ranked: List[int], relevant: Set[int], k: int) -> float:
    if not relevant:
        return 0.0
    topk = ranked[:k]
    hit = sum(1 for i in topk if i in relevant)
    return hit / len(relevant)


def mrr_at_k(ranked: List[int], relevant: Set[int], k: int) -> float:
    for rank, idx in enumerate(ranked[:k], start=1):
        if idx in relevant:
            return 1.0 / rank
    return 0.0


def safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


def f1_scores(y_true: List[str], y_pred: List[str], labels: List[str]) -> Tuple[float, Dict[str, float]]:
    classwise: Dict[str, float] = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        classwise[label] = f1

    macro = safe_div(sum(classwise.values()), len(labels))
    return macro, classwise


def evaluate(
    claims: Dict[Tuple[str, int], ClaimRecord],
    preds: Dict[Tuple[str, int], PredictionRecord],
    k: int,
) -> Dict[str, object]:
    ranking_recall = []
    ranking_mrr = []
    y_true: List[str] = []
    y_pred: List[str] = []

    missing_predictions = 0
    invalid_rank_indices = 0

    for key, claim in claims.items():
        pred = preds.get(key)
        if pred is None:
            missing_predictions += 1
            ranked = list(range(claim.num_traces))
            pred_verdict = ""
        else:
            ranked = pred.ranked_trace_indices
            pred_verdict = pred.predicted_verdict

        # Keep only valid trace indices and preserve order.
        filtered = []
        seen = set()
        for idx in ranked:
            if 0 <= idx < claim.num_traces and idx not in seen:
                filtered.append(idx)
                seen.add(idx)
            else:
                invalid_rank_indices += 1

        # Append any missing indices so each ranking can be treated as full.
        for idx in range(claim.num_traces):
            if idx not in seen:
                filtered.append(idx)

        relevant = {i for i, v in enumerate(claim.verdict_list) if v == claim.gold_verdict}
        ranking_recall.append(recall_at_k(filtered, relevant, k=k))
        ranking_mrr.append(mrr_at_k(filtered, relevant, k=k))

        y_true.append(claim.gold_verdict)
        y_pred.append(pred_verdict)

    labels = sorted(set(y_true))
    macro_f1, classwise_f1 = f1_scores(y_true, y_pred, labels)

    return {
        "num_claims": len(claims),
        "k": k,
        "recall_at_k": safe_div(sum(ranking_recall), len(ranking_recall)),
        "mrr_at_k": safe_div(sum(ranking_mrr), len(ranking_mrr)),
        "macro_f1": macro_f1,
        "macro_f1_recall_at_k_mean": 0.5
        * (macro_f1 + safe_div(sum(ranking_recall), len(ranking_recall))),
        "classwise_f1": classwise_f1,
        "missing_predictions": missing_predictions,
        "invalid_rank_indices": invalid_rank_indices,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CLEF Task 2 predictions.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"), help="Path to dataset root.")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation"])
    parser.add_argument("--predictions", type=Path, required=True, help="Path to predictions JSON.")
    parser.add_argument("--k", type=int, default=5, help="k for Recall@k and MRR@k.")
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=list(DEFAULT_LANGS),
        help="Languages to evaluate.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    languages = [x.strip().lower() for x in args.languages]
    claims = load_claims(args.dataset_dir, args.split, languages)
    preds = load_predictions(args.predictions)
    metrics = evaluate(claims, preds, k=args.k)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    if args.output:
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
