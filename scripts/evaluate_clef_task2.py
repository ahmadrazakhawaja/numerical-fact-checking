#!/usr/bin/env python3
"""Evaluate CLEF Task 2 predictions (ranking + verdict classification).

Accepted prediction JSON formats (list of objects):
[
  {
    "language": "english",
    "dataset_index": 0,
    "ranked_trace_indices": [3, 1, 2, ...],
    "predicted_verdict": "False"
  },
  {
    "query_id": 0,
    "Verdict_BoN": "False",
    "score_list": [4.2, -1.0, ...]
  },
  ...
]

`ranked_trace_indices` must be a permutation (or subset) of 0-based indices into
`Reasoning_traces` for the corresponding claim. For submission-style rows,
`score_list` is sorted descending to recover the ranking. Metrics:
- Recall@k on relevant traces (relevant iff Verdict_list[i] == gold claim verdict)
- Precision@k on relevant traces
- Recall@k upper bound (`min(k, |relevant|) / |relevant|`)
- MRR@k on relevant traces
- macro F1 and classwise F1 for claim-level verdict prediction
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:
    from task2_utils import get_gold_label, infer_language_name, load_dataset_rows, load_json_rows, normalize_label, safe_div
except ImportError:  # pragma: no cover - import path fallback
    from scripts.task2_utils import (
        get_gold_label,
        infer_language_name,
        load_dataset_rows,
        load_json_rows,
        normalize_label,
        safe_div,
    )


DEFAULT_LANGS = ("english", "spanish", "arabic")


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


def build_claim_record(language: str, dataset_index: int, row: dict) -> ClaimRecord:
    verdict_list = [normalize_label(v) for v in row.get("Verdict_list", [])]
    gold_verdict = get_gold_label(row)
    num_traces = len(row.get("Reasoning_traces", []))
    return ClaimRecord(
        language=language,
        dataset_index=dataset_index,
        gold_verdict=gold_verdict,
        verdict_list=verdict_list,
        num_traces=num_traces,
    )


def subset_rows(rows: List[dict], start_index: int, limit: Optional[int]) -> List[dict]:
    if start_index < 0:
        raise ValueError(f"start_index must be >= 0, got {start_index}")
    sliced = rows[start_index:]
    if limit is not None:
        if limit < 0:
            raise ValueError(f"limit must be >= 0, got {limit}")
        sliced = sliced[:limit]
    return sliced


def load_claims(
    dataset_dir: Path,
    split: str,
    languages: Iterable[str],
    start_index: int = 0,
    limit: Optional[int] = None,
) -> Dict[Tuple[str, int], ClaimRecord]:
    claims: Dict[Tuple[str, int], ClaimRecord] = {}
    for language in languages:
        rows = subset_rows(load_dataset_rows(dataset_dir, split, language), start_index=start_index, limit=limit)
        for idx, row in enumerate(rows, start=start_index):
            claims[(language, idx)] = build_claim_record(language=language, dataset_index=idx, row=row)
    return claims


def load_claims_from_file(
    dataset_path: Path,
    language_name: str,
    start_index: int = 0,
    limit: Optional[int] = None,
) -> Dict[Tuple[str, int], ClaimRecord]:
    claims: Dict[Tuple[str, int], ClaimRecord] = {}
    rows = subset_rows(load_json_rows(dataset_path), start_index=start_index, limit=limit)
    for idx, row in enumerate(rows, start=start_index):
        claims[(language_name, idx)] = build_claim_record(language=language_name, dataset_index=idx, row=row)
    return claims


def evaluate_predictions_against_dataset(
    predictions_path: Path,
    k: int,
    dataset_dir: Optional[Path] = None,
    split: Optional[str] = None,
    languages: Optional[Iterable[str]] = None,
    dataset_path: Optional[Path] = None,
    language_name: Optional[str] = None,
    start_index: int = 0,
    limit: Optional[int] = None,
) -> Dict[str, object]:
    if dataset_path is not None:
        resolved_language = language_name or infer_language_name(dataset_path)
        claims = load_claims_from_file(
            dataset_path,
            resolved_language,
            start_index=start_index,
            limit=limit,
        )
    else:
        if dataset_dir is None or split is None:
            raise ValueError("dataset_dir and split are required when dataset_path is not provided.")
        claims = load_claims(
            dataset_dir,
            split,
            languages or DEFAULT_LANGS,
            start_index=start_index,
            limit=limit,
        )

    default_language = resolved_language if dataset_path is not None else None
    preds = load_predictions(predictions_path, default_language=default_language)
    return evaluate(claims, preds, k=k)


def ranked_indices_from_scores(score_list: List[float]) -> List[int]:
    return sorted(range(len(score_list)), key=lambda idx: (-float(score_list[idx]), idx))


def load_predictions(path: Path, default_language: Optional[str] = None) -> Dict[Tuple[str, int], PredictionRecord]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    preds: Dict[Tuple[str, int], PredictionRecord] = {}
    for row in rows:
        raw_language = row.get("language", default_language)
        if raw_language is None:
            raise ValueError(
                "Prediction rows without a language require --dataset-path/--language-name "
                "or an explicit language field."
            )
        language = str(raw_language).strip().lower()

        if "dataset_index" in row:
            dataset_index = int(row["dataset_index"])
        elif "query_id" in row:
            dataset_index = int(row["query_id"])
        else:
            raise ValueError("Prediction row is missing dataset_index/query_id.")

        if "ranked_trace_indices" in row:
            ranked = [int(i) for i in row.get("ranked_trace_indices", [])]
        elif "score_list" in row:
            ranked = ranked_indices_from_scores([float(x) for x in row.get("score_list", [])])
        else:
            raise ValueError("Prediction row is missing ranked_trace_indices/score_list.")

        pred_verdict = normalize_label(row.get("predicted_verdict", row.get("Verdict_BoN", "")))
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


def precision_at_k(ranked: List[int], relevant: Set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    topk = ranked[:k]
    hit = sum(1 for i in topk if i in relevant)
    return hit / k


def recall_at_k_upper_bound(relevant: Set[int], k: int) -> float:
    if not relevant:
        return 0.0
    return min(k, len(relevant)) / len(relevant)


def mrr_at_k(ranked: List[int], relevant: Set[int], k: int) -> float:
    for rank, idx in enumerate(ranked[:k], start=1):
        if idx in relevant:
            return 1.0 / rank
    return 0.0


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
    ranking_precision = []
    ranking_recall_upper_bound = []
    ranking_mrr = []
    y_true: List[str] = []
    y_pred: List[str] = []
    gold_label_counts: Counter[str] = Counter()
    verdict_label_counts: Counter[str] = Counter()

    missing_predictions = 0
    invalid_rank_indices = 0
    empty_gold_labels = 0
    claims_without_relevant_traces = 0

    for key, claim in claims.items():
        gold_label_counts[claim.gold_verdict] += 1
        verdict_label_counts.update(v for v in claim.verdict_list if v)
        if not claim.gold_verdict:
            empty_gold_labels += 1

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
        if not relevant:
            claims_without_relevant_traces += 1
        ranking_recall.append(recall_at_k(filtered, relevant, k=k))
        ranking_precision.append(precision_at_k(filtered, relevant, k=k))
        ranking_recall_upper_bound.append(recall_at_k_upper_bound(relevant, k=k))
        ranking_mrr.append(mrr_at_k(filtered, relevant, k=k))

        y_true.append(claim.gold_verdict)
        y_pred.append(pred_verdict)

    if claims and empty_gold_labels == len(claims):
        raise ValueError(
            "All gold labels are empty. The evaluator expects the claim-level gold label in a supported "
            "gold-label field such as 'label' or 'Label'. Check the dataset file passed via --dataset-path, or do not pass --evaluate for "
            "unlabeled test files."
        )
    if claims and claims_without_relevant_traces == len(claims):
        raise ValueError(
            "No relevant traces were found for any claim. This usually means the claim-level 'label' values "
            "are missing or do not use the same label strings as Verdict_list. "
            f"Gold labels: {dict(gold_label_counts)}; trace verdict labels: {dict(verdict_label_counts)}"
        )

    labels = sorted(set(y_true))
    macro_f1, classwise_f1 = f1_scores(y_true, y_pred, labels)

    return {
        "num_claims": len(claims),
        "k": k,
        "recall_at_k": safe_div(sum(ranking_recall), len(ranking_recall)),
        "precision_at_k": safe_div(sum(ranking_precision), len(ranking_precision)),
        "recall_at_k_upper_bound": safe_div(sum(ranking_recall_upper_bound), len(ranking_recall_upper_bound)),
        "mrr_at_k": safe_div(sum(ranking_mrr), len(ranking_mrr)),
        "macro_f1": macro_f1,
        "macro_f1_recall_at_k_mean": 0.5
        * (macro_f1 + safe_div(sum(ranking_recall), len(ranking_recall))),
        "classwise_f1": classwise_f1,
        "missing_predictions": missing_predictions,
        "invalid_rank_indices": invalid_rank_indices,
        "empty_gold_labels": empty_gold_labels,
        "claims_without_relevant_traces": claims_without_relevant_traces,
        "gold_label_counts": dict(gold_label_counts),
        "trace_verdict_label_counts": dict(verdict_label_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CLEF Task 2 predictions.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"), help="Path to dataset root.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Optional path to a single dataset JSON file. Useful for translated variants.",
    )
    parser.add_argument(
        "--language-name",
        type=str,
        default=None,
        help="Language identifier to use with --dataset-path. Defaults to an inferred name.",
    )
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation"])
    parser.add_argument("--predictions", type=Path, required=True, help="Path to predictions JSON.")
    parser.add_argument("--k", type=int, default=5, help="k for Recall@k and MRR@k.")
    parser.add_argument("--start-index", type=int, default=0, help="Optional dataset row offset for subset evaluation.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of dataset rows to evaluate.")
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
    metrics = evaluate_predictions_against_dataset(
        predictions_path=args.predictions,
        k=args.k,
        dataset_dir=args.dataset_dir,
        split=args.split,
        languages=languages,
        dataset_path=args.dataset_path,
        language_name=args.language_name,
        start_index=args.start_index,
        limit=args.limit,
    )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    if args.output:
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
