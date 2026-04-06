#!/usr/bin/env python3
"""Evaluate cross-variant language invariance from prediction files.

Typical current use in this repo is comparing original-language predictions
against translated predictions for the same dataset indices.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from task2_utils import load_json_rows, normalize_label, safe_div
except ImportError:  # pragma: no cover - import path fallback
    from scripts.task2_utils import load_json_rows, normalize_label, safe_div


def resolve_variant_names(paths: Sequence[Path], override_names: Optional[Sequence[str]]) -> List[str]:
    if override_names is None:
        return [path.stem for path in paths]
    if len(override_names) != len(paths):
        raise ValueError("--names must have the same length as --predictions.")
    return [str(name).strip() for name in override_names]


def resolve_group_id(row: dict, group_key: str, fallback_group_key: str) -> Optional[str]:
    if group_key in row:
        return str(row[group_key])
    if fallback_group_key in row:
        return str(row[fallback_group_key])
    return None


def sanitize_ranking(raw_value) -> List[int]:
    if not isinstance(raw_value, list):
        return []

    ranked: List[int] = []
    seen = set()
    for item in raw_value:
        try:
            idx = int(item)
        except (TypeError, ValueError):
            continue
        if idx not in seen:
            ranked.append(idx)
            seen.add(idx)
    return ranked


def raw_agreement(labels_a: Sequence[str], labels_b: Sequence[str]) -> float:
    return safe_div(sum(a == b for a, b in zip(labels_a, labels_b)), len(labels_a))


def fleiss_kappa(items: Sequence[Sequence[str]]) -> Optional[float]:
    if not items:
        return None
    num_raters = len(items[0])
    if num_raters < 2 or any(len(item) != num_raters for item in items):
        return None

    categories = sorted({label for item in items for label in item})
    if not categories:
        return None

    per_item_agreement: List[float] = []
    category_totals = {category: 0 for category in categories}

    for item in items:
        counts = {category: 0 for category in categories}
        for label in item:
            counts[label] += 1
            category_totals[label] += 1
        agreement = safe_div(
            sum(count * count for count in counts.values()) - num_raters,
            num_raters * (num_raters - 1),
        )
        per_item_agreement.append(agreement)

    observed = sum(per_item_agreement) / len(per_item_agreement)
    expected = sum((count / (len(items) * num_raters)) ** 2 for count in category_totals.values())
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def krippendorff_alpha_nominal(items: Sequence[Sequence[str]]) -> Optional[float]:
    if not items:
        return None

    total_pair_count = 0
    total_disagreements = 0
    category_counts: Dict[str, int] = {}
    total_labels = 0

    for item in items:
        for label in item:
            category_counts[label] = category_counts.get(label, 0) + 1
            total_labels += 1

        item = list(item)
        if len(item) < 2:
            continue

        for idx in range(len(item)):
            for jdx in range(idx + 1, len(item)):
                total_pair_count += 1
                if item[idx] != item[jdx]:
                    total_disagreements += 1

    if total_pair_count == 0 or total_labels < 2:
        return None

    observed_disagreement = total_disagreements / total_pair_count
    expected_agreement = sum((count / total_labels) ** 2 for count in category_counts.values())
    expected_disagreement = 1.0 - expected_agreement

    if expected_disagreement <= 0.0:
        return 1.0 if observed_disagreement <= 0.0 else 0.0
    return 1.0 - (observed_disagreement / expected_disagreement)


def cohen_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> Optional[float]:
    if not labels_a or len(labels_a) != len(labels_b):
        return None

    categories = sorted(set(labels_a) | set(labels_b))
    if not categories:
        return None

    observed = raw_agreement(labels_a, labels_b)
    dist_a = {category: 0 for category in categories}
    dist_b = {category: 0 for category in categories}
    for label in labels_a:
        dist_a[label] += 1
    for label in labels_b:
        dist_b[label] += 1

    expected = sum(
        (dist_a[category] / len(labels_a)) * (dist_b[category] / len(labels_b))
        for category in categories
    )
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def ranking_pair_metrics(ranked_a: Sequence[int], ranked_b: Sequence[int], top_k: int) -> Dict[str, Optional[float]]:
    unique_a = sanitize_ranking(list(ranked_a))
    unique_b = sanitize_ranking(list(ranked_b))
    common_items = sorted(set(unique_a) & set(unique_b))

    if len(common_items) < 2:
        return {
            "kendall_tau": None,
            "spearman_rho": None,
            "top_k_jaccard": None,
            "common_items": len(common_items),
        }

    pos_a = {item: idx for idx, item in enumerate(unique_a)}
    pos_b = {item: idx for idx, item in enumerate(unique_b)}

    discordant = 0
    concordant = 0
    for idx in range(len(common_items)):
        for jdx in range(idx + 1, len(common_items)):
            left = common_items[idx]
            right = common_items[jdx]
            sign = (pos_a[left] - pos_a[right]) * (pos_b[left] - pos_b[right])
            if sign > 0:
                concordant += 1
            elif sign < 0:
                discordant += 1

    pair_count = concordant + discordant
    kendall_tau = safe_div(concordant - discordant, pair_count) if pair_count else None

    n = len(common_items)
    squared_diff = sum((pos_a[item] - pos_b[item]) ** 2 for item in common_items)
    spearman_denominator = n * (n * n - 1)
    spearman_rho = None
    if spearman_denominator:
        spearman_rho = 1.0 - (6.0 * squared_diff / spearman_denominator)

    top_a = set(unique_a[: min(top_k, len(unique_a))])
    top_b = set(unique_b[: min(top_k, len(unique_b))])
    top_union = top_a | top_b
    top_k_jaccard = None
    if top_union:
        top_k_jaccard = len(top_a & top_b) / len(top_union)

    return {
        "kendall_tau": kendall_tau,
        "spearman_rho": spearman_rho,
        "top_k_jaccard": top_k_jaccard,
        "common_items": len(common_items),
    }


def mean_optional(values: Sequence[Optional[float]]) -> Optional[float]:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def build_complete_groups(
    prediction_paths: Sequence[Path],
    variant_names: Sequence[str],
    group_key: str,
    fallback_group_key: str,
) -> Tuple[Dict[str, Dict[str, dict]], Dict[str, int]]:
    grouped: Dict[str, Dict[str, dict]] = {}
    duplicates = {name: 0 for name in variant_names}

    for path, name in zip(prediction_paths, variant_names):
        for row in load_json_rows(path):
            group_id = resolve_group_id(row, group_key=group_key, fallback_group_key=fallback_group_key)
            if group_id is None:
                continue
            variant_rows = grouped.setdefault(group_id, {})
            if name in variant_rows:
                duplicates[name] += 1
            variant_rows[name] = row

    complete = {
        group_id: rows
        for group_id, rows in grouped.items()
        if all(name in rows for name in variant_names)
    }
    return complete, duplicates


def evaluate_invariance(
    prediction_paths: Sequence[Path],
    variant_names: Sequence[str],
    group_key: str,
    fallback_group_key: str,
    top_k: int,
    include_per_group: bool,
) -> Dict[str, object]:
    complete_groups, duplicates = build_complete_groups(
        prediction_paths=prediction_paths,
        variant_names=variant_names,
        group_key=group_key,
        fallback_group_key=fallback_group_key,
    )
    sorted_group_ids = sorted(complete_groups.keys(), key=lambda value: (len(value), value))

    verdict_items: List[List[str]] = []
    pairwise_verdict_inputs: Dict[str, Tuple[List[str], List[str]]] = {}
    pairwise_ranking_inputs: Dict[str, List[Dict[str, Optional[float]]]] = {}
    per_group_details: List[dict] = []

    variant_pairs = list(combinations(variant_names, 2))
    for left_name, right_name in variant_pairs:
        pair_key = f"{left_name}__vs__{right_name}"
        pairwise_verdict_inputs[pair_key] = ([], [])
        pairwise_ranking_inputs[pair_key] = []

    for group_id in sorted_group_ids:
        rows = complete_groups[group_id]
        verdicts = [normalize_label(rows[name].get("predicted_verdict", "")) for name in variant_names]
        verdict_items.append(verdicts)

        group_detail = {
            "group_id": group_id,
            "verdicts": {name: rows[name].get("predicted_verdict", "") for name in variant_names},
        }

        if include_per_group:
            group_detail["pairwise_ranking"] = {}

        for left_name, right_name in variant_pairs:
            pair_key = f"{left_name}__vs__{right_name}"
            left_labels, right_labels = pairwise_verdict_inputs[pair_key]
            left_labels.append(normalize_label(rows[left_name].get("predicted_verdict", "")))
            right_labels.append(normalize_label(rows[right_name].get("predicted_verdict", "")))

            metrics = ranking_pair_metrics(
                rows[left_name].get("ranked_trace_indices", []),
                rows[right_name].get("ranked_trace_indices", []),
                top_k=top_k,
            )
            pairwise_ranking_inputs[pair_key].append(metrics)
            if include_per_group:
                group_detail["pairwise_ranking"][pair_key] = metrics

        if include_per_group:
            per_group_details.append(group_detail)

    verdict_pairwise_summary: Dict[str, object] = {}
    ranking_pairwise_summary: Dict[str, object] = {}
    for pair_key, (left_labels, right_labels) in pairwise_verdict_inputs.items():
        verdict_pairwise_summary[pair_key] = {
            "num_groups": len(left_labels),
            "raw_agreement": raw_agreement(left_labels, right_labels),
            "cohen_kappa": cohen_kappa(left_labels, right_labels),
        }

    for pair_key, metrics_list in pairwise_ranking_inputs.items():
        ranking_pairwise_summary[pair_key] = {
            "num_groups": len(metrics_list),
            "mean_kendall_tau": mean_optional([item["kendall_tau"] for item in metrics_list]),
            "mean_spearman_rho": mean_optional([item["spearman_rho"] for item in metrics_list]),
            "mean_top_k_jaccard": mean_optional([item["top_k_jaccard"] for item in metrics_list]),
            "mean_common_items": mean_optional([float(item["common_items"]) for item in metrics_list]),
        }

    summary: Dict[str, object] = {
        "prediction_files": [str(path) for path in prediction_paths],
        "variant_names": list(variant_names),
        "group_key": group_key,
        "fallback_group_key": fallback_group_key,
        "top_k": top_k,
        "num_variants": len(variant_names),
        "num_complete_groups": len(sorted_group_ids),
        "duplicate_group_rows": duplicates,
        "verdict_agreement": {
            "raw_complete_agreement": safe_div(
                sum(len(set(item)) == 1 for item in verdict_items),
                len(verdict_items),
            ),
            "fleiss_kappa": fleiss_kappa(verdict_items),
            "krippendorff_alpha_nominal": krippendorff_alpha_nominal(verdict_items),
            "pairwise": verdict_pairwise_summary,
        },
        "ranking_consistency": {
            "pairwise": ranking_pairwise_summary,
        },
    }

    if include_per_group:
        summary["per_group"] = per_group_details

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate cross-language or cross-variant prediction invariance.")
    parser.add_argument("--predictions", nargs="+", type=Path, required=True, help="Prediction JSON files to compare.")
    parser.add_argument("--names", nargs="+", default=None, help="Optional display names for prediction files.")
    parser.add_argument("--group-key", type=str, default="group_id")
    parser.add_argument("--fallback-group-key", type=str, default="dataset_index")
    parser.add_argument("--top-k", type=int, default=5, help="k used for top-k ranking overlap.")
    parser.add_argument("--include-per-group", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    variant_names = resolve_variant_names(args.predictions, args.names)
    results = evaluate_invariance(
        prediction_paths=args.predictions,
        variant_names=variant_names,
        group_key=args.group_key,
        fallback_group_key=args.fallback_group_key,
        top_k=args.top_k,
        include_per_group=args.include_per_group,
    )

    print(json.dumps(results, indent=2, ensure_ascii=False))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
