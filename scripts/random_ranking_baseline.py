#!/usr/bin/env python3
"""Random-ranking baseline for CLEF Task 2.

Workflow:
1) Read train split across selected languages (for label-space sanity checks).
2) Generate random ranking permutations for each claim in validation.
3) Predict verdict from top-ranked trace verdict (top-1 by default).
4) Write predictions JSON.
5) Optionally run the evaluator script and print metrics.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

try:
    from task2_utils import infer_language_name, load_dataset_rows, load_json_rows, normalize_label
except ImportError:  # pragma: no cover - import path fallback
    from scripts.task2_utils import infer_language_name, load_dataset_rows, load_json_rows, normalize_label


DEFAULT_LANGS = ("english", "spanish", "arabic")


def collect_train_label_space(dataset_dir: Path, languages: Iterable[str]) -> Set[str]:
    labels: Set[str] = set()
    for lang in languages:
        for row in load_dataset_rows(dataset_dir, "train", lang):
            labels.add(normalize_label(row.get("verdict", "")))
    return labels


def choose_predicted_verdict(verdict_list: Sequence[str], ranked: Sequence[int], top_n: int) -> str:
    top_n = max(1, top_n)
    picked = [normalize_label(verdict_list[i]) for i in ranked[:top_n]]

    # Majority vote over top-N. Tie-break by earliest occurrence in ranking.
    counts: Dict[str, int] = {}
    for v in picked:
        counts[v] = counts.get(v, 0) + 1

    best_label = ""
    best_count = -1
    for v in picked:
        c = counts[v]
        if c > best_count:
            best_count = c
            best_label = v
    return best_label


def build_predictions(
    dataset_dir: Path,
    languages: Iterable[str],
    target_split: str,
    seed: int,
    top_n_for_verdict: int,
) -> List[dict]:
    rng = random.Random(seed)
    predictions: List[dict] = []

    for lang in languages:
        rows = load_dataset_rows(dataset_dir, target_split, lang)
        for idx, row in enumerate(rows):
            num_traces = len(row.get("Reasoning_traces", []))
            verdict_list = row.get("Verdict_list", [])

            ranked = list(range(num_traces))
            rng.shuffle(ranked)

            pred_verdict = choose_predicted_verdict(verdict_list, ranked, top_n=top_n_for_verdict)

            predictions.append(
                {
                    "language": lang,
                    "dataset_index": idx,
                    "ranked_trace_indices": ranked,
                    "predicted_verdict": pred_verdict,
                }
            )

    return predictions


def build_predictions_from_file(
    dataset_path: Path,
    language: str,
    seed: int,
    top_n_for_verdict: int,
) -> List[dict]:
    rng = random.Random(seed)
    predictions: List[dict] = []

    rows = load_json_rows(dataset_path)
    for idx, row in enumerate(rows):
        num_traces = len(row.get("Reasoning_traces", []))
        verdict_list = row.get("Verdict_list", [])

        ranked = list(range(num_traces))
        rng.shuffle(ranked)

        pred_verdict = choose_predicted_verdict(verdict_list, ranked, top_n=top_n_for_verdict)

        predictions.append(
            {
                "language": language,
                "dataset_index": idx,
                "ranked_trace_indices": ranked,
                "predicted_verdict": pred_verdict,
            }
        )

    return predictions


def maybe_run_evaluator(
    dataset_dir: Path,
    predictions_path: Path,
    languages: List[str],
    target_split: str,
    seed: int,
    k: int,
    split: str,
    eval_output: Path | None = None,
    dataset_path: Path | None = None,
    language_name: str | None = None,
) -> None:
    evaluator = Path(__file__).with_name("evaluate_clef_task2.py")
    output_path = eval_output or Path(f"results/random_baseline_eval_{target_split}_results_seed{seed}.json")
    cmd = [
        sys.executable,
        str(evaluator),
        "--predictions",
        str(predictions_path),
        "--k",
        str(k),
        "--output",
        str(output_path),
    ]
    if dataset_path is not None:
        cmd.extend(["--dataset-path", str(dataset_path)])
        if language_name is not None:
            cmd.extend(["--language-name", language_name])
    else:
        cmd.extend(
            [
                "--dataset-dir",
                str(dataset_dir),
                "--split",
                split,
                "--languages",
                *languages,
            ]
        )
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build random baseline predictions for CLEF Task 2.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"), help="Path to dataset root.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Optional path to a single dataset JSON file, e.g. a gold-label test file.",
    )
    parser.add_argument(
        "--language-name",
        type=str,
        default=None,
        help="Language identifier to write in predictions when --dataset-path is used.",
    )
    parser.add_argument("--languages", nargs="+", default=list(DEFAULT_LANGS), help="Languages to include.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--top-n-for-verdict",
        type=int,
        default=1,
        help="Predict verdict via majority vote over top-N ranked traces (default top-1).",
    )
    parser.add_argument("--output", type=Path, default=Path("predictions_random_val.json"))
    parser.add_argument("--eval-k", type=int, default=5)
    parser.add_argument(
        "--target-split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
        help="Dataset split to generate predictions for.",
    )
    parser.add_argument("--evaluate", action="store_true", help="Run evaluator after writing predictions.")
    parser.add_argument("--eval-output", type=Path, default=None, help="Optional evaluator metrics JSON path.")
    args = parser.parse_args()

    languages = [x.strip().lower() for x in args.languages]
    language_name = (
        str(args.language_name).strip().lower()
        if args.language_name
        else (infer_language_name(args.dataset_path) if args.dataset_path is not None else None)
    )

    train_labels = collect_train_label_space(args.dataset_dir, [language_name] if language_name else languages)
    if not train_labels:
        raise ValueError("No train labels found. Check --dataset-dir and --languages.")

    if args.dataset_path is not None:
        preds = build_predictions_from_file(
            dataset_path=args.dataset_path,
            language=language_name or infer_language_name(args.dataset_path),
            seed=args.seed,
            top_n_for_verdict=args.top_n_for_verdict,
        )
    else:
        preds = build_predictions(
            dataset_dir=args.dataset_dir,
            languages=languages,
            target_split=args.target_split,
            seed=args.seed,
            top_n_for_verdict=args.top_n_for_verdict,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2, ensure_ascii=False)

    print(
        json.dumps(
            {
                "saved_predictions": str(args.output),
                "num_predictions": len(preds),
                "languages": [language_name] if args.dataset_path is not None else languages,
                "dataset_path": str(args.dataset_path) if args.dataset_path is not None else None,
                "target_split": args.target_split,
                "seed": args.seed,
                "top_n_for_verdict": args.top_n_for_verdict,
                "train_label_space": sorted(train_labels),
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    if args.evaluate:
        maybe_run_evaluator(
            dataset_dir=args.dataset_dir,
            predictions_path=args.output,
            languages=languages,
            k=args.eval_k,
            split=args.target_split,
            target_split=args.target_split,
            seed=args.seed,
            eval_output=args.eval_output,
            dataset_path=args.dataset_path,
            language_name=language_name,
        )


if __name__ == "__main__":
    main()
