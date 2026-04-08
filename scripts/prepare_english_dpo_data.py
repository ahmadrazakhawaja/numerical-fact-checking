#!/usr/bin/env python3
"""Prepare English-only DPO prompt/chosen/rejected pairs for CLEF Task 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence

try:
    from task2_ranking_utils import build_dpo_pairs, build_prompt, infer_label_space
    from task2_utils import load_json_rows
except ImportError:  # pragma: no cover - import path fallback
    from scripts.task2_ranking_utils import build_dpo_pairs, build_prompt, infer_label_space
    from scripts.task2_utils import load_json_rows


def maybe_limit_rows(rows: Sequence[dict], limit: Optional[int]) -> List[dict]:
    if limit is None:
        return list(rows)
    if limit < 0:
        raise ValueError(f"limit must be >= 0, got {limit}")
    return list(rows[:limit])


def build_dpo_records(
    rows: Sequence[dict],
    label_space: Sequence[str],
    max_evidence_items: int,
    max_evidence_chars: int,
    max_trace_chars: int,
    max_rejected_per_prompt: int,
) -> List[dict]:
    records: List[dict] = []
    for dataset_index, row in enumerate(rows):
        prompt = build_prompt(
            row=row,
            label_space=label_space,
            max_evidence_items=max_evidence_items,
            max_evidence_chars=max_evidence_chars,
            max_trace_chars=max_trace_chars,
        )
        for pair_index, pair in enumerate(
            build_dpo_pairs(
                row=row,
                label_space=label_space,
                max_rejected_per_prompt=max_rejected_per_prompt,
            )
        ):
            records.append(
                {
                    "dataset_index": dataset_index,
                    "pair_index": pair_index,
                    "prompt": prompt,
                    "chosen": pair["chosen"],
                    "rejected": pair["rejected"],
                    "label": row.get("label", ""),
                }
            )
    return records


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare English-only DPO data from the complete English splits.")
    parser.add_argument("--train-dataset", type=Path, default=Path("dataset/english/train_complete.json"))
    parser.add_argument("--validation-dataset", type=Path, default=Path("dataset/english/validation_complete.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-validation", type=int, default=None)
    parser.add_argument("--max-evidence-items", type=int, default=2)
    parser.add_argument("--max-evidence-chars", type=int, default=512)
    parser.add_argument("--max-trace-chars", type=int, default=160)
    parser.add_argument("--max-rejected-per-prompt", type=int, default=2)
    args = parser.parse_args()

    train_rows = maybe_limit_rows(load_json_rows(args.train_dataset), args.limit_train)
    validation_rows = maybe_limit_rows(load_json_rows(args.validation_dataset), args.limit_validation)
    label_space = infer_label_space(train_rows + validation_rows)

    train_records = build_dpo_records(
        rows=train_rows,
        label_space=label_space,
        max_evidence_items=args.max_evidence_items,
        max_evidence_chars=args.max_evidence_chars,
        max_trace_chars=args.max_trace_chars,
        max_rejected_per_prompt=args.max_rejected_per_prompt,
    )
    validation_records = build_dpo_records(
        rows=validation_rows,
        label_space=label_space,
        max_evidence_items=args.max_evidence_items,
        max_evidence_chars=args.max_evidence_chars,
        max_trace_chars=args.max_trace_chars,
        max_rejected_per_prompt=args.max_rejected_per_prompt,
    )

    train_path = args.output_dir / "train_dpo.json"
    validation_path = args.output_dir / "validation_dpo.json"
    summary_path = args.output_dir / "summary.json"

    write_json(train_path, train_records)
    write_json(validation_path, validation_records)
    write_json(
        summary_path,
        {
            "train_dataset": str(args.train_dataset),
            "validation_dataset": str(args.validation_dataset),
            "label_space": list(label_space),
            "num_train_rows": len(train_rows),
            "num_validation_rows": len(validation_rows),
            "num_train_pairs": len(train_records),
            "num_validation_pairs": len(validation_records),
            "max_rejected_per_prompt": args.max_rejected_per_prompt,
        },
    )

    print(
        json.dumps(
            {
                "train_output": str(train_path),
                "validation_output": str(validation_path),
                "summary_output": str(summary_path),
                "num_train_pairs": len(train_records),
                "num_validation_pairs": len(validation_records),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
