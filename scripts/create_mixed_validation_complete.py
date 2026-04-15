#!/usr/bin/env python3
"""Create mixed validation_complete files with configurable trace-language mixing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

from task2_utils import load_json_rows


DEFAULT_LANGUAGES = ("english", "spanish", "arabic")
DEFAULT_INPUT_FILENAME = "validation_complete.json"
DEFAULT_OUTPUT_FILENAME = "validation_complete_mixed.json"
DEFAULT_PAIRWISE_OUTPUT_TEMPLATE = "validation_complete_traces_{trace_language}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create mixed-language validation files where claim/evidence stay in the "
            "target language and reasoning traces are either cycled across all "
            "configured languages or mixed between the target language and one paired "
            "trace language."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset"),
        help="Dataset root containing one folder per language.",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=list(DEFAULT_LANGUAGES),
        help="Languages to load and mix in cyclic order.",
    )
    parser.add_argument(
        "--input-filename",
        default=DEFAULT_INPUT_FILENAME,
        help="Per-language source filename to read.",
    )
    parser.add_argument(
        "--output-filename",
        default=DEFAULT_OUTPUT_FILENAME,
        help="Per-language output filename to write in cyclic mode.",
    )
    parser.add_argument(
        "--mode",
        choices=("cyclic", "pairwise"),
        default="cyclic",
        help=(
            "Mixing mode: cyclic across all languages, or pairwise alternating "
            "between the target language and one paired trace language."
        ),
    )
    parser.add_argument(
        "--pairwise-output-template",
        default=DEFAULT_PAIRWISE_OUTPUT_TEMPLATE,
        help=(
            "Filename template for pairwise mode. Available fields: "
            "{target_language}, {trace_language}."
        ),
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=4,
        help="JSON indentation for output files.",
    )
    return parser.parse_args()


def language_file(dataset_dir: Path, language: str, filename: str) -> Path:
    return dataset_dir / language / filename


def load_language_rows(
    dataset_dir: Path,
    languages: Sequence[str],
    input_filename: str,
) -> Dict[str, List[dict]]:
    return {
        language: load_json_rows(language_file(dataset_dir, language, input_filename))
        for language in languages
    }


def validate_alignment(rows_by_language: Dict[str, List[dict]]) -> None:
    languages = list(rows_by_language)
    expected_count = len(rows_by_language[languages[0]])

    for language, rows in rows_by_language.items():
        if len(rows) != expected_count:
            raise ValueError(
                f"Row count mismatch for {language}: expected {expected_count}, got {len(rows)}."
            )

    for row_index in range(expected_count):
        trace_counts = {
            language: len(rows_by_language[language][row_index].get("Reasoning_traces", []))
            for language in languages
        }
        if len(set(trace_counts.values())) != 1:
            raise ValueError(
                f"Trace count mismatch at row {row_index}: {trace_counts}."
            )

        labels = {
            language: str(rows_by_language[language][row_index].get("label", "")).strip()
            for language in languages
        }
        if len(set(labels.values())) != 1:
            raise ValueError(f"Label mismatch at row {row_index}: {labels}.")

        verdict_lists = {
            language: rows_by_language[language][row_index].get("Verdict_list", [])
            for language in languages
        }
        normalized_verdict_lists = {
            language: [str(value).strip() for value in values]
            for language, values in verdict_lists.items()
        }
        if len({tuple(values) for values in normalized_verdict_lists.values()}) != 1:
            raise ValueError(
                f"Verdict_list mismatch at row {row_index}: {normalized_verdict_lists}."
            )


def build_cyclic_mixed_traces(
    row_index: int,
    rows_by_language: Dict[str, List[dict]],
    languages: Sequence[str],
) -> List[str]:
    trace_count = len(rows_by_language[languages[0]][row_index]["Reasoning_traces"])
    mixed_traces: List[str] = []

    for trace_index in range(trace_count):
        source_language = languages[trace_index % len(languages)]
        source_row = rows_by_language[source_language][row_index]
        mixed_traces.append(source_row["Reasoning_traces"][trace_index])

    return mixed_traces


def build_pairwise_mixed_traces(
    row_index: int,
    rows_by_language: Dict[str, List[dict]],
    target_language: str,
    trace_language: str,
) -> List[str]:
    trace_count = len(rows_by_language[target_language][row_index]["Reasoning_traces"])
    mixed_traces: List[str] = []

    for trace_index in range(trace_count):
        source_language = target_language if trace_index % 2 == 0 else trace_language
        source_row = rows_by_language[source_language][row_index]
        mixed_traces.append(source_row["Reasoning_traces"][trace_index])

    return mixed_traces


def create_cyclic_mixed_rows(
    target_language: str,
    rows_by_language: Dict[str, List[dict]],
    languages: Sequence[str],
) -> List[dict]:
    target_rows = rows_by_language[target_language]
    mixed_rows: List[dict] = []

    for row_index, target_row in enumerate(target_rows):
        mixed_row = dict(target_row)
        mixed_row["Reasoning_traces"] = build_cyclic_mixed_traces(
            row_index, rows_by_language, languages
        )
        mixed_rows.append(mixed_row)

    return mixed_rows


def create_pairwise_mixed_rows(
    target_language: str,
    trace_language: str,
    rows_by_language: Dict[str, List[dict]],
) -> List[dict]:
    target_rows = rows_by_language[target_language]
    mixed_rows: List[dict] = []

    for row_index, target_row in enumerate(target_rows):
        mixed_row = dict(target_row)
        mixed_row["Reasoning_traces"] = build_pairwise_mixed_traces(
            row_index,
            rows_by_language,
            target_language,
            trace_language,
        )
        mixed_rows.append(mixed_row)

    return mixed_rows


def write_json(path: Path, rows: Sequence[dict], indent: int) -> None:
    path.write_text(
        json.dumps(list(rows), ensure_ascii=False, indent=indent) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    rows_by_language = load_language_rows(
        dataset_dir=args.dataset_dir,
        languages=args.languages,
        input_filename=args.input_filename,
    )
    validate_alignment(rows_by_language)

    if args.mode == "cyclic":
        for target_language in args.languages:
            output_path = language_file(args.dataset_dir, target_language, args.output_filename)
            mixed_rows = create_cyclic_mixed_rows(target_language, rows_by_language, args.languages)
            write_json(output_path, mixed_rows, args.indent)
            print(f"Wrote {len(mixed_rows)} rows to {output_path}")
        return

    for target_language in args.languages:
        for trace_language in args.languages:
            if trace_language == target_language:
                continue
            output_filename = args.pairwise_output_template.format(
                target_language=target_language,
                trace_language=trace_language,
            )
            output_path = language_file(args.dataset_dir, target_language, output_filename)
            mixed_rows = create_pairwise_mixed_rows(
                target_language=target_language,
                trace_language=trace_language,
                rows_by_language=rows_by_language,
            )
            write_json(output_path, mixed_rows, args.indent)
            print(
                f"Wrote {len(mixed_rows)} rows to {output_path} "
                f"(claim/evidence={target_language}, traces alternate "
                f"{target_language}/{trace_language})"
            )


if __name__ == "__main__":
    main()
