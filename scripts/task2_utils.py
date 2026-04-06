#!/usr/bin/env python3
"""Shared helpers for CLEF Task 2 scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List


def normalize_label(label: str) -> str:
    return str(label).strip().lower()


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def load_json_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dataset_split_path(dataset_dir: Path, split: str, language: str) -> Path:
    return dataset_dir / language / f"{split}.json"


def load_dataset_rows(dataset_dir: Path, split: str, language: str) -> List[dict]:
    return load_json_rows(dataset_split_path(dataset_dir, split, language))


def infer_language_name(dataset_path: Path) -> str:
    parent = dataset_path.parent.name.strip().lower()
    stem = dataset_path.stem.strip().lower()
    if parent and parent != "dataset":
        if stem.endswith("_translated"):
            return f"{parent}_translated"
        if stem in {"train", "validation", "test"}:
            return parent
    return stem
