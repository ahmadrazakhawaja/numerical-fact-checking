#!/usr/bin/env python3
"""Translate CLEF dataset fields to English with googletrans.

Default behavior:
- Reads `train.json` and `validation.json` for `arabic` and `spanish`
- Translates text-heavy fields to English:
  - claim
  - Reasoning_traces
  - evidences
- Keeps class fields via canonical mapping (no translation):
  - label
  - verdict
  - Verdict_list
- Writes outputs to the same language folders as:
  - train_translated.json
  - validation_translated.json
"""

from __future__ import annotations

import asyncio
import argparse
import inspect
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    from googletrans import Translator
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "googletrans is not installed. Install it with: pip install googletrans"
    ) from exc


DEFAULT_LANGUAGES = ("arabic", "spanish")
DEFAULT_SPLITS = ("train", "validation")
LANGUAGE_TO_SOURCE_CODE = {
    "arabic": "ar",
    "spanish": "es",
    "english": "en",
}
CANONICAL_LABELS = {
    "true": "True",
    "false": "False",
    "conflicting": "Conflicting",
}


def normalize_label(label: str) -> str:
    normalized = str(label).strip().lower()
    return CANONICAL_LABELS.get(normalized, str(label).strip())


def chunk_items(items: Sequence[Tuple[int, str]], chunk_size: int) -> Iterable[Sequence[Tuple[int, str]]]:
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def split_long_text(text: str, max_chars: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + max_chars, text_len)
        if end < text_len:
            split_at = text.rfind("\n", start, end)
            if split_at == -1:
                split_at = text.rfind(" ", start, end)
            if split_at > start + (max_chars // 2):
                end = split_at + 1
        chunks.append(text[start:end])
        start = end
    return chunks


class GoogleTransClient:
    def __init__(
        self,
        dest_lang: str,
        batch_size: int,
        max_retries: int,
        retry_wait_seconds: float,
        max_chars_per_request: int,
    ) -> None:
        self.dest_lang = dest_lang
        self.batch_size = max(1, batch_size)
        self.max_retries = max(0, max_retries)
        self.retry_wait_seconds = max(0.0, retry_wait_seconds)
        self.max_chars_per_request = max(500, max_chars_per_request)
        self.translator = Translator()
        self._runner: asyncio.Runner | None = None

        # Cache short strings only to avoid large memory usage on long traces.
        self._short_cache: Dict[Tuple[str, str], str] = {}
        self._cache_limit_chars = 256

    def _get_runner(self) -> asyncio.Runner:
        if self._runner is None:
            self._runner = asyncio.Runner()
        return self._runner

    def _resolve_maybe_awaitable(self, value: object) -> object:
        if inspect.isawaitable(value):
            runner = self._get_runner()
            return runner.run(value)
        return value

    def close(self) -> None:
        # Best-effort close for async Translator client variants.
        translator_client = getattr(self.translator, "client", None)
        aclose = getattr(translator_client, "aclose", None)
        if callable(aclose):
            try:
                self._resolve_maybe_awaitable(aclose())
            except Exception:
                pass

        if self._runner is not None:
            try:
                self._runner.close()
            except Exception:
                pass
            self._runner = None

    def _translate_single(self, text: str, source_lang: str) -> str:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                result = self.translator.translate(
                    text,
                    src=source_lang,
                    dest=self.dest_lang,
                )
                result = self._resolve_maybe_awaitable(result)
                return str(result.text)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_wait_seconds)
        raise RuntimeError(
            f"Failed to translate text after {self.max_retries + 1} attempts."
        ) from last_error

    def _translate_batch(self, texts: Sequence[str], source_lang: str) -> List[str]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                result = self.translator.translate(
                    list(texts),
                    src=source_lang,
                    dest=self.dest_lang,
                )
                result = self._resolve_maybe_awaitable(result)
                if isinstance(result, list):
                    return [str(item.text) for item in result]
                return [str(result.text)]
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_wait_seconds)
        raise RuntimeError(
            f"Failed to translate batch after {self.max_retries + 1} attempts."
        ) from last_error

    def translate_text(self, text: str, source_lang: str) -> str:
        raw = str(text)
        if not raw.strip():
            return raw

        cache_key = (source_lang, raw)
        if len(raw) <= self._cache_limit_chars and cache_key in self._short_cache:
            return self._short_cache[cache_key]

        if len(raw) > self.max_chars_per_request:
            translated = "".join(
                self._translate_single(chunk, source_lang)
                for chunk in split_long_text(raw, self.max_chars_per_request)
            )
        else:
            translated = self._translate_single(raw, source_lang)

        if len(raw) <= self._cache_limit_chars:
            self._short_cache[cache_key] = translated
        return translated

    def translate_texts(self, texts: Sequence[str], source_lang: str) -> List[str]:
        output = [str(text) for text in texts]
        pending: List[Tuple[int, str]] = []

        for idx, value in enumerate(output):
            if not value.strip():
                continue

            cache_key = (source_lang, value)
            if len(value) <= self._cache_limit_chars and cache_key in self._short_cache:
                output[idx] = self._short_cache[cache_key]
                continue

            if len(value) > self.max_chars_per_request:
                translated = "".join(
                    self._translate_single(chunk, source_lang)
                    for chunk in split_long_text(value, self.max_chars_per_request)
                )
                output[idx] = translated
                if len(value) <= self._cache_limit_chars:
                    self._short_cache[cache_key] = translated
                continue

            pending.append((idx, value))

        for batch in chunk_items(pending, self.batch_size):
            batch_indices = [idx for idx, _ in batch]
            batch_texts = [text for _, text in batch]
            batch_translated = self._translate_batch(batch_texts, source_lang)

            if len(batch_translated) != len(batch_texts):
                raise RuntimeError(
                    "googletrans returned an unexpected number of outputs for a batch."
                )

            for idx, translated in zip(batch_indices, batch_translated):
                original = output[idx]
                output[idx] = translated
                cache_key = (source_lang, original)
                if len(original) <= self._cache_limit_chars:
                    self._short_cache[cache_key] = translated

        return output


def translate_row(row: dict, client: GoogleTransClient, source_lang: str) -> dict:
    translated = dict(row)

    if "claim" in row:
        translated["claim"] = client.translate_text(str(row["claim"]), source_lang)

    if "label" in row:
        translated["label"] = normalize_label(row["label"])

    if "verdict" in row:
        translated["verdict"] = normalize_label(row["verdict"])

    if "Reasoning_traces" in row and isinstance(row["Reasoning_traces"], list):
        translated["Reasoning_traces"] = client.translate_texts(
            [str(x) for x in row["Reasoning_traces"]],
            source_lang=source_lang,
        )

    if "evidences" in row and isinstance(row["evidences"], list):
        translated["evidences"] = client.translate_texts(
            [str(x) for x in row["evidences"]],
            source_lang=source_lang,
        )

    if "Verdict_list" in row and isinstance(row["Verdict_list"], list):
        translated["Verdict_list"] = [normalize_label(x) for x in row["Verdict_list"]]

    return translated


def translate_file(
    input_path: Path,
    output_path: Path,
    client: GoogleTransClient,
    source_lang: str,
    progress_every: int,
) -> None:
    with input_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list in {input_path}, got {type(rows).__name__}")

    translated_rows = []
    for idx, row in enumerate(rows, start=1):
        translated_rows.append(translate_row(row, client, source_lang=source_lang))
        if progress_every > 0 and idx % progress_every == 0:
            print(f"  translated {idx}/{len(rows)} rows from {input_path.name}")

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(translated_rows, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate Arabic/Spanish CLEF dataset files to English using googletrans."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset"),
        help="Path to dataset root (contains language subfolders).",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=list(DEFAULT_LANGUAGES),
        help="Languages to translate. Example: arabic spanish",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        choices=["train", "validation"],
        help="Dataset splits to process.",
    )
    parser.add_argument(
        "--destination-language",
        type=str,
        default="en",
        help="Destination language code for googletrans.",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_translated",
        help=(
            "Suffix inserted after the split name when writing outputs. "
            "For example, '_spanish_translated' produces 'train_spanish_translated.json'."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for list translation requests.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Number of retries per failed translation request.",
    )
    parser.add_argument(
        "--retry-wait-seconds",
        type=float,
        default=1.0,
        help="Sleep time between retries.",
    )
    parser.add_argument(
        "--max-chars-per-request",
        type=int,
        default=4500,
        help="Split long strings above this size before translation.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N rows (0 disables progress logs).",
    )
    args = parser.parse_args()

    languages = [lang.strip().lower() for lang in args.languages]
    splits = [split.strip().lower() for split in args.splits]

    client = GoogleTransClient(
        dest_lang=args.destination_language,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        retry_wait_seconds=args.retry_wait_seconds,
        max_chars_per_request=args.max_chars_per_request,
    )

    try:
        for language in languages:
            source_lang = LANGUAGE_TO_SOURCE_CODE.get(language, "auto")
            language_dir = args.dataset_dir / language
            if not language_dir.exists():
                raise FileNotFoundError(f"Language directory not found: {language_dir}")

            print(f"\nLanguage: {language} (src={source_lang} -> dest={args.destination_language})")
            for split in splits:
                input_path = language_dir / f"{split}.json"
                output_path = language_dir / f"{split}{args.output_suffix}.json"
                if not input_path.exists():
                    raise FileNotFoundError(f"Input file not found: {input_path}")

                print(f"Processing {input_path} -> {output_path}")
                translate_file(
                    input_path=input_path,
                    output_path=output_path,
                    client=client,
                    source_lang=source_lang,
                    progress_every=args.progress_every,
                )
                print(f"Saved: {output_path}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
