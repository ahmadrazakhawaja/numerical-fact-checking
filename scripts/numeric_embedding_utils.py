#!/usr/bin/env python3
"""Utilities for optional value-aware numeric embeddings."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


NUMERIC_SPECIAL_TOKEN = "<num>"
NUMERIC_CONFIG_FILENAME = "numeric_embedding_config.json"
NUMERIC_STATE_FILENAME = "numeric_value_encoder.pt"
NUMERIC_CHAR_VOCAB = ["<pad>"] + list("0123456789-+.e")
NUMERIC_CHAR_TO_ID = {ch: idx for idx, ch in enumerate(NUMERIC_CHAR_VOCAB)}
ARABIC_TO_ASCII_DIGIT_MAP = str.maketrans(
    {
        "٠": "0",
        "١": "1",
        "٢": "2",
        "٣": "3",
        "٤": "4",
        "٥": "5",
        "٦": "6",
        "٧": "7",
        "٨": "8",
        "٩": "9",
        "۰": "0",
        "۱": "1",
        "۲": "2",
        "۳": "3",
        "۴": "4",
        "۵": "5",
        "۶": "6",
        "۷": "7",
        "۸": "8",
        "۹": "9",
        "٫": ".",
        "٬": ",",
        "،": ",",
    }
)
NUMERIC_PATTERN = re.compile(
    r"(?<![\w<])(?:[-+]?(?:\d+(?:[.,٫٬،]\d+)*|\d+))(?:%|‰)?"
)


@dataclass
class NumericEmbeddingConfig:
    enabled: bool
    num_token: str
    num_token_id: int
    max_numeric_chars: int
    numeric_char_embedding_dim: int
    numeric_gru_hidden_size: int
    hidden_size: int


def add_numeric_embedding_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--use-numeric-embedding",
        action="store_true",
        help="Enable value-aware numeric embeddings using a learned <num> token representation.",
    )
    parser.add_argument(
        "--max-numeric-chars",
        type=int,
        default=24,
        help="Maximum canonical numeric characters encoded per <num> token.",
    )
    parser.add_argument(
        "--numeric-char-embedding-dim",
        type=int,
        default=32,
        help="Character embedding size for the numeric value encoder.",
    )
    parser.add_argument(
        "--numeric-gru-hidden-size",
        type=int,
        default=128,
        help="GRU hidden size for the numeric value encoder.",
    )


def tokenizer_source_for_adapter(model_id: str, adapter_path: Optional[Path] = None) -> str:
    if adapter_path is not None and (adapter_path / "tokenizer_config.json").exists():
        return str(adapter_path)
    return model_id


def normalize_number_surface(raw: str) -> str:
    normalized = raw.translate(ARABIC_TO_ASCII_DIGIT_MAP).replace(" ", "")
    suffix = ""
    if normalized.endswith("%") or normalized.endswith("‰"):
        suffix = normalized[-1]
        normalized = normalized[:-1]

    if normalized.count(".") > 1 and normalized.count(",") == 0:
        tail = normalized.rsplit(".", 1)[-1]
        if len(tail) <= 2:
            normalized = normalized.replace(".", "")
            normalized = normalized[:-len(tail)] + "." + tail
        else:
            normalized = normalized.replace(".", "")

    if normalized.count(",") > 1 and normalized.count(".") == 0:
        tail = normalized.rsplit(",", 1)[-1]
        if len(tail) <= 2:
            normalized = normalized.replace(",", "")
            normalized = normalized[:-len(tail)] + "." + tail
        else:
            normalized = normalized.replace(",", "")

    if "." in normalized and "," in normalized:
        last_dot = normalized.rfind(".")
        last_comma = normalized.rfind(",")
        if last_dot > last_comma:
            normalized = normalized.replace(",", "")
        else:
            normalized = normalized.replace(".", "")
            normalized = normalized.replace(",", ".")
    elif "," in normalized:
        tail = normalized.rsplit(",", 1)[-1]
        if len(tail) <= 2:
            normalized = normalized.replace(",", ".")
        else:
            normalized = normalized.replace(",", "")

    normalized = re.sub(r"[^0-9eE+.\-]", "", normalized)
    if suffix == "%":
        try:
            return format(float(normalized) / 100.0, ".15g")
        except ValueError:
            return normalized
    return normalized


def canonicalize_numeric_surface(raw: str) -> Optional[str]:
    normalized = normalize_number_surface(raw)
    if not normalized:
        return None
    try:
        value = float(normalized)
    except ValueError:
        return None
    return format(value, ".15g")


def annotate_numeric_text(text: str) -> Tuple[str, List[str]]:
    canonicals: List[str] = []

    def replace(match: re.Match[str]) -> str:
        canonical = canonicalize_numeric_surface(match.group(0))
        if canonical is None:
            return match.group(0)
        canonicals.append(canonical)
        return f"{NUMERIC_SPECIAL_TOKEN} {match.group(0)}"

    return NUMERIC_PATTERN.sub(replace, str(text)), canonicals


def canonical_to_char_ids(canonical: str, max_numeric_chars: int) -> List[int]:
    cleaned = canonical[:max_numeric_chars]
    output = [0] * max_numeric_chars
    for idx, ch in enumerate(cleaned):
        output[idx] = NUMERIC_CHAR_TO_ID.get(ch, 0)
    return output


def build_numeric_dense_features(
    sequence_length: int,
    positions: Sequence[int],
    canonicals: Sequence[str],
    max_numeric_chars: int,
) -> Tuple[List[int], List[List[int]]]:
    numeric_mask = [0] * sequence_length
    numeric_char_ids = [[0] * max_numeric_chars for _ in range(sequence_length)]
    for position, canonical in zip(positions, canonicals):
        if 0 <= position < sequence_length:
            numeric_mask[position] = 1
            numeric_char_ids[position] = canonical_to_char_ids(canonical, max_numeric_chars)
    return numeric_mask, numeric_char_ids


def build_numeric_prompt_features(
    prompt_ids: Sequence[int],
    prompt_numeric_canonicals: Sequence[str],
    num_token_id: int,
    max_numeric_chars: int,
    overflow: int = 0,
    sequence_length: Optional[int] = None,
) -> Tuple[List[int], List[List[int]]]:
    prompt_num_positions = [idx for idx, token_id in enumerate(prompt_ids) if token_id == num_token_id]
    if len(prompt_num_positions) != len(prompt_numeric_canonicals):
        raise ValueError(
            "Number of <num> tokens does not match the parsed numeric values. "
            f"Found {len(prompt_num_positions)} tokens but {len(prompt_numeric_canonicals)} numeric values."
        )

    kept_positions: List[int] = []
    kept_canonicals: List[str] = []
    for position, canonical in zip(prompt_num_positions, prompt_numeric_canonicals):
        if position >= overflow:
            kept_positions.append(position - overflow)
            kept_canonicals.append(canonical)

    target_length = sequence_length if sequence_length is not None else max(0, len(prompt_ids) - overflow)
    return build_numeric_dense_features(
        sequence_length=target_length,
        positions=kept_positions,
        canonicals=kept_canonicals,
        max_numeric_chars=max_numeric_chars,
    )


def ensure_numeric_token(tokenizer) -> int:
    if tokenizer.convert_tokens_to_ids(NUMERIC_SPECIAL_TOKEN) == tokenizer.unk_token_id:
        tokenizer.add_special_tokens({"additional_special_tokens": [NUMERIC_SPECIAL_TOKEN]})
    return tokenizer.convert_tokens_to_ids(NUMERIC_SPECIAL_TOKEN)


def resize_model_embeddings_if_needed(model, tokenizer) -> None:
    embedding = model.get_input_embeddings()
    if embedding is None:
        return
    if embedding.num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))


def numeric_artifacts_path(base_path: Path) -> Tuple[Path, Path]:
    return base_path / NUMERIC_CONFIG_FILENAME, base_path / NUMERIC_STATE_FILENAME


def load_numeric_embedding_config(base_path: Path) -> Optional[NumericEmbeddingConfig]:
    config_path, _ = numeric_artifacts_path(base_path)
    if not config_path.exists():
        return None
    with config_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return NumericEmbeddingConfig(**payload)


class NumericValueEncoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        max_numeric_chars: int,
        numeric_char_embedding_dim: int,
        numeric_gru_hidden_size: int,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.max_numeric_chars = max_numeric_chars
        self.char_embedding = nn.Embedding(
            num_embeddings=len(NUMERIC_CHAR_VOCAB),
            embedding_dim=numeric_char_embedding_dim,
            padding_idx=0,
        )
        self.gru = nn.GRU(
            input_size=numeric_char_embedding_dim,
            hidden_size=numeric_gru_hidden_size,
            batch_first=True,
        )
        self.output_projection = nn.Linear(numeric_gru_hidden_size, output_dim)

    def forward(self, numeric_char_ids: torch.Tensor) -> torch.Tensor:
        if numeric_char_ids.numel() == 0:
            return torch.zeros(
                (0, self.output_dim),
                dtype=self.output_projection.weight.dtype,
                device=self.output_projection.weight.device,
            )

        lengths = numeric_char_ids.ne(0).sum(dim=-1)
        lengths = lengths.clamp_min(1)
        embeddings = self.char_embedding(numeric_char_ids)
        packed = pack_padded_sequence(
            embeddings,
            lengths=lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        return self.output_projection(hidden[-1])


class NumericValueEmbeddingWrapper(nn.Module):
    def __init__(
        self,
        base_model,
        config: NumericEmbeddingConfig,
        value_encoder: Optional[NumericValueEncoder] = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.numeric_embedding_config = config
        self.value_encoder = value_encoder or NumericValueEncoder(
            output_dim=config.hidden_size,
            max_numeric_chars=config.max_numeric_chars,
            numeric_char_embedding_dim=config.numeric_char_embedding_dim,
            numeric_gru_hidden_size=config.numeric_gru_hidden_size,
        )
        self.config = base_model.config
        self.generation_config = getattr(base_model, "generation_config", None)
        self.value_encoder.to(base_model.get_input_embeddings().weight.device)

    @property
    def hf_device_map(self):
        device_map = getattr(self.base_model, "hf_device_map", None)
        if device_map is None:
            raise AttributeError("hf_device_map is not set on the wrapped base model.")
        return device_map

    @property
    def device(self):
        return next(self.parameters()).device

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def save_pretrained(self, save_directory: str | Path, **kwargs) -> None:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        self.base_model.save_pretrained(save_path, **kwargs)
        config_path, state_path = numeric_artifacts_path(save_path)
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self.numeric_embedding_config), f, indent=2, ensure_ascii=False)
        torch.save(self.value_encoder.state_dict(), state_path)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        numeric_mask=None,
        numeric_char_ids=None,
        **kwargs,
    ):
        if numeric_mask is not None and numeric_char_ids is not None and input_ids is not None:
            mask = numeric_mask.to(dtype=torch.bool, device=input_ids.device)
            if mask.any():
                input_embeddings = self.base_model.get_input_embeddings()(input_ids)
                selected_numeric = numeric_char_ids[mask].to(device=self.value_encoder.output_projection.weight.device)
                encoded = self.value_encoder(selected_numeric).to(dtype=input_embeddings.dtype, device=input_embeddings.device)
                input_embeddings = input_embeddings.clone()
                input_embeddings[mask] = encoded
                kwargs["inputs_embeds"] = input_embeddings
                input_ids = None

        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )


def build_numeric_embedding_config(
    model,
    tokenizer,
    *,
    max_numeric_chars: int,
    numeric_char_embedding_dim: int,
    numeric_gru_hidden_size: int,
) -> NumericEmbeddingConfig:
    return NumericEmbeddingConfig(
        enabled=True,
        num_token=NUMERIC_SPECIAL_TOKEN,
        num_token_id=tokenizer.convert_tokens_to_ids(NUMERIC_SPECIAL_TOKEN),
        max_numeric_chars=max_numeric_chars,
        numeric_char_embedding_dim=numeric_char_embedding_dim,
        numeric_gru_hidden_size=numeric_gru_hidden_size,
        hidden_size=int(model.config.hidden_size),
    )


def wrap_model_with_numeric_embeddings(
    model,
    tokenizer,
    *,
    artifact_path: Optional[Path],
    enable_numeric_embedding: bool,
    max_numeric_chars: int,
    numeric_char_embedding_dim: int,
    numeric_gru_hidden_size: int,
    freeze_value_encoder: bool = False,
):
    if not enable_numeric_embedding:
        return model, None

    ensure_numeric_token(tokenizer)
    resize_model_embeddings_if_needed(model, tokenizer)

    loaded_config = load_numeric_embedding_config(artifact_path) if artifact_path is not None else None
    if loaded_config is not None:
        config = loaded_config
        config.num_token_id = tokenizer.convert_tokens_to_ids(config.num_token)
    else:
        config = build_numeric_embedding_config(
            model,
            tokenizer,
            max_numeric_chars=max_numeric_chars,
            numeric_char_embedding_dim=numeric_char_embedding_dim,
            numeric_gru_hidden_size=numeric_gru_hidden_size,
        )

    wrapped = NumericValueEmbeddingWrapper(base_model=model, config=config)
    if artifact_path is not None:
        _, state_path = numeric_artifacts_path(artifact_path)
        if state_path.exists():
            state_dict = torch.load(state_path, map_location="cpu")
            wrapped.value_encoder.load_state_dict(state_dict)

    if freeze_value_encoder:
        for parameter in wrapped.value_encoder.parameters():
            parameter.requires_grad = False

    return wrapped, config
