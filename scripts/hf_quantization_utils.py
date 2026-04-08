#!/usr/bin/env python3
"""Shared helpers for optional bitsandbytes 4-bit loading."""

from __future__ import annotations

import argparse
import importlib.util

import torch


def add_4bit_loading_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Enable bitsandbytes 4-bit loading. Use this with LoRA for QLoRA-style training.",
    )
    parser.add_argument(
        "--bnb-4bit-quant-type",
        type=str,
        default="nf4",
        choices=["nf4", "fp4"],
        help="Quantization data type for bitsandbytes 4-bit loading.",
    )
    parser.add_argument(
        "--bnb-4bit-compute-dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Computation dtype used by bitsandbytes 4-bit layers.",
    )
    parser.add_argument(
        "--bnb-4bit-use-double-quant",
        dest="bnb_4bit_use_double_quant",
        action="store_true",
        help="Enable nested quantization for 4-bit loading.",
    )
    parser.add_argument(
        "--no-bnb-4bit-use-double-quant",
        dest="bnb_4bit_use_double_quant",
        action="store_false",
        help="Disable nested quantization for 4-bit loading.",
    )
    parser.set_defaults(bnb_4bit_use_double_quant=True)


def resolve_bnb_compute_dtype(name: str, fallback_dtype) -> torch.dtype:
    lowered = name.strip().lower()
    if lowered == "auto":
        if isinstance(fallback_dtype, torch.dtype):
            return fallback_dtype
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if lowered == "bfloat16":
        return torch.bfloat16
    if lowered == "float16":
        return torch.float16
    if lowered == "float32":
        return torch.float32
    raise ValueError(f"Unsupported bitsandbytes compute dtype: {name}")


def build_4bit_quantization_config(
    *,
    load_in_4bit: bool,
    quant_type: str,
    compute_dtype_name: str,
    use_double_quant: bool,
    fallback_dtype,
):
    if not load_in_4bit:
        return None

    if importlib.util.find_spec("bitsandbytes") is None:
        raise RuntimeError(
            "bitsandbytes is required for --load-in-4bit but is not installed in this environment."
        )

    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=use_double_quant,
        bnb_4bit_compute_dtype=resolve_bnb_compute_dtype(compute_dtype_name, fallback_dtype),
    )


def prepare_model_for_4bit_training(model, *, use_gradient_checkpointing: bool):
    from peft import prepare_model_for_kbit_training

    try:
        return prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=use_gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    except TypeError:
        try:
            return prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=use_gradient_checkpointing,
            )
        except TypeError:
            return prepare_model_for_kbit_training(model)
