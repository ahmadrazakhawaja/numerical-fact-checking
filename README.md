# Numerical fact checking

## English SFT + LoRA

Train an English-only SFT adapter on `Mistral-7B-Instruct-v0.3`:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/train_english_sft_lora.py \
  --train-dataset dataset/english/train.json \
  --validation-dataset dataset/english/validation.json \
  --output-dir checkpoints/english_sft_lora \
  --attn-implementation sdpa
```

Train with QLoRA-style 4-bit loading:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/train_english_sft_lora.py \
  --train-dataset dataset/english/train.json \
  --validation-dataset dataset/english/validation.json \
  --output-dir checkpoints/english_sft_qlora \
  --load-in-4bit \
  --target-modules all-linear \
  --attn-implementation sdpa
```

Train with QLoRA-style 4-bit loading plus value-aware numeric embeddings:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/train_english_sft_lora.py \
  --train-dataset dataset/english/train_complete.json \
  --validation-dataset dataset/english/validation_complete.json \
  --output-dir checkpoints/english_sft_qlora_numeric \
  --load-in-4bit \
  --target-modules all-linear \
  --attn-implementation sdpa \
  --use-numeric-embedding
```

Run inference from the saved adapter:

```bash
python scripts/infer_sft_lora.py \
  --dataset-path dataset/english/validation.json \
  --language-name english \
  --adapter-path checkpoints/english_sft_lora/final_adapter \
  --output results/english_sft_lora_validation_predictions.json \
  --evaluate \
  --eval-output results/english_sft_lora_validation_eval.json
```

Run inference with 4-bit base-model loading:

```bash
python scripts/infer_sft_lora.py \
  --dataset-path dataset/english/validation.json \
  --language-name english \
  --adapter-path checkpoints/english_sft_qlora/final_adapter \
  --output results/english_sft_qlora_validation_predictions.json \
  --load-in-4bit \
  --attn-implementation sdpa
```

Run inference with 4-bit base-model loading plus numeric embeddings:

```bash
python scripts/infer_sft_lora.py \
  --dataset-path dataset/english/validation_complete.json \
  --language-name english \
  --adapter-path checkpoints/english_sft_qlora_numeric/final_adapter \
  --output results/english_sft_qlora_numeric_validation_complete_predictions.json \
  --load-in-4bit \
  --device-map auto \
  --attn-implementation sdpa \
  --use-numeric-embedding
```

## English DPO + LoRA

Prepare deterministic DPO pairs from the English complete splits:

```bash
python scripts/prepare_english_dpo_data.py \
  --train-dataset dataset/english/train_complete.json \
  --validation-dataset dataset/english/validation_complete.json \
  --output-dir dataset_samples/english_dpo_pairs
```

Train DPO starting from an existing SFT adapter:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/train_english_dpo_lora.py \
  --train-dataset dataset/english/train_complete.json \
  --validation-dataset dataset/english/validation_complete.json \
  --sft-adapter-path checkpoints/english_sft_qlora/final_adapter \
  --output-dir checkpoints/english_dpo_qlora \
  --load-in-4bit \
  --device-map auto \
  --attn-implementation sdpa
```

Train numeric-aware DPO starting from a numeric-aware SFT adapter:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/train_english_dpo_lora.py \
  --train-dataset dataset/english/train_complete.json \
  --validation-dataset dataset/english/validation_complete.json \
  --sft-adapter-path checkpoints/english_sft_qlora_numeric/final_adapter \
  --output-dir checkpoints/english_dpo_qlora_numeric \
  --load-in-4bit \
  --device-map auto \
  --attn-implementation sdpa \
  --use-numeric-embedding
```

Run inference from the DPO adapter:

```bash
python scripts/infer_sft_lora.py \
  --dataset-path dataset/english/validation_complete.json \
  --language-name english \
  --adapter-path checkpoints/english_dpo_qlora/final_adapter \
  --output results/english_dpo_qlora_validation_complete_predictions.json \
  --load-in-4bit \
  --device-map auto \
  --attn-implementation sdpa \
  --evaluate \
  --eval-output results/english_dpo_qlora_validation_complete_eval.json
```

Run inference from the numeric-aware DPO adapter:

```bash
python scripts/infer_sft_lora.py \
  --dataset-path dataset/english/validation_complete.json \
  --language-name english \
  --adapter-path checkpoints/english_dpo_qlora_numeric/final_adapter \
  --output results/english_dpo_qlora_numeric_validation_complete_predictions.json \
  --load-in-4bit \
  --device-map auto \
  --attn-implementation sdpa \
  --use-numeric-embedding \
  --evaluate \
  --eval-output results/english_dpo_qlora_numeric_validation_complete_eval.json
```

Quick debug run on a small subset:

```bash
python scripts/train_english_sft_lora.py \
  --train-dataset dataset/english/train.json \
  --validation-dataset dataset/english/validation.json \
  --output-dir checkpoints/english_sft_lora_debug \
  --limit-train 32 \
  --limit-validation 16 \
  --gradient-checkpointing
```

## Prompted Baseline

Run the frozen prompted baseline on a small validation subset:

```bash
python scripts/prompt_ranking_baseline.py \
  --dataset-path dataset/english/validation.json \
  --language-name english \
  --output results/prompt_baseline_english_val_10.json \
  --limit 10 \
  --evaluate \
  --eval-output results/prompt_baseline_english_val_10_eval.json
```

## Mini verifier experiments (legacy 5-sample setup)

Train the 3 requested settings on `dataset/english/train.json`:

```bash
python scripts/train_mini_verifier_experiments.py \
  --dataset-path dataset/english/train.json \
  --num-claims 5 \
  --modes ntp cls_frozen cls_unfrozen \
  --adapter-id IlyaGusev/saiga_mistral_7b_lora \
  --output-root checkpoints/mini_factcheck
```

Run inference from saved checkpoints:

```bash
python scripts/infer_mini_verifier_experiments.py \
  --dataset-path dataset/english/train.json \
  --num-claims 5 \
  --checkpoint-root checkpoints/mini_factcheck \
  --modes ntp cls_frozen cls_unfrozen \
  --output results/mini_factcheck_inference.json
```

Notes:
- If `--model-id` is omitted, the base model is inferred from the LoRA adapter config.
- Mistral checkpoints may require a Hugging Face token/license acceptance.
- Checkpoints are written under `checkpoints/` and ignored by git.
- `--load-in-4bit` requires `bitsandbytes`. If you use `uv sync --frozen` on the cluster, add `bitsandbytes` to the environment and refresh `uv.lock` before submitting the job.
