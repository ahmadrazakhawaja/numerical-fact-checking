# Numerical fact checking

## English SFT + LoRA

Train an English-only SFT adapter on `Mistral-7B-Instruct-v0.3`:

```bash
python scripts/train_english_sft_lora.py \
  --train-dataset dataset/english/train.json \
  --validation-dataset dataset/english/validation.json \
  --output-dir checkpoints/english_sft_lora \
  --gradient-checkpointing
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
