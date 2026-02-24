# Numerical fact checking

## Mini verifier experiments (5-sample setup)

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
