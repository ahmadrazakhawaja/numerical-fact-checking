# Numerical fact checking

## Trace Scorer Baseline

Create filtered training/validation copies that remove claims with zero
label-matching traces:

```bash
python scripts/filter_zero_positive_claims.py \
  --dataset-dir dataset \
  --languages english spanish arabic \
  --splits train validation \
  --report-path results/zero_positive_filter_report.json
```

This writes:
- `dataset/english/train_positive_only.json`
- `dataset/english/validation_positive_only.json`
- `dataset/spanish/train_positive_only.json`
- `dataset/spanish/validation_positive_only.json`
- `dataset/arabic/train_positive_only.json`
- `dataset/arabic/validation_positive_only.json`

Train the binary trace scorer baseline on the English complete splits:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/train_trace_scorer.py \
  --train-dataset dataset/english/train_complete.json \
  --validation-dataset dataset/english/validation_complete.json \
  --output-dir checkpoints/english_trace_scorer_qlora \
  --load-in-4bit \
  --target-modules all-linear \
  --attn-implementation sdpa
```

The trace scorer input now includes truncated evidence snippets in addition to
the claim, trace verdict, and justification. Control this context with
`--max-evidence-items` and `--max-evidence-chars` if you need to trade recall
against sequence length.

For longer runs, enable early stopping so the final adapter is loaded from the
best validation checkpoint instead of blindly using the last epoch:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/train_trace_scorer.py \
  --train-dataset dataset/english/train_complete.json \
  --validation-dataset dataset/english/validation_complete.json \
  --output-dir checkpoints/english_trace_scorer_qlora_early_stop \
  --load-in-4bit \
  --target-modules all-linear \
  --attn-implementation sdpa \
  --num-train-epochs 4 \
  --learning-rate 1e-4 \
  --early-stopping-patience 1 \
  --metric-for-best-model positive_f1
```

Early stopping requires matching save/eval strategies. The defaults already use
`--save-strategy epoch` and `--eval-strategy epoch`.

Enable Weights & Biases logging with `--report-to wandb`. The Trainer will log
train/eval metrics at each logging/evaluation interval, and the run metadata is
also written locally under the checkpoint directory:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/train_trace_scorer.py \
  --train-dataset dataset/english/train_complete.json \
  --validation-dataset dataset/english/validation_complete.json \
  --output-dir checkpoints/english_trace_scorer_qlora_wandb \
  --load-in-4bit \
  --target-modules all-linear \
  --attn-implementation sdpa \
  --report-to wandb \
  --run-name english-trace-scorer-qlora \
  --wandb-project numerical-fact-checking \
  --wandb-group trace-scorer \
  --wandb-tags english qlora
```

Use `--wandb-mode offline` on clusters without outbound network access, then run
`wandb sync` later from a machine with internet access.

The inference script also accepts these reporting flags. When `--evaluate` is
set, it logs the downstream Task2 metrics under keys like
`task2/english/macro_f1` and `task2/english/recall_at_k`.

If the scheduler walltime expires before a long run finishes, resubmit with the
same output directory and add `--resume-from-checkpoint latest` to continue from
the newest saved Trainer checkpoint.

Train the trace scorer with value-aware numeric embeddings enabled:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/train_trace_scorer.py \
  --train-dataset dataset/english/train_complete.json \
  --validation-dataset dataset/english/validation_complete.json \
  --output-dir checkpoints/english_trace_scorer_qlora_numeric \
  --load-in-4bit \
  --target-modules all-linear \
  --attn-implementation sdpa \
  --use-numeric-embedding
```

Run inference from the saved trace scorer adapter:

```bash
python scripts/infer_trace_scorer.py \
  --dataset-path dataset/english/validation_complete.json \
  --language-name english \
  --adapter-path checkpoints/english_trace_scorer_qlora/final_adapter \
  --output results/Task2_Numerical_claims_English.json \
  --internal-output results/english_trace_scorer_validation_complete_internal.json \
  --load-in-4bit \
  --attn-implementation sdpa \
  --evaluate \
  --eval-output results/english_trace_scorer_validation_complete_eval.json
```

`infer_trace_scorer.py` writes the required submission schema by default:
`query_id`, `Claim`, `Verdict_BoN`, `BoN_Verdict_list`, `Reasoning_traces`, and `score_list`.
Use `--output-format internal` only when you explicitly need the old local-debug schema.

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

Run the prompted baseline in pairwise ranking mode and derive the verdict from the
top-5 ranked traces:

```bash
python scripts/prompt_ranking_baseline.py \
  --dataset-path dataset/english/validation_complete.json \
  --language-name english \
  --output results/prompt_baseline_english_validation_complete_pairwise.json \
  --ranking-mode pairwise \
  --verdict-top-k 5 \
  --evaluate \
  --eval-output results/prompt_baseline_english_validation_complete_pairwise_eval.json
```

Run adapter inference in pairwise ranking mode:

```bash
python scripts/infer_sft_lora.py \
  --dataset-path dataset/english/validation_complete.json \
  --language-name english \
  --adapter-path checkpoints/english_sft_qlora/final_adapter \
  --output results/english_sft_qlora_validation_complete_pairwise_predictions.json \
  --load-in-4bit \
  --ranking-mode pairwise \
  --pairwise-orientation balanced \
  --verdict-top-k 5 \
  --evaluate \
  --eval-output results/english_sft_qlora_validation_complete_pairwise_eval.json
```

Debug pairwise adapter inference on 20 claims with batched comparisons:

```bash
python scripts/infer_sft_lora.py \
  --dataset-path dataset/english/validation_complete.json \
  --language-name english \
  --adapter-path checkpoints/english_sft_qlora/final_adapter \
  --output results/english_sft_qlora_validation_complete_pairwise_20_predictions.json \
  --load-in-4bit \
  --ranking-mode pairwise \
  --pairwise-orientation balanced \
  --limit 20 \
  --inference-batch-size 8 \
  --pairwise-max-new-tokens 16 \
  --verdict-top-k 5 \
  --evaluate \
  --eval-output results/english_sft_qlora_validation_complete_pairwise_20_eval.json
```

Use bidirectional pairwise comparisons to counter A/B position bias. This doubles
the number of pairwise comparisons:

```bash
python scripts/prompt_ranking_baseline.py \
  --dataset-path dataset/english/validation_complete.json \
  --language-name english \
  --output results/prompt_baseline_english_validation_complete_pairwise_bidirectional_20.json \
  --ranking-mode pairwise \
  --pairwise-orientation bidirectional \
  --limit 20 \
  --inference-batch-size 8 \
  --pairwise-max-new-tokens 16 \
  --verdict-top-k 5 \
  --save-debug-fields \
  --evaluate \
  --eval-output results/prompt_baseline_english_validation_complete_pairwise_bidirectional_20_eval.json
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
