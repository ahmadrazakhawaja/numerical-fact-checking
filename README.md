# Numerical Fact-Checking

Research code for ranking reasoning traces and predicting numerical claim verdicts in English, Spanish, and Arabic for CLEF Task 2. The project studies whether a model can select useful reasoning consistently across languages, including translated and mixed-language inputs.

Given a claim, evidence snippets, and candidate reasoning traces, the system ranks the traces and derives a claim-level verdict from the highest-ranked candidates. Supported verdicts in the supplied validation data are **True**, **False**, and **Conflicting**.

The repository includes:

- A trainable binary trace scorer with LoRA or 4-bit QLoRA.
- Multilingual training with an optional score-invariance loss on aligned traces.
- Optional value-aware numeric embeddings and normalized numeric hints.
- Supervised fine-tuning (SFT), direct preference optimization (DPO), and prompted ranking baselines.
- Random baselines, ranking and verdict evaluation, and cross-language consistency analysis.
- Experiment outputs and the accompanying [master's thesis](thesis/README.md).

## How it works

```text
Claim + evidence + candidate reasoning traces
                    │
              Score or rank traces
                    │
            Select top-k trace verdicts
                    │
              Predict claim verdict
                    │
      Evaluate ranking, verdict, and consistency
```

The trace scorer learns whether each candidate's verdict matches the gold claim label. This is a label-agreement training signal: a positive trace is not necessarily a fully correct explanation. Inference uses the supplied candidates and evidence; it does not retrieve new evidence from the web.

## Repository layout

| Path | Contents |
| --- | --- |
| `scripts/` | Training, inference, data preparation, and evaluation entry points |
| `dataset/` | Language-specific JSON splits and translated/mixed variants |
| `dataset_samples/` | Example datasets and prepared DPO artifacts |
| `results/` | Saved predictions and evaluation reports |
| `checkpoints/` | Local training checkpoints and adapters; ignored by Git |
| `thesis/` | LaTeX thesis source, figures, tables, and build instructions |
| `job.sh` | Cluster-specific Slurm experiment commands |
| `pyproject.toml`, `uv.lock` | Python dependencies and locked environment |

## Setup

Use Python **3.11** to match `.python-version` (the project declares Python ≥3.11). The full dependency set targets **Linux with NVIDIA CUDA**, with explicit CUDA, Triton, and bitsandbytes dependencies. It is not a portable macOS/CPU environment as currently declared.

From the repository root on a compatible machine with `uv` installed:

```bash
uv sync --frozen
source .venv/bin/activate
```

All commands below run from the repository root with that environment active. The default base model is `mistralai/Mistral-7B-Instruct-v0.3`; select another compatible model with `--model-id`. Model runs require downloading the base model or having it cached locally. For a model that requires authentication, authenticate with `hf auth login` and obtain any required model access first.

Training and model inference are intended for GPU execution. Memory needs depend on the model, input length, batch size, and quantization. `--load-in-4bit` enables bitsandbytes quantization; bitsandbytes is already declared in the project dependencies.

The random baseline and standalone evaluators use only the Python standard library and can run without installing the model-training environment.

## Data

Scripts consume UTF-8 JSON arrays. A simplified, synthetic input row looks like this:

```json
[
  {
    "claim": "Revenue increased from 100 to 120, a rise of 20%.",
    "evidences": ["Reported revenue was 100 in year one and 120 in year two."],
    "Reasoning_traces": [
      "Label: True. Justification: (120 - 100) / 100 = 20%.",
      "Label: False. Justification: The increase was only 10%."
    ],
    "Verdict_list": ["True", "False"],
    "label": "True"
  }
]
```

- `Reasoning_traces` and `Verdict_list` must correspond in length and order.
- `label` is the gold claim label for training and evaluation. The gold-label loader also accepts aliases such as `Label` and `gold_label`; it does **not** use the dataset's `verdict` field as gold.
- Dataset row indices and trace indices are significant. Preserve their order when preparing aligned translations or comparing prediction files.
- Omit `--evaluate` for test files without gold labels.

The standard layout is `dataset/<language>/{train,validation,test}.json`. Some experiments use `train_complete.json` and `validation_complete.json`, and several test files have release-specific names. Pass `--dataset-path` to inference/evaluation to select a particular file.

**A fresh clone does not contain every training input.** In particular, English `train.json` and `train_complete.json` are ignored by Git, and complete training splits referenced by experiments may need to be supplied locally. Obtain the appropriate task data and place it at the paths used by your command. The training examples below assume those files are available; the quick start uses tracked Spanish data.

## Quick start: random baseline without a GPU

Run with any compatible Python interpreter, even without the full dependency environment:

```bash
python3 scripts/random_ranking_baseline.py \
  --dataset-dir dataset \
  --languages spanish \
  --target-split validation \
  --seed 42 \
  --top-n-for-verdict 5 \
  --output results/readme_random_spanish_predictions.json \
  --evaluate \
  --eval-output results/readme_random_spanish_eval.json
```

This shuffles candidate traces, votes over the top five verdicts, and writes predictions and metrics. The script also reads `dataset/spanish/train.json` for a label-space check.

## Train and use a trace scorer

### Train an English QLoRA adapter

```bash
python scripts/train_trace_scorer.py \
  --train-dataset dataset/english/train_complete.json \
  --validation-dataset dataset/english/validation_complete.json \
  --output-dir checkpoints/english_trace_scorer_qlora \
  --load-in-4bit \
  --target-modules all-linear \
  --attn-implementation sdpa \
  --num-train-epochs 4 \
  --learning-rate 1e-4 \
  --early-stopping-patience 1 \
  --metric-for-best-model macro_f1_recall_at_k_mean
```

The default scorer uses a two-logit cross-entropy head (`--scorer-head ce`); a one-logit binary cross-entropy head is available with `--scorer-head bce`. Inputs include the claim, truncated evidence, trace verdict, and justification.

Training saves the adapter and tokenizer in `<output-dir>/final_adapter`, along with run metadata under the output directory. Early stopping enables best-checkpoint loading and requires matching save/evaluation strategies; both default to `epoch`. Resume an interrupted trace-scorer run with the same output directory and `--resume-from-checkpoint latest`.

For a small training check, add `--limit-train 32 --limit-validation 16 --num-train-epochs 1`. To reduce memory use, adjust `--max-length`, batch sizes, and evidence/trace truncation limits.

### Rank traces and evaluate

```bash
python scripts/infer_trace_scorer.py \
  --dataset-path dataset/english/validation_complete.json \
  --language-name english \
  --adapter-path checkpoints/english_trace_scorer_qlora/final_adapter \
  --output results/english_trace_scorer_predictions.json \
  --internal-output results/english_trace_scorer_internal.json \
  --load-in-4bit \
  --attn-implementation sdpa \
  --verdict-top-k 5 \
  --evaluate \
  --eval-k 5 \
  --eval-output results/english_trace_scorer_eval.json
```

Keep the base model, scorer head, numeric options, and input preprocessing consistent with training. Pass the same `--model-id` at inference if training used a non-default model.

`--verdict-top-k` controls how many ranked trace verdicts participate in the vote. `--eval-k` controls the ranking metric cutoff. Set these explicitly for comparable experiments: trace-scorer training defaults to top-5 voting, while inference defaults to top-1.

The default output uses the submission fields `query_id`, `Claim`, `Verdict_BoN`, `BoN_Verdict_list`, `Reasoning_traces`, and `score_list`. Scores correspond to the original trace order; larger scores rank higher. `query_id` is the zero-based dataset row index. `--internal-output` additionally saves the internal prediction format for analysis; `--output-format internal` selects that format as the primary output.

### Numeric-aware variants

Add `--use-numeric-embedding` to training and inference to enable value-aware numeric embeddings. Keep `numeric_embedding_config.json` and `numeric_value_encoder.pt` with the saved adapter.

The trace scorer also supports `--append-normalized-numbers`, which appends normalized numeric hints as text. Match this setting and `--max-normalized-numbers` between training and inference; their current maximum-count defaults differ.

## Multilingual training and consistency

Train with aligned language versions of the same claims and candidate traces:

```bash
python scripts/train_trace_scorer.py \
  --train-datasets \
    english=dataset/english/train_complete.json \
    spanish=dataset/spanish/train_complete.json \
    arabic=dataset/arabic/train_complete.json \
  --validation-datasets \
    english=dataset/english/validation_complete.json \
    spanish=dataset/spanish/validation_complete.json \
    arabic=dataset/arabic/validation_complete.json \
  --output-dir checkpoints/multilingual_trace_scorer \
  --load-in-4bit \
  --language-invariance-weight 0.2 \
  --language-invariance-warmup-ratio 0.2
```

The invariance loss encourages matching scores across aligned language versions; a weight of `0` disables that penalty. Alignment is positional by claim and trace index, so unrelated language-specific datasets cannot be treated as translations simply by giving them matching filenames.

Run `infer_trace_scorer.py` for each aligned variant, changing `--dataset-path`, `--language-name`, `--output`, and `--internal-output`. Then compare the **internal** prediction files:

```bash
python scripts/evaluate_language_invariance.py \
  --predictions \
    results/english_trace_scorer_internal.json \
    results/spanish_trace_scorer_internal.json \
    results/arabic_trace_scorer_internal.json \
  --names english spanish arabic \
  --top-k 5 \
  --output results/language_invariance_eval.json
```

This reports verdict agreement, Cohen's/Fleiss' kappa, and ranking consistency measures including Kendall's tau, Spearman's rho, and top-k overlap. Predictions are matched by `group_id`, falling back to `dataset_index`. Only compare files whose matched identifiers refer to the same underlying claims and trace candidates.

## Other training and ranking methods

### Prompted baseline

Run the base model without training an adapter:

```bash
python scripts/prompt_ranking_baseline.py \
  --dataset-path dataset/english/validation_complete.json \
  --language-name english \
  --output results/prompt_baseline_predictions.json \
  --ranking-mode pairwise \
  --pairwise-orientation balanced \
  --verdict-top-k 5 \
  --limit 20 \
  --evaluate \
  --eval-output results/prompt_baseline_eval.json
```

Pairwise ranking compares candidate traces in pairs. `--pairwise-orientation bidirectional` evaluates both orders to address position bias, doubling the comparisons. Adapter inference also supports these ranking options.

### SFT followed by DPO

Train an SFT adapter:

```bash
python scripts/train_english_sft_lora.py \
  --train-dataset dataset/english/train_complete.json \
  --validation-dataset dataset/english/validation_complete.json \
  --output-dir checkpoints/english_sft_qlora \
  --load-in-4bit \
  --target-modules all-linear \
  --attn-implementation sdpa
```

Optionally continue with DPO using the SFT adapter:

```bash
python scripts/train_english_dpo_lora.py \
  --train-dataset dataset/english/train_complete.json \
  --validation-dataset dataset/english/validation_complete.json \
  --sft-adapter-path checkpoints/english_sft_qlora/final_adapter \
  --output-dir checkpoints/english_dpo_qlora \
  --load-in-4bit \
  --attn-implementation sdpa
```

The DPO trainer builds preference pairs from the input splits. `prepare_english_dpo_data.py` can separately export deterministic pairs for inspection; its exports are not a prerequisite for the command above.

Use the shared inference entry point for either adapter:

```bash
python scripts/infer_sft_lora.py \
  --dataset-path dataset/english/validation_complete.json \
  --language-name english \
  --adapter-path checkpoints/english_dpo_qlora/final_adapter \
  --output results/english_dpo_predictions.json \
  --load-in-4bit \
  --attn-implementation sdpa \
  --verdict-top-k 5 \
  --evaluate \
  --eval-output results/english_dpo_eval.json
```

To evaluate SFT alone, substitute its `final_adapter` path. Numeric-aware SFT/DPO runs require `--use-numeric-embedding` throughout, including the SFT adapter used to initialize DPO.

## Evaluation

Re-evaluate a saved trace-scorer prediction file independently:

```bash
python3 scripts/evaluate_clef_task2.py \
  --dataset-path dataset/english/validation_complete.json \
  --language-name english \
  --predictions results/english_trace_scorer_predictions.json \
  --k 5 \
  --output results/english_trace_scorer_reeval.json
```

The evaluator accepts submission-style predictions or internal rows containing `language`, `dataset_index`, `ranked_trace_indices`, and `predicted_verdict`. Trace indices are zero-based.

| Metric | Meaning |
| --- | --- |
| Recall@k | Fraction of gold-label-matching traces retrieved in the top k |
| Precision@k | Precision of the top-k trace selection |
| Recall@k upper bound | Maximum achievable recall given k and the number of matching traces |
| MRR@k | Reciprocal rank of the first matching trace, truncated at k |
| Macro/classwise F1 | Claim-level verdict classification quality |
| `macro_f1_recall_at_k_mean` | Arithmetic mean of macro F1 and Recall@k |

Claims with no label-matching traces contribute zero ranking scores. Reports include missing-prediction and invalid-index counts; inspect these alongside the headline metrics. For predictions generated from a subset, pass the corresponding `--start-index` and `--limit` to the evaluator so it evaluates the same rows.

Use identical datasets, row subsets, seeds, voting settings, and ranking cutoffs when comparing runs. The `results/` directory contains outputs from multiple experimental configurations; filenames alone are not enough to establish a controlled comparison.

## Data preparation and experiment utilities

| Script | Purpose |
| --- | --- |
| `filter_zero_positive_claims.py` | Write splits containing only claims with at least one gold-label-matching trace |
| `translate_dataset_googletrans.py` | Translate textual fields while preserving canonical verdict labels; requires network access |
| `create_mixed_validation_complete.py` | Mix aligned trace languages while retaining the target-language claim and evidence |
| `prepare_english_dpo_data.py` | Export preference pairs and preparation summaries |
| `evaluate_language_invariance.py` | Compare verdicts and rankings across aligned prediction variants |

Filtering out zero-positive claims changes the evaluation population. Report filtered and unfiltered results separately, and preserve alignment if filtering multilingual variants.

Use each script's `--help` for its supported arguments. Training/inference help commands require the model dependencies to be installed.

### Logging and cluster runs

Training scripts support Weights & Biases via `--report-to wandb`, with optional `--run-name`, `--wandb-project`, `--wandb-group`, and `--wandb-tags`. Use `--wandb-mode offline` when outbound connectivity is unavailable and sync the run later. Local metadata is saved with training outputs.

`job.sh` records Slurm experiments with site-specific resource requests, paths, modules, and authentication setup. Adapt it to your cluster before use, and supply your own authentication through the environment or Hugging Face login. For multi-GPU trace-scorer training, the scripts support launching through `torchrun`.

## Thesis

See [thesis/README.md](thesis/README.md) for the research document and build requirements. With the required TeX tools and Inkscape installed:

```bash
make -C thesis
```
