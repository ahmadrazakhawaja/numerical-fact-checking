#!/bin/bash
#SBATCH -J numerical-fact-checking
#SBATCH -A ag_gipp
#SBATCH -p scc-gpu
#SBATCH -G A100:4
#SBATCH -C inet
#SBATCH --time=02:00:00
#SBATCH -o gpu_test.out
#SBATCH -e gpu_test.err

set -euo pipefail

export HF_HOME="$HOME/numerical-fact-checking/hf"
# export HF_HOME="$PROJECT/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

set -e

echo "Node: $(hostname)"
echo "Starting at: $(date)"

# Move to repo (already in HOME)
cd $HOME/numerical-fact-checking

module purge
module load gcc/13.2.0-nvptx
module load python/3.11.9
module load uv

# Ensure venv exists (uv will reuse it)
[ -d ".venv" ] || uv venv

# Install deps every run (recommended).
# Use --frozen if you have uv.lock and want strict reproducibility.
uv sync --frozen

uv run scripts/train_mini_verifier_experiments.py \
  --dataset-path dataset/english/train.json \
  --num-claims 5 \
  --modes cls_unfrozen \
  --adapter-id IlyaGusev/saiga_mistral_7b_lora \
  --device cuda \
  --dtype float32 \
  --optimizer-cls adamw \
  --batch-size 1 \
  --output-root checkpoints/mini_factcheck

# python test_torch.py
uv run scripts/infer_mini_verifier_experiments.py \
  --dataset-path dataset/english/train.json \
  --num-claims 5 \
  --checkpoint-root checkpoints/mini_factcheck \
  --modes ntp cls_frozen cls_unfrozen \
  --output results/mini_factcheck_inference.json

echo "Finished at: $(date)"