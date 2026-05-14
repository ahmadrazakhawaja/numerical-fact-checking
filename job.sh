#!/bin/bash
#SBATCH -J numerical-fact-checking
#SBATCH -A ag_gipp
#SBATCH -p scc-gpu
#SBATCH -G A100:2
#SBATCH -C inet
#SBATCH --time=48:00:00
#SBATCH -o gpu_test_inv_normal2.out
#SBATCH -e gpu_test_inv_normal2.err
#SBATCH -N 1

set -euo pipefail

export HF_TOKEN="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"
export HF_HOME=/scratch-scc/projects/ag_gipp/$USER/hf
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

uv run hf auth login --token $HF_TOKEN

##################################### lang inv loss normal

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run torchrun --standalone --nproc_per_node=2 scripts/train_trace_scorer.py \
  --train-datasets \
    english=dataset/english/train_complete.json \
    spanish=dataset/spanish/train_complete.json \
    arabic=dataset/arabic/train_complete.json \
  --validation-datasets \
    english=dataset/english/validation_complete.json \
    spanish=dataset/spanish/validation_complete.json \
    arabic=dataset/arabic/validation_complete.json \
  --output-dir /scratch-scc/projects/ag_gipp/$USER/checkpoints/multilingual_trace_scorer_invariance \
  --language-invariance-weight 0.1 \
  --language-invariance-warmup-ratio 0 \
  --load-in-4bit \
  --target-modules all-linear \
  --attn-implementation sdpa \
  --device-map none \
  --per-device-train-batch-size 1 \
  --save-best-adapter-at-each-eval \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --metric-for-best-model macro_f1_recall_at_k_mean \
  --num-train-epochs 10 \
  --learning-rate 2e-4 \
  --warmup-ratio 0.08 \
  --logging-steps 10 \
  --eval-strategy epoch \
  --save-strategy epoch \
  --max-evidence-items 5 \
  --max-evidence-chars 512 \
  --max-length 1024 \
  --max-trace-chars 800 \
  --report-to wandb \
  --resume-from-checkpoint latest \
  --wandb-entity ahmadrazakhawaja-g-ttingen-campus \
  --wandb-project numerical-fact-checking \
  --wandb-group multilingual-trace-scorer-invariance \
  --wandb-tags multilingual qlora trace-scorer invariance lambda0p1 warmup0p2 task2metric no_num_norm lr5e-5 \
  --run-name multi-trace-scorer-invariance-lambda0p1-lr5e-5-no_num_norm \
  --no-ddp-find-unused-parameters


##################################### lang inv loss regularization

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# uv run torchrun --standalone --nproc_per_node=2 scripts/train_trace_scorer.py \
#   --train-datasets \
#     english=dataset/english/train_complete.json \
#     spanish=dataset/spanish/train_complete.json \
#     arabic=dataset/arabic/train_complete.json \
#   --validation-datasets \
#     english=dataset/english/validation_complete.json \
#     spanish=dataset/spanish/validation_complete.json \
#     arabic=dataset/arabic/validation_complete.json \
#   --output-dir /scratch-scc/projects/ag_gipp/$USER/checkpoints/multilingual_trace_scorer_invariance_regularization \
#   --language-invariance-weight 0.1 \
#   --language-invariance-warmup-ratio 0 \
#   --load-in-4bit \
#   --target-modules all-linear \
#   --attn-implementation sdpa \
#   --device-map none \
#   --per-device-train-batch-size 1 \
#   --save-best-adapter-at-each-eval \
#   --per-device-eval-batch-size 1 \
#   --gradient-accumulation-steps 16 \
#   --metric-for-best-model macro_f1_recall_at_k_mean \
#   --num-train-epochs 10 \
#   --learning-rate 2e-4 \
#   --warmup-ratio 0.08 \
#   --weight-decay 0.01 \
#   --lora-dropout 0.10 \
#   --logging-steps 10 \
#   --eval-strategy epoch \
#   --save-strategy epoch \
#   --max-evidence-items 5 \
#   --max-evidence-chars 512 \
#   --max-length 1024 \
#   --max-trace-chars 800 \
#   --report-to wandb \
#   --wandb-entity ahmadrazakhawaja-g-ttingen-campus \
#   --wandb-project numerical-fact-checking \
#   --wandb-group multilingual-trace-scorer-invariance-regularization \
#   --wandb-tags multilingual qlora trace-scorer invariance lambda0p1 warmup0p2 task2metric regularized no_num_norm lr5e-5 \
#   --run-name multi-trace-scorer-invariance-lambda0p1-lr5e-5-no_num_norm-regularized \
#   --no-ddp-find-unused-parameters


##################################### lang inv loss Numerical Norm

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# uv run torchrun --standalone --nproc_per_node=2 scripts/train_trace_scorer.py \
#   --train-datasets \
#     english=dataset/english/train_complete.json \
#     spanish=dataset/spanish/train_complete.json \
#     arabic=dataset/arabic/train_complete.json \
#   --validation-datasets \
#     english=dataset/english/validation_complete.json \
#     spanish=dataset/spanish/validation_complete.json \
#     arabic=dataset/arabic/validation_complete.json \
#   --output-dir /scratch-scc/projects/ag_gipp/$USER/checkpoints/multilingual_trace_scorer_invariance_num_norm \
#   --language-invariance-weight 0.1 \
#   --language-invariance-warmup-ratio 0 \
#   --load-in-4bit \
#   --target-modules all-linear \
#   --attn-implementation sdpa \
#   --device-map none \
#   --append-normalized-numbers \
#   --per-device-train-batch-size 1 \
#   --save-best-adapter-at-each-eval \
#   --per-device-eval-batch-size 1 \
#   --gradient-accumulation-steps 16 \
#   --metric-for-best-model macro_f1_recall_at_k_mean \
#   --num-train-epochs 10 \
#   --learning-rate 2e-4 \
#   --warmup-ratio 0.08 \
#   --logging-steps 10 \
#   --eval-strategy epoch \
#   --save-strategy epoch \
#   --max-evidence-items 5 \
#   --max-evidence-chars 512 \
#   --max-length 1024 \
#   --max-trace-chars 800 \
#   --report-to wandb \
#   --wandb-entity ahmadrazakhawaja-g-ttingen-campus \
#   --wandb-project numerical-fact-checking \
#   --wandb-group multilingual-trace-scorer-invariance-regularization \
#   --wandb-tags multilingual qlora trace-scorer invariance lambda0p1 warmup0p2 task2metric num_norm lr5e-5 \
#   --run-name multi-trace-scorer-invariance-lambda0p1-lr5e-5-num_norm \
#   --no-ddp-find-unused-parameters


##################################### lang inv loss regularization Numerical Norm

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# uv run torchrun --standalone --nproc_per_node=2 scripts/train_trace_scorer.py \
#   --train-datasets \
#     english=dataset/english/train_complete.json \
#     spanish=dataset/spanish/train_complete.json \
#     arabic=dataset/arabic/train_complete.json \
#   --validation-datasets \
#     english=dataset/english/validation_complete.json \
#     spanish=dataset/spanish/validation_complete.json \
#     arabic=dataset/arabic/validation_complete.json \
#   --output-dir /scratch-scc/projects/ag_gipp/$USER/checkpoints/multilingual_trace_scorer_invariance_regularization_num_norm \
#   --language-invariance-weight 0.1 \
#   --language-invariance-warmup-ratio 0 \
#   --load-in-4bit \
#   --target-modules all-linear \
#   --attn-implementation sdpa \
#   --device-map none \
#   --append-normalized-numbers \
#   --per-device-train-batch-size 1 \
#   --save-best-adapter-at-each-eval \
#   --per-device-eval-batch-size 1 \
#   --gradient-accumulation-steps 16 \
#   --metric-for-best-model macro_f1_recall_at_k_mean \
#   --num-train-epochs 10 \
#   --learning-rate 2e-4 \
#   --warmup-ratio 0.08 \
#   --weight-decay 0.01 \
#   --lora-dropout 0.10 \
#   --logging-steps 10 \
#   --eval-strategy epoch \
#   --save-strategy epoch \
#   --max-evidence-items 5 \
#   --max-evidence-chars 512 \
#   --max-length 1024 \
#   --max-trace-chars 800 \
#   --report-to wandb \
#   --wandb-entity ahmadrazakhawaja-g-ttingen-campus \
#   --wandb-project numerical-fact-checking \
#   --wandb-group multilingual-trace-scorer-invariance-regularization \
#   --wandb-tags multilingual qlora trace-scorer invariance lambda0p1 warmup0p2 task2metric regularized num_norm lr5e-5 \
#   --run-name multi-trace-scorer-invariance-lambda0p1-lr5e-5-num_norm-regularized \
#   --no-ddp-find-unused-parameters


######################################

# export CUDA_VISIBLE_DEVICES=0
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ADAPTER_PATH="/scratch-scc/projects/ag_gipp/$USER/checkpoints/english_dpo_qlora/final_adapter"

# for LANG in english spanish arabic; do
#   uv run scripts/infer_sft_lora.py \
#     --dataset-path "dataset/${LANG}/validation.json" \
#     --language-name "${LANG}" \
#     --adapter-path "${ADAPTER_PATH}" \
#     --output "results/english_dpo_qlora_${LANG}_validation_complete_predictions_top5.json" \
#     --load-in-4bit \
#     --device-map auto \
#     --attn-implementation sdpa \
#     --derive-verdict-from-ranking \
#     --verdict-top-k 5 \
#     --evaluate \
#     --eval-output "results/english_dpo_qlora_${LANG}_validation_complete_eval_top5.json"
# done

########################################### BCE scorer

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# uv run torchrun --standalone --nproc_per_node=2 scripts/train_trace_scorer.py \
#   --model-id meta-llama/Llama-3.1-8B-Instruct \
#   --train-dataset dataset/train_multi.json \
#   --validation-dataset dataset/validation_multi.json \
#   --output-dir /scratch-scc/projects/ag_gipp/$USER/checkpoints/multi_trace_scorer_qlora_bce \
#   --load-in-4bit \
#   --scorer-head bce \
#   --lr-scheduler-type cosine \
#   --target-modules q_proj,k_proj,v_proj \
#   --attn-implementation sdpa \
#   --per-device-train-batch-size 8 \
#   --per-device-eval-batch-size 8 \
#   --num-train-epochs 10 \
#   --save-best-adapter-at-each-eval \
#   --gradient-accumulation-steps 2 \
#   --learning-rate 1e-4 \
#   --warmup-ratio 0.08 \
#   --weight-decay 0.01 \
#   --lora-dropout 0.10 \
#   --logging-steps 25 \
#   --metric-for-best-model macro_f1_recall_at_k_mean \
#   --max-evidence-items 20 \
#   --max-evidence-chars 3078 \
#   --max-length 15000 \
#   --greater-is-better \
#   --eval-strategy epoch \
#   --save-strategy epoch \
#   --save-total-limit 3 \
#   --report-to wandb \
#   --resume-from-checkpoint latest \
#   --wandb-entity ahmadrazakhawaja-g-ttingen-campus \
#   --wandb-project numerical-fact-checking \
#   --wandb-group multilingual-trace-scorer \
#   --wandb-tags multilingual qlora trace-scorer early-stopping task2metric bce lr1e-4 \
#   --run-name multi-trace-scorer-task2metric-lr1e-4-bce \
#   --no-ddp-find-unused-parameters

# CUDA_VISIBLE_DEVICES=0 \
# uv run scripts/infer_trace_scorer.py \
#   --dataset-path dataset/english/validation.json \
#   --model-id meta-llama/Llama-3.1-8B-Instruct \
#   --language-name english \
#   --adapter-path /scratch-scc/projects/ag_gipp/u26587/checkpoints/multi_trace_scorer_qlora_bce/final_adapter \
#   --output results/multi_trace_scorer_llama_qlora_bce_english_validation_predictions_top5_2.json \
#   --supervisor-output results/Task2_Numerical_claims_English_multilingual_top5.json \
#   --load-in-4bit \
#   --scorer-head bce \
#   --attn-implementation sdpa \
#   --max-evidence-items 20 \
#   --max-evidence-chars 3078 \
#   --max-length 15000 \
#   --scoring-batch-size 8 \
#   --verdict-top-k 5 \
#   --eval-k 5 \
#   --evaluate \
#   --eval-output results/multi_trace_scorer_llama_qlora_bce_english_validation_eval_top5_2.json


# CUDA_VISIBLE_DEVICES=0 \
# CUDA_VISIBLE_DEVICES=0 \
# uv run scripts/infer_trace_scorer.py \
#   --dataset-path dataset/spanish/validation.json \
#   --model-id meta-llama/Llama-3.1-8B-Instruct \
#   --language-name spanish \
#   --adapter-path /scratch-scc/projects/ag_gipp/u26587/checkpoints/multi_trace_scorer_qlora_bce/final_adapter \
#   --output results/multi_trace_scorer_llama_qlora_bce_spanish_validation_predictions_top5_2.json \
#   --supervisor-output results/Task2_Numerical_claims_Spanish_multilingual_top5.json \
#   --load-in-4bit \
#   --scorer-head bce \
#   --attn-implementation sdpa \
#   --max-evidence-items 20 \
#   --max-evidence-chars 3078 \
#   --max-length 15000 \
#   --scoring-batch-size 8 \
#   --verdict-top-k 5 \
#   --eval-k 5 \
#   --evaluate \
#   --eval-output results/multi_trace_scorer_llama_qlora_bce_spanish_validation_eval_top5_2.json


# CUDA_VISIBLE_DEVICES=0 \
# uv run scripts/infer_trace_scorer.py \
#   --dataset-path dataset/arabic/validation.json \
#   --model-id meta-llama/Llama-3.1-8B-Instruct \
#   --language-name arabic \
#   --adapter-path /scratch-scc/projects/ag_gipp/u26587/checkpoints/multi_trace_scorer_qlora_bce/final_adapter \
#   --output results/multi_trace_scorer_llama_qlora_bce_arabic_validation_predictions_top5_2.json \
#   --supervisor-output results/Task2_Numerical_claims_Arabic_multilingual_top5.json \
#   --load-in-4bit \
#   --scorer-head bce \
#   --attn-implementation sdpa \
#   --max-evidence-items 20 \
#   --max-evidence-chars 3078 \
#   --max-length 15000 \
#   --scoring-batch-size 8 \
#   --verdict-top-k 5 \
#   --eval-k 5 \
#   --evaluate \
#   --eval-output results/multi_trace_scorer_llama_qlora_bce_arabic_validation_eval_top5_2.json


########################################### no num norm


# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# uv run torchrun --standalone --nproc_per_node=2 scripts/train_trace_scorer.py \
#   --model-id meta-llama/Llama-3.1-8B-Instruct \
#   --train-dataset dataset/train_multi.json \
#   --validation-dataset dataset/validation_multi.json \
#   --output-dir /scratch-scc/projects/ag_gipp/$USER/checkpoints/multi_trace_scorer_qlora_task2metric_no_num_norm \
#   --load-in-4bit \
#   --target-modules all-linear \
#   --attn-implementation sdpa \
#   --per-device-train-batch-size 8 \
#   --per-device-eval-batch-size 8 \
#   --num-train-epochs 10 \
#   --save-best-adapter-at-each-eval \
#   --gradient-accumulation-steps 2 \
#   --learning-rate 5e-5 \
#   --warmup-ratio 0.08 \
#   --weight-decay 0.01 \
#   --lora-dropout 0.10 \
#   --logging-steps 25 \
#   --metric-for-best-model macro_f1_recall_at_k_mean \
#   --max-evidence-items 20 \
#   --max-evidence-chars 3078 \
#   --max-length 15000 \
#   --greater-is-better \
#   --eval-strategy epoch \
#   --save-strategy epoch \
#   --save-total-limit 3 \
#   --report-to wandb \
#   --resume-from-checkpoint latest \
#   --wandb-entity ahmadrazakhawaja-g-ttingen-campus \
#   --wandb-project numerical-fact-checking \
#   --wandb-group multilingual-trace-scorer \
#   --wandb-tags multilingual qlora trace-scorer early-stopping task2metric no_num_norm lr5e-5 \
#   --run-name multi-trace-scorer-task2metric-lr5e-5-no_num_norm \
#   --no-ddp-find-unused-parameters

# CUDA_VISIBLE_DEVICES=0 \
# uv run scripts/infer_trace_scorer.py \
#   --dataset-path dataset/english/validation.json \
#   --model-id meta-llama/Llama-3.1-8B-Instruct \
#   --language-name english \
#   --adapter-path /scratch-scc/projects/ag_gipp/u26587/checkpoints/multi_trace_scorer_qlora_task2metric_no_num_norm/final_adapter \
#   --output results/multi_trace_scorer_llama_qlora_task2metric_no_num_norm_english_validation_predictions_top5_2.json \
#   --supervisor-output results/Task2_Numerical_claims_English_multilingual_top5.json \
#   --load-in-4bit \
#   --attn-implementation sdpa \
#   --scoring-batch-size 8 \
#    --max-evidence-items 20 \
#   --max-evidence-chars 3078 \
#   --max-length 15000 \
#   --verdict-top-k 5 \
#   --evaluate \
#   --eval-output results/multi_trace_scorer_llama_qlora_task2metric_no_num_norm_english_validation_eval_top5_2.json

# CUDA_VISIBLE_DEVICES=0 \
# uv run scripts/infer_trace_scorer.py \
#   --dataset-path dataset/spanish/validation.json \
#   --model-id meta-llama/Llama-3.1-8B-Instruct \
#   --language-name spanish \
#   --adapter-path /scratch-scc/projects/ag_gipp/$USER/checkpoints/multi_trace_scorer_qlora_task2metric_no_num_norm/final_adapter \
#   --output results/multi_trace_scorer_llama_qlora_task2metric_no_num_norm_spanish_validation_predictions_top5_2.json \
#   --supervisor-output results/Task2_Numerical_claims_Spanish_multilingual_top5.json \
#   --load-in-4bit \
#   --attn-implementation sdpa \
#    --max-evidence-items 20 \
#   --max-evidence-chars 3078 \
#   --max-length 15000 \
#   --scoring-batch-size 8 \
#   --verdict-top-k 5 \
#   --evaluate \
#   --eval-output results/multi_trace_scorer_llama_qlora_task2metric_no_num_norm_spanish_validation_eval_top5_2.json

# CUDA_VISIBLE_DEVICES=0 \
# uv run scripts/infer_trace_scorer.py \
#   --dataset-path dataset/arabic/clef_2026_final_arabic_test.json \
#   --model-id meta-llama/Llama-3.1-8B-Instruct \
#   --language-name arabic \
#   --adapter-path /scratch-scc/projects/ag_gipp/$USER/checkpoints/multi_trace_scorer_qlora_task2metric_no_num_norm/final_adapter \
#   --output results/multi_trace_scorer_llama_qlora_task2metric_no_num_norm_arabic_test_predictions_top5.json \
#   --supervisor-output results/Task2_Numerical_claims_Arabic_multilingual_top5_test2.json \
#   --load-in-4bit \
#   --attn-implementation sdpa \
#   --scoring-batch-size 8 \
#   --max-evidence-items 20 \
#   --max-evidence-chars 3078 \
#   --max-length 15000 \
#   --verdict-top-k 5 \



########################################### num norm


# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# uv run torchrun --standalone --nproc_per_node=2 scripts/train_trace_scorer.py \
#   --model-id meta-llama/Llama-3.1-8B-Instruct \
#   --train-dataset dataset/train_multi.json \
#   --validation-dataset dataset/validation_multi.json \
#   --output-dir /scratch-scc/projects/ag_gipp/$USER/checkpoints/multi_trace_scorer_qlora_task2metric_num_norm \
#   --load-in-4bit \
#   --target-modules all-linear \
#   --attn-implementation sdpa \
#   --append-normalized-numbers \
#   --per-device-train-batch-size 8 \
#   --per-device-eval-batch-size 8 \
#   --gradient-accumulation-steps 2 \
#   --num-train-epochs 10 \
#   --save-best-adapter-at-each-eval \
#   --learning-rate 5e-5 \
#   --warmup-ratio 0.08 \
#   --weight-decay 0.01 \
#   --lora-dropout 0.10 \
#   --logging-steps 25 \
#   --metric-for-best-model macro_f1_recall_at_k_mean \
#   --max-evidence-items 20 \
#   --max-evidence-chars 3078 \
#   --max-length 15000 \
#   --greater-is-better \
#   --eval-strategy epoch \
#   --save-strategy epoch \
#   --save-total-limit 3 \
#   --report-to wandb \
#   --resume-from-checkpoint latest \
#   --wandb-entity ahmadrazakhawaja-g-ttingen-campus \
#   --wandb-project numerical-fact-checking \
#   --wandb-group multilingual-trace-scorer \
#   --wandb-tags multilingual qlora trace-scorer early-stopping task2metric num_norm lr5e-5 \
#   --run-name multi-trace-scorer-task2metric-lr5e-5-num_norm \
#   --no-ddp-find-unused-parameters

# CUDA_VISIBLE_DEVICES=0 \
# uv run scripts/infer_trace_scorer.py \
#   --dataset-path dataset/english/clef_2026_final_english_test.json \
#   --model-id meta-llama/Llama-3.1-8B-Instruct \
#   --language-name english \
#   --adapter-path /scratch-scc/projects/ag_gipp/u26587/checkpoints/multi_trace_scorer_qlora_task2metric_num_norm/final_adapter \
#   --output results/multi_trace_scorer_llama_qlora_task2metric_num_norm_english_test_predictions_top5.json \
#   --supervisor-output results/Task2_Numerical_claims_English_multilingual_top5_test2.json \
#   --load-in-4bit \
#   --attn-implementation sdpa \
#   --scoring-batch-size 8 \
#   --verdict-top-k 5 \
#   --max-evidence-items 20 \
#   --max-evidence-chars 3078 \
#   --max-length 15000 \
#   --verdict-top-k 5 \


# CUDA_VISIBLE_DEVICES=0 \
# uv run scripts/infer_trace_scorer.py \
#   --dataset-path dataset/spanish/validation.json \
#   --model-id meta-llama/Llama-3.1-8B-Instruct \
#   --language-name spanish \
#   --adapter-path /scratch-scc/projects/ag_gipp/$USER/checkpoints/multi_trace_scorer_qlora_task2metric_num_norm/final_adapter \
#   --output results/multi_trace_scorer_llama_qlora_task2metric_num_norm_spanish_validation_predictions_top5.json \
#   --supervisor-output results/Task2_Numerical_claims_Spanish_multilingual_top5.json \
#   --load-in-4bit \
#   --attn-implementation sdpa \
#   --scoring-batch-size 8 \
#   --verdict-top-k 5 \
#   --evaluate \
#   --eval-output results/multi_trace_scorer_llama_qlora_task2metric_num_norm_spanish_validation_eval_top5.json

# CUDA_VISIBLE_DEVICES=0 \
# uv run scripts/infer_trace_scorer.py \
#   --dataset-path dataset/arabic/validation.json \
#   --model-id meta-llama/Llama-3.1-8B-Instruct \
#   --language-name arabic \
#   --adapter-path /scratch-scc/projects/ag_gipp/$USER/checkpoints/multi_trace_scorer_qlora_task2metric_num_norm/final_adapter \
#   --output results/multi_trace_scorer_llama_qlora_task2metric_num_norm_arabic_validation_predictions_top5.json \
#   --supervisor-output results/Task2_Numerical_claims_Arabic_multilingual_top5.json \
#   --load-in-4bit \
#   --attn-implementation sdpa \
#   --scoring-batch-size 8 \
#   --verdict-top-k 5 \
#   --evaluate \
#   --eval-output results/multi_trace_scorer_llama_qlora_task2metric_num_norm_arabic_validation_eval_top5.json


###########################################

# CUDA_VISIBLE_DEVICES=0 \
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# uv run scripts/train_trace_scorer.py \
#   --train-dataset dataset/train_multi.json \
#   --validation-dataset dataset/validation_multi.json \
#   --output-dir /scratch-scc/projects/ag_gipp/$USER/checkpoints/multi_trace_scorer_qlora_en_val \
#   --load-in-4bit \
#   --target-modules all-linear \
#   --attn-implementation sdpa \
#   --per-device-train-batch-size 8 \
#   --per-device-eval-batch-size 8 \
#   --gradient-accumulation-steps 2

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# uv run torchrun --standalone --nproc_per_node=2 scripts/train_trace_scorer.py \
#   --model-id meta-llama/Llama-3.1-8B-Instruct \
#   --train-dataset dataset/train_multi.json \
#   --validation-dataset dataset/validation_multi.json \
#   --output-dir /scratch-scc/projects/ag_gipp/$USER/checkpoints/multi_trace_scorer_llama_qlora_es2_lr1e4_wandb \
#   --load-in-4bit \
#   --target-modules all-linear \
#   --attn-implementation sdpa \
#   --per-device-train-batch-size 8 \
#   --per-device-eval-batch-size 8 \
#   --gradient-accumulation-steps 1 \
#   --num-train-epochs 2 \
#   --learning-rate 1e-4 \
#   --warmup-ratio 0.03 \
#   --logging-steps 25 \
#   --early-stopping-patience 2 \
#   --metric-for-best-model positive_f1 \
#   --max-evidence-items 20 \
#   --max-evidence-chars 3078 \
#   --greater-is-better \
#   --max-length 15000 \
#   --eval-strategy epoch \
#   --save-strategy epoch \
#   --save-total-limit 3 \
#   --report-to wandb \
#   --wandb-entity ahmadrazakhawaja-g-ttingen-campus \
#   --wandb-project numerical-fact-checking \
#   --wandb-group multilingual-trace-scorer \
#   --wandb-tags multilingual qlora llama trace-scorer early-stopping lr1e-4 \
#   --run-name multi-trace-scorer-llama-es2-lr1e4-4gpu \
#   --no-ddp-find-unused-parameters

# CUDA_VISIBLE_DEVICES=0 \
# uv run scripts/train_trace_scorer.py \
#   --model-id meta-llama/Llama-3.1-8B-Instruct \
#   --train-dataset dataset/train_multi.json \
#   --validation-dataset dataset/validation_multi.json \
#   --output-dir /scratch-scc/projects/ag_gipp/$USER/checkpoints/multi_trace_scorer_llama_qlora_noevidence_es2_lr1e4_wandb \
#   --load-in-4bit \
#   --target-modules all-linear \
#   --attn-implementation sdpa \
#   --per-device-train-batch-size 8 \
#   --per-device-eval-batch-size 8 \
#   --gradient-accumulation-steps 1 \
#   --num-train-epochs 2 \
#   --learning-rate 2e-4 \
#   --warmup-ratio 0.03 \
#   --logging-steps 25 \
#   --early-stopping-patience 2 \
#   --metric-for-best-model positive_f1 \
#   --max-evidence-items 3 \
#   --max-evidence-chars 512 \
#   --max-length 5000 \
#   --greater-is-better \
#   --eval-strategy epoch \
#   --save-strategy epoch \
#   --save-total-limit 3 \
#   --report-to wandb \
#   --wandb-entity ahmadrazakhawaja-g-ttingen-campus \
#   --wandb-project numerical-fact-checking \
#   --wandb-group multilingual-trace-scorer \
#   --wandb-tags multilingual qlora llama trace-scorer noevidence early-stopping lr1e-4 \
#   --run-name multi-trace-scorer-llama-noevidence-es2-lr1e4-4gpu \
#   --no-ddp-find-unused-parameters

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# uv run torchrun --standalone --nproc_per_node=2 scripts/train_trace_scorer.py \
#   --train-dataset dataset/train_multi.json \
#   --validation-dataset dataset/validation_multi.json \
#   --output-dir /scratch-scc/projects/ag_gipp/$USER/checkpoints/multi_trace_scorer_qlora_es2_lr1e4_wandb \
#   --load-in-4bit \
#   --target-modules all-linear \
#   --attn-implementation sdpa \
#   --per-device-train-batch-size 8 \
#   --per-device-eval-batch-size 8 \
#   --gradient-accumulation-steps 1 \
#   --num-train-epochs 2 \
#   --learning-rate 1e-4 \
#   --warmup-ratio 0.03 \
#   --logging-steps 25 \
#   --early-stopping-patience 2 \
#   --metric-for-best-model positive_f1 \
#   --max-evidence-items 20 \
#   --max-evidence-chars 3078 \
#   --max-length 15000 \
#   --greater-is-better \
#   --eval-strategy epoch \
#   --save-strategy epoch \
#   --save-total-limit 3 \
#   --report-to wandb \
#   --wandb-entity ahmadrazakhawaja-g-ttingen-campus \
#   --wandb-project numerical-fact-checking \
#   --wandb-group multilingual-trace-scorer \
#   --wandb-tags multilingual qlora trace-scorer early-stopping lr1e-4 \
#   --run-name multi-trace-scorer-es2-lr1e4-4gpu \
#   --no-ddp-find-unused-parameters

# CUDA_VISIBLE_DEVICES=0 \
# uv run scripts/train_trace_scorer.py \
#   --train-dataset dataset/train_multi.json \
#   --validation-dataset dataset/validation_multi.json \
#   --output-dir /scratch-scc/projects/ag_gipp/$USER/checkpoints/multi_trace_scorer_qlora_es2_lr1e4_wandb \
#   --load-in-4bit \
#   --target-modules all-linear \
#   --attn-implementation sdpa \
#   --per-device-train-batch-size 8 \
#   --per-device-eval-batch-size 8 \
#   --gradient-accumulation-steps 1 \
#   --num-train-epochs 2 \
#   --learning-rate 1e-4 \
#   --warmup-ratio 0.03 \
#   --logging-steps 25 \
#   --early-stopping-patience 2 \
#   --metric-for-best-model positive_f1 \
#   --max-evidence-items 3 \
#   --max-evidence-chars 512 \
#   --greater-is-better \
#   --eval-strategy epoch \
#   --save-strategy epoch \
#   --save-total-limit 3 \
#   --report-to wandb \
#   --resume-from-checkpoint latest \
#   --wandb-entity ahmadrazakhawaja-g-ttingen-campus \
#   --wandb-project numerical-fact-checking \
#   --wandb-group multilingual-trace-scorer \
#   --wandb-tags multilingual qlora trace-scorer early-stopping lr1e-4 \
#   --run-name multi-trace-scorer-es2-lr1e4-4gpu \
#   --no-ddp-find-unused-parameters

# CUDA_VISIBLE_DEVICES=0 \
# uv run scripts/infer_trace_scorer.py \
#   --dataset-path dataset/arabic/clef_2026_final_arabic_test.json \
#   --model-id meta-llama/Llama-3.1-8B-Instruct \
#   --language-name arabic \
#   --adapter-path /scratch-scc/projects/ag_gipp/$USER/checkpoints/multi_trace_scorer_llama_qlora_es2_lr1e4_wandb/final_adapter \
#   --output results/multi_trace_scorer_llama_qlora_arabic_test_predictions_top5.json \
#   --supervisor-output results/Task2_Numerical_claims_Arabic_multilingual_top5_test.json \
#   --load-in-4bit \
#   --max-evidence-items 20 \
#   --max-evidence-chars 3078 \
#   --max-length 15000 \
#   --attn-implementation sdpa \
#   --scoring-batch-size 8 \
#   --verdict-top-k 5 \

# CUDA_VISIBLE_DEVICES=0 \
# uv run scripts/infer_trace_scorer.py \
#   --dataset-path dataset/spanish/clef_spanish_test_final.json \
#   --model-id meta-llama/Llama-3.1-8B-Instruct \
#   --language-name spanish \
#   --adapter-path /scratch-scc/projects/ag_gipp/$USER/checkpoints/multi_trace_scorer_llama_qlora_es2_lr1e4_wandb/final_adapter \
#   --output results/multi_trace_scorer_llama_qlora_spanish_test_predictions_top5.json \
#   --supervisor-output results/Task2_Numerical_claims_Spanish_multilingual_top5_test.json \
#   --load-in-4bit \
#   --max-evidence-items 20 \
#   --max-evidence-chars 3078 \
#   --max-length 15000 \
#   --attn-implementation sdpa \
#   --scoring-batch-size 8 \
#   --verdict-top-k 5 \

# CUDA_VISIBLE_DEVICES=0 \
# uv run scripts/infer_trace_scorer.py \
#   --dataset-path dataset/english/clef_2026_final_english_test.json \
#   --model-id meta-llama/Llama-3.1-8B-Instruct \
#   --language-name english \
#   --adapter-path /scratch-scc/projects/ag_gipp/$USER/checkpoints/multi_trace_scorer_llama_qlora_es2_lr1e4_wandb/final_adapter \
#   --output results/multi_trace_scorer_llama_qlora_english_test_predictions_top5.json \
#   --supervisor-output results/Task2_Numerical_claims_English_multilingual_top5_test.json \
#   --load-in-4bit \
#   --max-evidence-items 20 \
#   --max-evidence-chars 3078 \
#   --max-length 15000 \
#   --attn-implementation sdpa \
#   --scoring-batch-size 8 \
#   --verdict-top-k 5 \


# CUDA_VISIBLE_DEVICES=0 \
# uv run scripts/infer_trace_scorer.py \
#   --dataset-path dataset/english/validation.json \
#   --language-name english \
#   --adapter-path /scratch-scc/projects/ag_gipp/$USER/checkpoints/multi_trace_scorer_qlora_en_val/final_adapter \
#   --output results/multi_trace_scorer_qlora_en_validation_predictions_top5.json \
#   --supervisor-output results/Task2_Numerical_claims_English_multilingual_top5.json \
#   --load-in-4bit \
#   --attn-implementation sdpa \
#   --scoring-batch-size 8 \
#   --verdict-top-k 5 \
#   --evaluate \
#   --eval-output results/multi_trace_scorer_qlora_en_validation_eval_top5.json

# CUDA_VISIBLE_DEVICES=0 \
# uv run scripts/infer_trace_scorer.py \
#   --dataset-path dataset/spanish/validation.json \
#   --language-name spanish \
#   --adapter-path /scratch-scc/projects/ag_gipp/$USER/checkpoints/multi_trace_scorer_qlora_en_val/final_adapter \
#   --output results/multi_trace_scorer_qlora_spanish_validation_predictions_top5.json \
#   --supervisor-output results/Task2_Numerical_claims_Spanish_multilingual_top5.json \
#   --load-in-4bit \
#   --attn-implementation sdpa \
#   --scoring-batch-size 8 \
#   --verdict-top-k 5 \
#   --evaluate \
#   --eval-output results/multi_trace_scorer_qlora_spanish_validation_eval_top5.json

# CUDA_VISIBLE_DEVICES=0 \
# uv run scripts/infer_trace_scorer.py \
#   --dataset-path dataset/arabic/validation.json \
#   --language-name arabic \
#   --adapter-path /scratch-scc/projects/ag_gipp/$USER/checkpoints/multi_trace_scorer_qlora_en_val/final_adapter \
#   --output results/multi_trace_scorer_qlora_arabic_validation_predictions_top5.json \
#   --supervisor-output results/Task2_Numerical_claims_Arabic_multilingual_top5.json \
#   --load-in-4bit \
#   --attn-implementation sdpa \
#   --scoring-batch-size 8 \
#   --verdict-top-k 5 \
#   --evaluate \
#   --eval-output results/multi_trace_scorer_qlora_arabic_validation_eval_top5.json

##############

# CUDA_VISIBLE_DEVICES=0 \
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# uv run scripts/train_trace_scorer.py \
#   --train-dataset dataset/train_multi.json \
#   --validation-dataset dataset/validation_multi.json \
#   --output-dir /scratch-scc/projects/ag_gipp/$USER/checkpoints/multi_trace_scorer_qlora_numeric_en_val \
#   --load-in-4bit \
#   --target-modules all-linear \
#   --attn-implementation sdpa \
#   --per-device-train-batch-size 8 \
#   --per-device-eval-batch-size 8 \
#   --gradient-accumulation-steps 2 \
#   --use-numeric-embedding

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# uv run torchrun --standalone --nproc_per_node=2 scripts/train_trace_scorer.py \
#   --train-dataset dataset/train_multi.json \
#   --validation-dataset dataset/validation_multi.json \
#   --output-dir /scratch-scc/projects/ag_gipp/$USER/checkpoints/multi_trace_scorer_qlora_numeric_10epoch_es2_lr1e4_wandb \
#   --load-in-4bit \
#   --target-modules all-linear \
#   --attn-implementation sdpa \
#   --per-device-train-batch-size 8 \
#   --per-device-eval-batch-size 8 \
#   --gradient-accumulation-steps 1 \
#   --num-train-epochs 10 \
#   --learning-rate 1e-4 \
#   --warmup-ratio 0.03 \
#   --logging-steps 25 \
#   --early-stopping-patience 2 \
#   --metric-for-best-model positive_f1 \
#   --greater-is-better \
#   --eval-strategy epoch \
#   --save-strategy epoch \
#   --save-total-limit 3 \
#   --report-to wandb \
#   --use-numeric-embedding \
#   --wandb-entity ahmadrazakhawaja-g-ttingen-campus \
#   --wandb-project numerical-fact-checking \
#   --wandb-group multilingual-trace-scorer \
#   --wandb-tags multilingual qlora trace-scorer numeric early-stopping 10epoch lr1e-4 \
#   --run-name multi-trace-scorer-numeric-10epoch-es2-lr1e4-4gpu \
#   --no-ddp-find-unused-parameters


# CUDA_VISIBLE_DEVICES=0 \
# uv run scripts/infer_trace_scorer.py \
#   --dataset-path dataset/english/validation.json \
#   --language-name english \
#   --adapter-path /scratch-scc/projects/ag_gipp/$USER/checkpoints/multi_trace_scorer_qlora_numeric_en_val/final_adapter \
#   --output results/multi_trace_scorer_qlora_numeric_en_validation_predictions_top5.json \
#   --supervisor-output results/Task2_Numerical_claims_English_multilingual_numeric_top5.json \
#   --load-in-4bit \
#   --attn-implementation sdpa \
#   --scoring-batch-size 8 \
#   --verdict-top-k 5 \
#   --use-numeric-embedding \
#   --evaluate \
#   --eval-output results/multi_trace_scorer_qlora_numeric_en_validation_eval_top5.json

# CUDA_VISIBLE_DEVICES=0 \
# uv run scripts/infer_trace_scorer.py \
#   --dataset-path dataset/spanish/validation.json \
#   --language-name spanish \
#   --adapter-path /scratch-scc/projects/ag_gipp/$USER/checkpoints/multi_trace_scorer_qlora_numeric_en_val/final_adapter \
#   --output results/multi_trace_scorer_qlora_numeric_spanish_validation_predictions_top5.json \
#   --supervisor-output results/Task2_Numerical_claims_Spanish_multilingual_numeric_top5.json \
#   --load-in-4bit \
#   --attn-implementation sdpa \
#   --scoring-batch-size 8 \
#   --verdict-top-k 5 \
#   --use-numeric-embedding \
#   --evaluate \
#   --eval-output results/multi_trace_scorer_qlora_numeric_spanish_validation_eval_top5.json

# CUDA_VISIBLE_DEVICES=0 \
# uv run scripts/infer_trace_scorer.py \
#   --dataset-path dataset/arabic/validation.json \
#   --language-name arabic \
#   --adapter-path /scratch-scc/projects/ag_gipp/$USER/checkpoints/multi_trace_scorer_qlora_numeric_en_val/final_adapter \
#   --output results/multi_trace_scorer_qlora_numeric_arabic_validation_predictions_top5.json \
#   --supervisor-output results/Task2_Numerical_claims_Arabic_multilingual_numeric_top5.json \
#   --load-in-4bit \
#   --attn-implementation sdpa \
#   --scoring-batch-size 8 \
#   --verdict-top-k 5 \
#   --use-numeric-embedding \
#   --evaluate \
#   --eval-output results/multi_trace_scorer_qlora_numeric_arabic_validation_eval_top5.json



#################### English Normal

# CUDA_VISIBLE_DEVICES=0 \
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# uv run scripts/train_trace_scorer.py \
#   --train-dataset dataset/english/train.json \
#   --validation-dataset dataset/english/validation.json \
#   --output-dir checkpoints/english_trace_scorer_qlora \
#   --load-in-4bit \
#   --target-modules all-linear \
#   --attn-implementation sdpa \
#   --per-device-train-batch-size 8 \
#   --per-device-eval-batch-size 8 \
#   --gradient-accumulation-steps 2

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# uv run torchrun --standalone --nproc_per_node=2 scripts/train_trace_scorer.py \
#   --train-dataset dataset/english/train.json \
#   --validation-dataset dataset/english/validation.json \
#   --output-dir /scratch-scc/projects/ag_gipp/$USER/checkpoints/english_trace_scorer_qlora_10epoch_es2_lr1e4_wandb \
#   --load-in-4bit \
#   --target-modules all-linear \
#   --attn-implementation sdpa \
#   --per-device-train-batch-size 8 \
#   --per-device-eval-batch-size 8 \
#   --gradient-accumulation-steps 1 \
#   --num-train-epochs 10 \
#   --learning-rate 1e-4 \
#   --warmup-ratio 0.03 \
#   --logging-steps 25 \
#   --early-stopping-patience 2 \
#   --metric-for-best-model positive_f1 \
#   --greater-is-better \
#   --eval-strategy epoch \
#   --save-strategy epoch \
#   --save-total-limit 3 \
#   --report-to wandb \
#   --wandb-entity ahmadrazakhawaja-g-ttingen-campus \
#   --wandb-project numerical-fact-checking \
#   --wandb-group english-trace-scorer \
#   --wandb-tags english qlora trace-scorer early-stopping 10epoch lr1e-4 \
#   --run-name english-trace-scorer-10epoch-es2-lr1e4-4gpu \
#   --no-ddp-find-unused-parameters


# CUDA_VISIBLE_DEVICES=0 \
# uv run scripts/infer_trace_scorer.py \
#   --dataset-path dataset/english/validation.json \
#   --language-name english \
#   --adapter-path checkpoints/english_trace_scorer_qlora/final_adapter \
#   --output results/english_trace_scorer_validation_predictions_top5.json \
#   --supervisor-output results/Task2_Numerical_claims_English_validation_top5.json \
#   --load-in-4bit \
#   --attn-implementation sdpa \
#   --scoring-batch-size 8 \
#   --verdict-top-k 5 \
#   --evaluate \
#   --eval-output results/english_trace_scorer_validation_eval_top5.json

######################

###################### English Numeric

# CUDA_VISIBLE_DEVICES=0 \
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# uv run scripts/train_trace_scorer.py \
#   --train-dataset dataset/english/train.json \
#   --validation-dataset dataset/english/validation.json \
#   --output-dir checkpoints/english_trace_scorer_qlora_numeric \
#   --load-in-4bit \
#   --target-modules all-linear \
#   --attn-implementation sdpa \
#   --per-device-train-batch-size 8 \
#   --per-device-eval-batch-size 8 \
#   --gradient-accumulation-steps 2 \
#   --use-numeric-embedding

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# uv run torchrun --standalone --nproc_per_node=3 scripts/train_trace_scorer.py \
#   --train-dataset dataset/english/train.json \
#   --validation-dataset dataset/english/validation.json \
#   --output-dir /scratch-scc/projects/ag_gipp/$USER/checkpoints/english_trace_scorer_numeric_qlora_10epoch_es2_lr1e4_wandb \
#   --load-in-4bit \
#   --target-modules all-linear \
#   --attn-implementation sdpa \
#   --per-device-train-batch-size 8 \
#   --per-device-eval-batch-size 8 \
#   --gradient-accumulation-steps 1 \
#   --num-train-epochs 10 \
#   --learning-rate 1e-4 \
#   --warmup-ratio 0.03 \
#   --logging-steps 25 \
#   --early-stopping-patience 2 \
#   --metric-for-best-model positive_f1 \
#   --greater-is-better \
#   --eval-strategy epoch \
#   --save-strategy epoch \
#   --save-total-limit 3 \
#   --report-to wandb \
#   --wandb-entity ahmadrazakhawaja-g-ttingen-campus \
#   --wandb-project numerical-fact-checking \
#   --wandb-group english-trace-scorer \
#   --wandb-tags english qlora trace-scorer numeric early-stopping 10epoch lr1e-4 \
#   --run-name english-trace-scorer-numeric-10epoch-es2-lr1e4-4gpu \
#   --no-ddp-find-unused-parameters \
#   --use-numeric-embedding


# CUDA_VISIBLE_DEVICES=0 \
# uv run scripts/infer_trace_scorer.py \
#   --dataset-path dataset/english/validation.json \
#   --language-name english \
#   --adapter-path checkpoints/english_trace_scorer_qlora_numeric/final_adapter \
#   --output results/english_trace_scorer_numeric_validation_predictions_top5.json \
#   --supervisor-output results/Task2_Numerical_claims_English_numeric_validation_top5.json \
#   --load-in-4bit \
#   --attn-implementation sdpa \
#   --scoring-batch-size 8 \
#   --verdict-top-k 5 \
#   --use-numeric-embedding \
#   --evaluate \
#   --eval-output results/english_trace_scorer_numeric_validation_eval_top5.json

#################################

########################## Spanish Normal

# CUDA_VISIBLE_DEVICES=0 \
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# uv run scripts/train_trace_scorer.py \
#   --train-dataset dataset/spanish/train.json \
#   --validation-dataset dataset/spanish/validation.json \
#   --output-dir checkpoints/spanish_trace_scorer_qlora \
#   --load-in-4bit \
#   --target-modules all-linear \
#   --attn-implementation sdpa \
#   --per-device-train-batch-size 8 \
#   --per-device-eval-batch-size 8 \
#   --gradient-accumulation-steps 2

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# uv run torchrun --standalone --nproc_per_node=3 scripts/train_trace_scorer.py \
#   --train-dataset dataset/spanish/train.json \
#   --validation-dataset dataset/spanish/validation.json \
#   --output-dir /scratch-scc/projects/ag_gipp/$USER/checkpoints/spanish_trace_scorer_qlora_10epoch_es2_lr1e4_wandb \
#   --load-in-4bit \
#   --target-modules all-linear \
#   --attn-implementation sdpa \
#   --per-device-train-batch-size 8 \
#   --per-device-eval-batch-size 8 \
#   --gradient-accumulation-steps 1 \
#   --num-train-epochs 10 \
#   --learning-rate 1e-4 \
#   --warmup-ratio 0.03 \
#   --logging-steps 25 \
#   --early-stopping-patience 2 \
#   --metric-for-best-model positive_f1 \
#   --greater-is-better \
#   --eval-strategy epoch \
#   --save-strategy epoch \
#   --save-total-limit 3 \
#   --report-to wandb \
#   --wandb-entity ahmadrazakhawaja-g-ttingen-campus \
#   --wandb-project numerical-fact-checking \
#   --wandb-group spanish-trace-scorer \
#   --wandb-tags spanish qlora trace-scorer early-stopping 10epoch lr1e-4 \
#   --run-name spanish-trace-scorer-10epoch-es2-lr1e4-4gpu \
#   --no-ddp-find-unused-parameters


# CUDA_VISIBLE_DEVICES=0 \
# uv run scripts/infer_trace_scorer.py \
#   --dataset-path dataset/spanish/validation.json \
#   --language-name spanish \
#   --adapter-path checkpoints/spanish_trace_scorer_qlora/final_adapter \
#   --output results/spanish_trace_scorer_validation_predictions_top5.json \
#   --supervisor-output results/Task2_Numerical_claims_Spanish_validation_top5.json \
#   --load-in-4bit \
#   --attn-implementation sdpa \
#   --scoring-batch-size 8 \
#   --verdict-top-k 5 \
#   --evaluate \
#   --eval-output results/spanish_trace_scorer_validation_eval_top5.json

##########################

########################### Spanish Numeric

# CUDA_VISIBLE_DEVICES=0 \
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# uv run scripts/train_trace_scorer.py \
#   --train-dataset dataset/spanish/train.json \
#   --validation-dataset dataset/spanish/validation.json \
#   --output-dir checkpoints/spanish_trace_scorer_qlora_numeric \
#   --load-in-4bit \
#   --target-modules all-linear \
#   --attn-implementation sdpa \
#   --per-device-train-batch-size 8 \
#   --per-device-eval-batch-size 8 \
#   --gradient-accumulation-steps 2 \
#   --use-numeric-embedding

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# uv run torchrun --standalone --nproc_per_node=3 scripts/train_trace_scorer.py \
#   --train-dataset dataset/spanish/train.json \
#   --validation-dataset dataset/spanish/validation.json \
#   --output-dir /scratch-scc/projects/ag_gipp/$USER/checkpoints/spanish_trace_scorer_numeric_qlora_10epoch_es2_lr1e4_wandb \
#   --load-in-4bit \
#   --target-modules all-linear \
#   --attn-implementation sdpa \
#   --per-device-train-batch-size 8 \
#   --per-device-eval-batch-size 8 \
#   --gradient-accumulation-steps 1 \
#   --num-train-epochs 10 \
#   --learning-rate 1e-4 \
#   --warmup-ratio 0.03 \
#   --logging-steps 25 \
#   --early-stopping-patience 2 \
#   --metric-for-best-model positive_f1 \
#   --greater-is-better \
#   --eval-strategy epoch \
#   --save-strategy epoch \
#   --save-total-limit 3 \
#   --report-to wandb \
#   --wandb-entity ahmadrazakhawaja-g-ttingen-campus \
#   --wandb-project numerical-fact-checking \
#   --wandb-group spanish-trace-scorer \
#   --wandb-tags spanish qlora trace-scorer numeric early-stopping 10epoch lr1e-4 \
#   --run-name spanish-trace-scorer-numeric-10epoch-es2-lr1e4-4gpu \
#   --no-ddp-find-unused-parameters \
#   --use-numeric-embedding


# CUDA_VISIBLE_DEVICES=0 \
# uv run scripts/infer_trace_scorer.py \
#   --dataset-path dataset/spanish/validation.json \
#   --language-name spanish \
#   --adapter-path checkpoints/spanish_trace_scorer_qlora_numeric/final_adapter \
#   --output results/spanish_trace_scorer_numeric_validation_predictions_top5.json \
#   --supervisor-output results/Task2_Numerical_claims_Spanish_numeric_validation_top5.json \
#   --load-in-4bit \
#   --attn-implementation sdpa \
#   --scoring-batch-size 8 \
#   --verdict-top-k 5 \
#   --use-numeric-embedding \
#   --evaluate \
#   --eval-output results/spanish_trace_scorer_numeric_validation_eval_top5.json


###########################

############################ Arabic Normal

# CUDA_VISIBLE_DEVICES=0 \
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# uv run scripts/train_trace_scorer.py \
#   --train-dataset dataset/arabic/train.json \
#   --validation-dataset dataset/arabic/validation.json \
#   --output-dir checkpoints/arabic_trace_scorer_qlora \
#   --load-in-4bit \
#   --target-modules all-linear \
#   --attn-implementation sdpa \
#   --per-device-train-batch-size 8 \
#   --per-device-eval-batch-size 8 \
#   --gradient-accumulation-steps 2

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# uv run torchrun --standalone --nproc_per_node=3 scripts/train_trace_scorer.py \
#   --train-dataset dataset/arabic/train.json \
#   --validation-dataset dataset/arabic/validation.json \
#   --output-dir /scratch-scc/projects/ag_gipp/$USER/checkpoints/arabic_trace_scorer_qlora_10epoch_es2_lr1e4_wandb \
#   --load-in-4bit \
#   --target-modules all-linear \
#   --attn-implementation sdpa \
#   --per-device-train-batch-size 8 \
#   --per-device-eval-batch-size 8 \
#   --gradient-accumulation-steps 1 \
#   --num-train-epochs 10 \
#   --learning-rate 1e-4 \
#   --warmup-ratio 0.03 \
#   --logging-steps 25 \
#   --early-stopping-patience 2 \
#   --metric-for-best-model positive_f1 \
#   --greater-is-better \
#   --eval-strategy epoch \
#   --save-strategy epoch \
#   --save-total-limit 3 \
#   --report-to wandb \
#   --wandb-entity ahmadrazakhawaja-g-ttingen-campus \
#   --wandb-project numerical-fact-checking \
#   --wandb-group arabic-trace-scorer \
#   --wandb-tags arabic qlora trace-scorer early-stopping 10epoch lr1e-4 \
#   --run-name arabic-trace-scorer-10epoch-es2-lr1e4-4gpu \
#   --no-ddp-find-unused-parameters


# CUDA_VISIBLE_DEVICES=0 \
# uv run scripts/infer_trace_scorer.py \
#   --dataset-path dataset/arabic/validation.json \
#   --language-name arabic \
#   --adapter-path checkpoints/arabic_trace_scorer_qlora/final_adapter \
#   --output results/arabic_trace_scorer_validation_predictions_top5.json \
#   --supervisor-output results/Task2_Numerical_claims_Arabic_validation_top5.json \
#   --load-in-4bit \
#   --attn-implementation sdpa \
#   --scoring-batch-size 8 \
#   --verdict-top-k 5 \
#   --evaluate \
#   --eval-output results/arabic_trace_scorer_validation_eval_top5.json


############################

############################ Arabic Numeric

# CUDA_VISIBLE_DEVICES=0 \
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# uv run scripts/train_trace_scorer.py \
#   --train-dataset dataset/arabic/train.json \
#   --validation-dataset dataset/arabic/validation.json \
#   --output-dir checkpoints/arabic_trace_scorer_qlora_numeric \
#   --load-in-4bit \
#   --target-modules all-linear \
#   --attn-implementation sdpa \
#   --per-device-train-batch-size 8 \
#   --per-device-eval-batch-size 8 \
#   --gradient-accumulation-steps 2 \
#   --use-numeric-embedding

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# uv run torchrun --standalone --nproc_per_node=3 scripts/train_trace_scorer.py \
#   --train-dataset dataset/arabic/train.json \
#   --validation-dataset dataset/arabic/validation.json \
#   --output-dir /scratch-scc/projects/ag_gipp/$USER/checkpoints/arabic_trace_scorer_numeric_qlora_10epoch_es2_lr1e4_wandb \
#   --load-in-4bit \
#   --target-modules all-linear \
#   --attn-implementation sdpa \
#   --per-device-train-batch-size 8 \
#   --per-device-eval-batch-size 8 \
#   --gradient-accumulation-steps 1 \
#   --num-train-epochs 10 \
#   --learning-rate 1e-4 \
#   --warmup-ratio 0.03 \
#   --logging-steps 25 \
#   --early-stopping-patience 2 \
#   --metric-for-best-model positive_f1 \
#   --greater-is-better \
#   --eval-strategy epoch \
#   --save-strategy epoch \
#   --save-total-limit 3 \
#   --report-to wandb \
#   --wandb-entity ahmadrazakhawaja-g-ttingen-campus \
#   --wandb-project numerical-fact-checking \
#   --wandb-group arabic-trace-scorer \
#   --wandb-tags arabic qlora trace-scorer numeric early-stopping 10epoch lr1e-4 \
#   --run-name arabic-trace-scorer-numeric-10epoch-es2-lr1e4-4gpu \
#   --no-ddp-find-unused-parameters \
#   --use-numeric-embedding


# CUDA_VISIBLE_DEVICES=0 \
# uv run scripts/infer_trace_scorer.py \
#   --dataset-path dataset/arabic/validation.json \
#   --language-name arabic \
#   --adapter-path checkpoints/arabic_trace_scorer_qlora_numeric/final_adapter \
#   --output results/arabic_trace_scorer_numeric_validation_predictions_top5.json \
#   --supervisor-output results/Task2_Numerical_claims_Arabic_numeric_validation_top5.json \
#   --load-in-4bit \
#   --attn-implementation sdpa \
#   --scoring-batch-size 8 \
#   --verdict-top-k 5 \
#   --use-numeric-embedding \
#   --evaluate \
#   --eval-output results/arabic_trace_scorer_numeric_validation_eval_top5.json


#########################




echo "Finished at: $(date)"