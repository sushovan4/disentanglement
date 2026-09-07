#!/bin/bash
#SBATCH --job-name=ecp-conc
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=cluster/logs/conc_%A_%a.out
#SBATCH --error=cluster/logs/conc_%A_%a.err

# ──────────────────────────────────────────────────────────────────
# Concentration sweep: extract frozen features from 8 pretrained
# encoders across two modalities, then run the scale-window
# diagnostic across depth.  NO TRAINING -- one forward pass each.
#
# PREFETCH FIRST, on the login node.  Compute nodes have no usable
# internet, and every one of these pulls weights (and 20 Newsgroups
# for the text models):
#
#   python -m experiments.foundation --prefetch
#
# Then:
#   sbatch --array=0-7 cluster/run_concentration.sbatch
#   python -m experiments.foundation --report   # spectra + realized tightness
#   python -m experiments.ambient --diagnose    # the scale-window diagnostic
#
# Indices follow experiments.foundation.MODELS (--list to confirm).
# Text models (gpt2, bert_base) need `transformers` and `scikit-learn`
# in the env; they skip with a message rather than failing the array.
# ──────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$HOME/disentanglement"
mkdir -p cluster/logs

module purge
module load anaconda/2023.03
PYTHON="$HOME/.conda/envs/mixup-env/bin/python"

# HF and torch.hub must read the caches populated by --prefetch on the
# login node; without these they would try (and fail) to reach the network.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "task $SLURM_ARRAY_TASK_ID"
"$PYTHON" -m experiments.foundation "$SLURM_ARRAY_TASK_ID"
