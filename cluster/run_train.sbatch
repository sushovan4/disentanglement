#!/bin/bash
#SBATCH --job-name=ecp-train
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=cpu
#SBATCH --array=0-103
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=cluster/logs/train_%A_%a.out
#SBATCH --error=cluster/logs/train_%A_%a.err

# ──────────────────────────────────────────────────────────────────
# Disentangle paper: train one model per array task (experiments/common.py
# CONFIGS — 5 E3 seeds + 3 E2 models + 96 E5 population = 104 tasks).
# CPU sizing: ResNet-20 ≈ 4 min/epoch on 16 threads → ≈ 6 h for 80 epochs;
# worst case (ResNet-56 2×) fits in the 5-day wall comfortably.
# If a GPU partition is available (`sinfo -o "%P %G"`), switch
#   --partition and add `#SBATCH --gres=gpu:1` — the code auto-detects CUDA.
# ──────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$HOME/disentanglement"

module purge
module load anaconda/2023.03
# `source activate` does not fix PATH on this cluster's compute nodes;
# use the env's python by absolute path (see pleb/cluster/README.md).
PYTHON="$HOME/.conda/envs/mixup-env/bin/python"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

"$PYTHON" -m experiments.train "$SLURM_ARRAY_TASK_ID"
