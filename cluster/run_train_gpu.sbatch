#!/bin/bash
#SBATCH --job-name=ecp-train
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --array=0-103
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=cluster/logs/train_%A_%a.out
#SBATCH --error=cluster/logs/train_%A_%a.err

# ──────────────────────────────────────────────────────────────────
# GPU variant of run_train.sbatch (gpu partition: 7-day limit, ~40 nodes).
# A ResNet-20/CIFAR-10 run is ~20-30 min on one GPU vs ~6 h on 16 CPU
# threads; the whole 104-model roster clears in a few hours of queue time.
# Requires the CUDA torch wheel in mixup-env (see cluster/README.md).
# Pegasus requires a typed gres (--gres=gpu:v100:1); if v100 isn't right,
# check types with `sinfo -p gpu -o "%N %G" | sort -u` and override at
# submit time:  sbatch --gres=gpu:<type>:1 --array=... run_train_gpu.sbatch
# Submit via: bash cluster/submit_all.sh --gpu
# ──────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$HOME/disentanglement"

module purge
module load anaconda/2023.03
PYTHON="$HOME/.conda/envs/mixup-env/bin/python"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"$PYTHON" -m experiments.train "$SLURM_ARRAY_TASK_ID"
