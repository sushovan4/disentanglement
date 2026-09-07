#!/bin/bash
#SBATCH --job-name=ecp-e6
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=cluster/logs/e6_%A_%a.out
#SBATCH --error=cluster/logs/e6_%A_%a.err

# ──────────────────────────────────────────────────────────────────
# E6: train the differentiable-regularizer models (ResNet-20/CIFAR-10,
# dis_lambda in {0,0.5,2,8} x 2 seeds = 8 configs, appended last in
# CONFIGS).  Training auto-detects CUDA; ~25 min/model on a GPU.
#
# Submit (the e6 configs are the last 8; as of writing, indices 111-118):
#   mkdir -p cluster/logs
#   sbatch --array=111-118 cluster/run_e6.sbatch
# Resolve the indices rather than trusting that number if CONFIGS has changed:
#   python -c "from experiments.common import CONFIGS; \
#     i=[k for k,c in enumerate(CONFIGS) if c.tag=='e6']; print(f'{i[0]}-{i[-1]}')"
# Then aggregate (CPU login node is fine):
#   python -m experiments.e6_regularize
# CPU fallback: change partition to cpu, drop --gres, raise --time to 1-00:00.
# ──────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$HOME/disentanglement"

# SLURM does not create the --output directory and fails the task outright if
# it is missing, which is easy to hit after a filesystem restore.  The header
# is already parsed by now, so this only helps the next submission -- but that
# is exactly the case worth protecting.
mkdir -p cluster/logs

module purge
module load anaconda/2023.03
PYTHON="$HOME/.conda/envs/mixup-env/bin/python"

echo "task $SLURM_ARRAY_TASK_ID"
"$PYTHON" -m experiments.train "$SLURM_ARRAY_TASK_ID"
