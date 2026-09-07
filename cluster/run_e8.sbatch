#!/bin/bash
#SBATCH --job-name=ecp-e8-d0
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=cpu
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=cluster/logs/e8_%A_%a.out
#SBATCH --error=cluster/logs/e8_%A_%a.err

# ──────────────────────────────────────────────────────────────────
# E8: d0-sensitivity of the depth conclusions (Limitations defense).
# 12 cells: {ResNet-56/C10, ResNet-56/C100s, ViT-8, ViT-B/16-pre} x
# d0 in {3,4,6}, final epochs, full pairwise, B=199.  Loads features
# one layer at a time, so 48G suffices even for the ViT-B/16 cells.
# Submit:  sbatch --array=0-11 cluster/run_e8.sbatch
# Then:    python -m experiments.e8_d0 --report
# ──────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$HOME/disentanglement"

module purge
module load anaconda/2023.03
PYTHON="$HOME/.conda/envs/mixup-env/bin/python"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "task $SLURM_ARRAY_TASK_ID"
"$PYTHON" -m experiments.e8_d0 "$SLURM_ARRAY_TASK_ID"
