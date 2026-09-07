#!/bin/bash
#SBATCH --job-name=ecp-e7
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=cpu
#SBATCH --array=0-3
#SBATCH --time=1-12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=cluster/logs/e7_%A_%a.out
#SBATCH --error=cluster/logs/e7_%A_%a.err

# ──────────────────────────────────────────────────────────────────
# E7: interaction spectrum (floors chi(C_j), onset vectors) on the
# final-epoch features of the three E2 models + E3 seed 0.
# Reuses cached features; no training dependency.
# ──────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$HOME/disentanglement"

module purge
module load anaconda/2023.03
PYTHON="$HOME/.conda/envs/mixup-env/bin/python"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

case "$SLURM_ARRAY_TASK_ID" in
  0) "$PYTHON" -m experiments.e7_spectrum 5 80 ;;   # ResNet-56 CIFAR-10
  1) "$PYTHON" -m experiments.e7_spectrum 6 80 ;;   # ResNet-56 CIFAR-100s
  2) "$PYTHON" -m experiments.e7_spectrum 7 100 ;;  # ViT
  3) "$PYTHON" -m experiments.e7_spectrum 0 80 ;;   # ResNet-20 (E3 s0)
esac
