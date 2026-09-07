#!/bin/bash
#SBATCH --job-name=ecp-singles
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=cpu
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=cluster/logs/singles_%j.out
#SBATCH --error=cluster/logs/singles_%j.err

# ──────────────────────────────────────────────────────────────────
# The torch-free single-node experiments: E1 (MNIST pairwise at scale)
# and E4 (calibration).  Usage:  sbatch run_cpu_singles.sbatch e1|e4
# E4's guard audit reads results/measure/*, so run e4 AFTER the measure
# array (submit_all.sh chains it with --dependency).
# ──────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$HOME/disentanglement"

module purge
module load anaconda/2023.03
PYTHON="$HOME/.conda/envs/mixup-env/bin/python"

# e1/e4 parallelize over pairs/replicates via process pools; keep BLAS serial
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

case "${1:-}" in
  e1) "$PYTHON" -m experiments.e1_mnist ;;
  e4) "$PYTHON" -m experiments.e4_calibration ;;
  *)  echo "usage: sbatch run_cpu_singles.sbatch e1|e4"; exit 1 ;;
esac
