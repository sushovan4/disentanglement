#!/bin/bash
#SBATCH --job-name=ecp-wagner
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=cpu
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=cluster/logs/wagner_%A_%a.out
#SBATCH --error=cluster/logs/wagner_%A_%a.err

# ──────────────────────────────────────────────────────────────────
# G1: total-mixup baseline via Wagner et al.'s vendored reference code.
# One (config, final-epoch) cell per array task.  Submit:
#   python -m experiments.wagner_baseline --manifest > cluster/wagner_manifest.txt
#   sbatch --array=0-$(($(wc -l < cluster/wagner_manifest.txt) - 1)) \
#       cluster/run_wagner.sbatch
# Prereq: vendor/Mixup-SoCG26 cloned + ripser_hacked built (vendor/README.md).
# ──────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$HOME/disentanglement"

module purge
module load anaconda/2023.03
PYTHON="$HOME/.conda/envs/mixup-env/bin/python"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

ROW=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" cluster/wagner_manifest.txt)
read -r CFG EPOCH <<< "$ROW"
echo "task $SLURM_ARRAY_TASK_ID -> config $CFG (final epoch $EPOCH)"

"$PYTHON" -m experiments.wagner_baseline "$CFG"
