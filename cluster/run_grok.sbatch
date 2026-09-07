#!/bin/bash
#SBATCH --job-name=ecp-grok
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=cluster/logs/grok_%A_%a.out
#SBATCH --error=cluster/logs/grok_%A_%a.err

# ──────────────────────────────────────────────────────────────────
# Direction B, stage 1: train the grokking runs (5 configs: 3 seeds at
# wd=1.0 that should grok, 2 at wd=0 that should not).
#
#   python -m experiments.grok --list          # see the 5 configs
#   sbatch --array=0-4 cluster/run_grok.sbatch
#
# NOTE the typed GRES (gpu:v100:1) -- Pegasus rejects an untyped gpu:1.
# CPU fallback: --partition=cpu, drop --gres, --time=1-00:00 (slower but
# these models are tiny).
#
# Stage 2 is the measurement, which is CPU-only:
#   sbatch --array=0-4 cluster/run_e9.sbatch
# ──────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$HOME/disentanglement"

module purge
module load anaconda/2023.03
PYTHON="$HOME/.conda/envs/mixup-env/bin/python"

echo "task $SLURM_ARRAY_TASK_ID"
"$PYTHON" -m experiments.grok "$SLURM_ARRAY_TASK_ID"
