#!/bin/bash
#SBATCH --job-name=ecp-e9
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=cpu
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=cluster/logs/e9_%A_%a.out
#SBATCH --error=cluster/logs/e9_%A_%a.err

# ──────────────────────────────────────────────────────────────────
# Direction B, stage 2: measure each grokking run's checkpoints and run the
# certified consecutive-checkpoint paired tests.  CPU only (no training).
#
#   sbatch --array=0-4 cluster/run_e9.sbatch      # after run_grok completes
#   python -m experiments.e9_grok --report        # align reorg vs grok step
#
# Cost: the trajectory claim rests on FOLD statistics, which need no
# permutations.  --B adds absolute quotients at linear cost; B=99 on one
# layer over ~40 checkpoints x 45 pairs is the default.  Set --B 0 for a
# fast trajectory-only pass, or add layers with --layers block1,block2,penult.
# ──────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$HOME/disentanglement"

module purge
module load anaconda/2023.03
PYTHON="$HOME/.conda/envs/mixup-env/bin/python"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "task $SLURM_ARRAY_TASK_ID"
"$PYTHON" -m experiments.e9_grok "$SLURM_ARRAY_TASK_ID" --layers penult --B 99
