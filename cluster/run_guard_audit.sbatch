#!/bin/bash
#SBATCH --job-name=ecp-guard
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=cpu
#SBATCH --array=0-23
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=cluster/logs/guard_%A_%a.out
#SBATCH --error=cluster/logs/guard_%A_%a.err
# Re-evaluate the cancellation guard on every archived pairwise cell with the
# campaign window (skip=0) and the refined window that starts one grid point
# after first contact (skip=1) and the plateau criterion.  Serial per shard, ~15 min
# per measurement file; shard count = array size (24 -> ~10 files, ~2.5 h per shard);
# writes incrementally and resumes, so a resubmission continues where it stopped.
# Afterwards:  python -m experiments.guard_audit --report
set -euo pipefail
cd "$HOME/disentanglement"
mkdir -p cluster/logs
module purge
module load anaconda/2023.03
PY="$HOME/.conda/envs/mixup-env/bin/python"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PY" -m experiments.guard_audit --shard "$SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT"
echo "SHARD $SLURM_ARRAY_TASK_ID DONE"
