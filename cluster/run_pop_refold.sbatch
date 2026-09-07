#!/bin/bash
#SBATCH --job-name=ecp-poprefold
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=cpu
#SBATCH --array=0-7
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=cluster/logs/poprefold_%A_%a.out
#SBATCH --error=cluster/logs/poprefold_%A_%a.err

# ──────────────────────────────────────────────────────────────────
# Follow-up (2026-09-07): scale-free fold statistics + cheap baselines
# (Fisher, mean-distance, energy, 5-NN) on the 96-model E5 population at
# the final epoch, so that the disagreement between the topological
# quotient and the neighbour rate can be scored per (model, layer) cell and
# tested as a predictor of calibration/robustness (cluster/run_robust_eval.sbatch).
# 8 interleaved shards of 12 configs; ~4 min per config on 8 cores.
# Results: results/refold/pop_*_epoch80.json.  Aggregate afterwards with
#   python -m experiments.disagreement
# ──────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$HOME/disentanglement"
mkdir -p cluster/logs
module purge
module load anaconda/2023.03
PY="$HOME/.conda/envs/mixup-env/bin/python"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PY" -m experiments.refold --tags pop --final-only --shard "$SLURM_ARRAY_TASK_ID/8" --jobs "$SLURM_CPUS_PER_TASK"
echo "SHARD $SLURM_ARRAY_TASK_ID DONE"
