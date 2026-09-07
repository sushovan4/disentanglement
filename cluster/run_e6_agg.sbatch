#!/bin/bash
#SBATCH --job-name=ecp-e6agg
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --output=cluster/logs/e6agg_%j.out
#SBATCH --error=cluster/logs/e6agg_%j.err

# ──────────────────────────────────────────────────────────────────
# E6 aggregation: re-extract penultimate features for each of the 8
# trained models and run the exact ECP permutation test over all 45
# class pairs, i.e. 360 pair-tests at B=199.
#
# This does NOT belong on a login node.  measure.N_JOBS falls back to
# os.cpu_count() when SLURM_CPUS_PER_TASK is unset, so on a login node
# it forks ~45 workers against the two cores cgroups actually grants
# and thrashes.  Under sbatch, SLURM_CPUS_PER_TASK sizes the pool
# correctly.
#
#   sbatch cluster/run_e6_agg.sbatch
#   cat results/e6_regularize.json
#
# Note the JSON is written only after every model is measured, so a
# killed run leaves nothing behind -- let it finish.
# ──────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$HOME/disentanglement"
mkdir -p cluster/logs

module purge
module load anaconda/2023.03
PYTHON="$HOME/.conda/envs/mixup-env/bin/python"

# the pool parallelizes over pairs; keep BLAS serial so the two do not
# multiply into oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$PYTHON" -m experiments.e6_regularize
