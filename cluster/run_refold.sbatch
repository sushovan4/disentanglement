#!/bin/bash
#SBATCH --job-name=ecp-refold
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=cpu
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-3
#SBATCH --output=cluster/logs/refold_%A_%a.out
#SBATCH --error=cluster/logs/refold_%A_%a.err

# ──────────────────────────────────────────────────────────────────
# Scale-free paired statistics (2026-09-06).  The campaign's fold statistic
# was the raw profile mass (length units); a paired test on it across epochs
# certified feature-norm dynamics.  These tasks recompute every fold statistic
# on the dimensionless mass \int|Dchi|dr / r_max, on the SAME fold clouds, and
# compute the cheap separability baselines (Fisher, mean distance, 5-NN
# cross-class rate, energy distance) on those clouds for the paired-baseline
# comparison of E3.
#   task 0  refold: the five E3 seeds            -> results/refold/e3_*.json
#   task 1  refold: the LR-schedule ablation      -> results/refold/ablr_*.json
#   task 2  refold: the three E2 models           -> results/refold/e2_*.json
#   task 3  E9 (grokking) re-measured with the scale-free fold statistic:
#           the five dense runs + shuffled controls, then the report.
#           Old (raw-mass) results are moved to results/e9_grok_rawmass/.
# Afterwards, on the login node:
#   $PY -m experiments.e3_dynamics        # writes e3_dynamics_{ecp,raw,fisher,knn,energy,meandist}.json
#   $PY -m experiments.e9_grok --report --dense-only
# then commit results/refold results/e3_dynamics_*.json results/e9_grok*.
# ──────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$HOME/disentanglement"
mkdir -p cluster/logs
module purge
module load anaconda/2023.03
PY="$HOME/.conda/envs/mixup-env/bin/python"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

case "$SLURM_ARRAY_TASK_ID" in
  0) "$PY" -m experiments.refold --tags e3   --jobs "$SLURM_CPUS_PER_TASK" ;;
  1) "$PY" -m experiments.refold --tags ablr --jobs "$SLURM_CPUS_PER_TASK" ;;
  2) "$PY" -m experiments.refold --tags e2   --jobs "$SLURM_CPUS_PER_TASK" ;;
  3) if [ -d results/e9_grok ] && [ ! -d results/e9_grok_rawmass ]; then mv results/e9_grok results/e9_grok_rawmass; fi
     mkdir -p results/e9_grok
     # indices follow experiments.grok CONFIGS, as in run_e9.sbatch / run_grok.sbatch
     for i in $("$PY" -c "from experiments.grok import CONFIGS; print(' '.join(str(i) for i,c in enumerate(CONFIGS)))"); do
       "$PY" -m experiments.e9_grok "$i" --layers penult --B 99
       "$PY" -m experiments.e9_grok "$i" --layers penult --B 0 --shuffle-classes || true
     done
     "$PY" -m experiments.e9_grok --report --dense-only ;;
esac
echo "TASK $SLURM_ARRAY_TASK_ID DONE"
