#!/bin/bash
#SBATCH --job-name=ecp-measure
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=cpu
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=cluster/logs/measure_%A_%a.out
#SBATCH --error=cluster/logs/measure_%A_%a.err

# ──────────────────────────────────────────────────────────────────
# ECP measurement of one (config, epoch) cell per array task.
# The array range is NOT hardcoded: submit_all.sh generates
# cluster/measure_manifest.txt (`python -m experiments.measure --manifest`)
# and passes `--array=0-$((rows-1))`.  Each row: "<config_idx> <epoch>".
# ──────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$HOME/disentanglement"

module purge
module load anaconda/2023.03
PYTHON="$HOME/.conda/envs/mixup-env/bin/python"

# measurement parallelizes over pairs via a process pool; keep BLAS serial
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

MANIFEST="${DISENTANGLE_MANIFEST:-cluster/measure_manifest.txt}"
ROW=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
read -r CFG EPOCH <<< "$ROW"
echo "task $SLURM_ARRAY_TASK_ID -> config $CFG epoch $EPOCH"

"$PYTHON" -m experiments.measure "$CFG" "$EPOCH"
