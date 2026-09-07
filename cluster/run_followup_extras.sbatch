#!/bin/bash
#SBATCH --job-name=ecp-extras
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=cpu
#SBATCH --array=0-5
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=cluster/logs/extras_%A_%a.out
#SBATCH --error=cluster/logs/extras_%A_%a.err

# ──────────────────────────────────────────────────────────────────
# Loophole-closing cells on EXISTING models (no new training needed):
#   0-2  deep-layer triple rescans at d0=5 for the three E2 finals
#        (the PCA-to-4 projection is the scan's most attackable choice;
#        deep layers are the only hunting ground where pairs are null)
#   3-5  mid-training triple scans (transient higher-order structure),
#        E3 seed 0 at epochs 2, 8, 40
# Pairwise matrices are NOT recomputed (SKIP_PAIRS); outputs carry a
# suffix so the original cells stay untouched.
# ──────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$HOME/disentanglement"

module purge
module load anaconda/2023.03
PYTHON="$HOME/.conda/envs/mixup-env/bin/python"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export DISENTANGLE_FORCE_TRIPLES=1
export DISENTANGLE_SKIP_PAIRS=1

case "$SLURM_ARRAY_TASK_ID" in
  # deep-layer d0=5 rescans (escalation trimmed to B=249 for tractability)
  0) export DISENTANGLE_TRIPLE_D0=5 DISENTANGLE_B_ESC=249 \
            DISENTANGLE_TRIPLE_LAYERS=stage3,penult \
            DISENTANGLE_OUT_SUFFIX=_tripd5
     "$PYTHON" -m experiments.measure 5 80 ;;
  1) export DISENTANGLE_TRIPLE_D0=5 DISENTANGLE_B_ESC=249 \
            DISENTANGLE_TRIPLE_LAYERS=stage3,penult \
            DISENTANGLE_OUT_SUFFIX=_tripd5
     "$PYTHON" -m experiments.measure 6 80 ;;
  2) export DISENTANGLE_TRIPLE_D0=5 DISENTANGLE_B_ESC=249 \
            DISENTANGLE_TRIPLE_LAYERS=block6,block8,penult \
            DISENTANGLE_OUT_SUFFIX=_tripd5
     "$PYTHON" -m experiments.measure 7 100 ;;
  # mid-training scans at the standard d0=4 (config 0 = e3 seed 0)
  3) export DISENTANGLE_TRIPLE_LAYERS=stage2,stage3,penult \
            DISENTANGLE_OUT_SUFFIX=_tripmid
     "$PYTHON" -m experiments.measure 0 2 ;;
  4) export DISENTANGLE_TRIPLE_LAYERS=stage2,stage3,penult \
            DISENTANGLE_OUT_SUFFIX=_tripmid
     "$PYTHON" -m experiments.measure 0 8 ;;
  5) export DISENTANGLE_TRIPLE_LAYERS=stage2,stage3,penult \
            DISENTANGLE_OUT_SUFFIX=_tripmid
     "$PYTHON" -m experiments.measure 0 40 ;;
esac
