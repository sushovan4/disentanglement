#!/bin/bash
# ──────────────────────────────────────────────────────────────────
# Driver for the full disentangle experiment suite on Pegasus.
#
#   bash cluster/submit_all.sh              # smoke → train → measure → aggregate
#   bash cluster/submit_all.sh --gpu        # train on the gpu partition
#   bash cluster/submit_all.sh --skip-smoke
#   bash cluster/submit_all.sh --smoke-only
#   bash cluster/submit_all.sh --measure-only   # trainings already done
#
# Dependency chain (all automatic via --dependency=afterok):
#   smoke → train array (104) ─┬→ measure array (manifest-sized) ─┬→ e4
#   e1 (independent)           └→ (E2 triple scans ride measure)  └→ aggregate
# ──────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$HOME/disentanglement"
mkdir -p cluster/logs results data checkpoints features

module load anaconda/2023.03 2>/dev/null || true
PYTHON="$HOME/.conda/envs/mixup-env/bin/python"

SKIP_SMOKE=0; SMOKE_ONLY=0; MEASURE_ONLY=0; TRAIN_SBATCH=cluster/run_train.sbatch
for arg in "$@"; do
  case "$arg" in
    --skip-smoke) SKIP_SMOKE=1 ;;
    --smoke-only) SMOKE_ONLY=1 ;;
    --measure-only) MEASURE_ONLY=1 ;;
    --gpu) TRAIN_SBATCH=cluster/run_train_gpu.sbatch ;;
    *) echo "unknown arg: $arg"; exit 1 ;;
  esac
done

# datasets must be fetched on the login node BEFORE the smoke job:
# compute nodes lack internet, and the smoke test loads every dataset.
echo "→ prefetching datasets on login node..."
"$PYTHON" -m experiments.data

if [ "$SKIP_SMOKE" -ne 1 ] && [ "$MEASURE_ONLY" -ne 1 ]; then
  echo "→ smoke test (waits to completion)..."
  sbatch --wait cluster/smoke_test.sbatch
  echo "  smoke passed."
  [ "$SMOKE_ONLY" -eq 1 ] && exit 0
fi

N_CFG=$("$PYTHON" -m experiments.common | tail -1 | awk '{print $2}')
echo "→ $N_CFG training configs"

DEP=""
if [ "$MEASURE_ONLY" -ne 1 ]; then
  TRAIN_JOB=$(sbatch --array=0-$((N_CFG - 1)) "$TRAIN_SBATCH" | awk '{print $NF}')
  echo "  train array: $TRAIN_JOB"
  DEP="--dependency=afterok:$TRAIN_JOB"
fi

"$PYTHON" -m experiments.measure --manifest > cluster/measure_manifest.txt
N_ROWS=$(wc -l < cluster/measure_manifest.txt)
MEASURE_JOB=$(sbatch $DEP --array=0-$((N_ROWS - 1)) cluster/run_measure.sbatch | awk '{print $NF}')
echo "  measure array: $MEASURE_JOB ($N_ROWS cells)"

E1_JOB=$(sbatch cluster/run_cpu_singles.sbatch e1 | awk '{print $NF}')
echo "  e1: $E1_JOB"

E4_JOB=$(sbatch --dependency=afterok:$MEASURE_JOB cluster/run_cpu_singles.sbatch e4 | awk '{print $NF}')
echo "  e4 (after measure): $E4_JOB"

AGG=$(sbatch --dependency=afterok:$MEASURE_JOB --account=YOUR_ALLOCATION --partition=cpu \
      --time=01:00:00 --mem=8G --job-name=ecp-aggregate \
      --output=cluster/logs/aggregate_%j.out --error=cluster/logs/aggregate_%j.err \
      --wrap="cd $HOME/disentanglement && $PYTHON -m experiments.e3_dynamics && $PYTHON -m experiments.e5_predict" \
      | awk '{print $NF}')
echo "  aggregate (e3+e5, after measure): $AGG"

echo
echo "monitor:   bash cluster/monitor.sh"
echo "when done: git add results/ && commit on a results branch"
