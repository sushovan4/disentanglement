#!/bin/bash
#SBATCH --job-name=ecp-e9
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=cluster/logs/e9_%j.out
#SBATCH --error=cluster/logs/e9_%j.err

# ──────────────────────────────────────────────────────────────────
# E9 (grokking) from scratch, 2026-09-06: the checkpoints were lost, so this
# retrains the four dense-window runs the paper uses (grok CONFIGS 9-12:
# three seeds at weight decay 1.0 and one at 0, dense checkpoints every 100
# steps on [1800, 3200]), then measures each with the scale-free fold
# statistic (B=0: the paired claims need fold statistics only; the B=99
# quotients cost ~50 min per checkpoint serially and are not used)
# statistic (true classes and shuffled groupings) and writes the report.
# Small transformers: minutes per run on a V100.  CPU fallback works too
# (--partition=cpu, drop --gres, longer --time).
# ──────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$HOME/disentanglement"
mkdir -p cluster/logs
module purge
module load anaconda/2023.03
PY="$HOME/.conda/envs/mixup-env/bin/python"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

"$PY" -m experiments.grok --list
for i in 9 10 11 12; do
  "$PY" -m experiments.grok "$i"
  "$PY" -m experiments.e9_grok "$i" --layers penult --B 0
  "$PY" -m experiments.e9_grok "$i" --layers penult --B 0 --shuffle-classes
done
# drop the empty records left by the checkpoint-less rerun, then report
"$PY" - <<'PYEOF'
import glob, json, os
for f in glob.glob("results/e9_grok/*.json"):
    if not json.load(open(f)).get("paired"):
        os.remove(f); print("removed empty", f)
PYEOF
"$PY" -m experiments.e9_grok --report --dense-only
echo "E9 DONE"
