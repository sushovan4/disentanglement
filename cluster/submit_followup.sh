#!/bin/bash
# ──────────────────────────────────────────────────────────────────
# Follow-up campaign (paper upgrade A+B+C):
#   B/C-train : configs 104-110 (LR-schedule ablation x4, SimCLR x2,
#               pretrained ViT-B/16 x1) on the gpu partition
#   B/C-measure: their measure cells (manifest rows >= 171), chained
#   B-extras  : loophole-closing triple rescans on EXISTING models
#               (d0=5 deep layers; mid-training) — submits immediately
#
#   bash cluster/submit_followup.sh
#   bash cluster/submit_followup.sh --extras-only
# ──────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$HOME/disentanglement"
mkdir -p cluster/logs

module load anaconda/2023.03 2>/dev/null || true
PYTHON="$HOME/.conda/envs/mixup-env/bin/python"

EXTRAS_ONLY=0
[ "${1:-}" = "--extras-only" ] && EXTRAS_ONLY=1

echo "→ extras (existing models; no training dependency)..."
EXTRAS=$(sbatch cluster/run_followup_extras.sbatch | awk '{print $NF}')
echo "  extras array: $EXTRAS"
[ "$EXTRAS_ONLY" -eq 1 ] && exit 0

echo "→ prefetching ViT-B/16 weights on the login node (compute nodes are throttled)..."
"$PYTHON" - <<'EOF'
import torchvision
torchvision.models.vit_b_16(
    weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)
print("vit_b_16 weights cached")
EOF

N_CFG=$("$PYTHON" -m experiments.common | tail -1 | awk '{print $2}')
echo "→ training configs 104-$((N_CFG - 1))..."
TRAIN=$(sbatch --array=104-$((N_CFG - 1)) cluster/run_train_gpu.sbatch | awk '{print $NF}')
echo "  train array: $TRAIN"

# measure rows for the new configs only
"$PYTHON" -m experiments.measure --manifest | awk '$1 >= 104' > cluster/followup_manifest.txt
N_ROWS=$(wc -l < cluster/followup_manifest.txt)
MEASURE=$(sbatch --dependency=afterok:$TRAIN --array=0-$((N_ROWS - 1)) \
          --export=ALL,DISENTANGLE_MANIFEST=cluster/followup_manifest.txt \
          cluster/run_measure.sbatch | awk '{print $NF}')
echo "  measure array: $MEASURE ($N_ROWS cells)"

AGG=$(sbatch --dependency=afterok:$MEASURE --account=YOUR_ALLOCATION --partition=cpu \
      --time=01:00:00 --mem=8G --job-name=ecp-fu-agg \
      --output=cluster/logs/fuagg_%j.out --error=cluster/logs/fuagg_%j.err \
      --wrap="cd $HOME/disentanglement && $PYTHON -m experiments.e3_dynamics" \
      | awk '{print $NF}')
echo "  ablr aggregation (after measure): $AGG"

echo
echo "monitor: bash cluster/monitor.sh"
