#!/bin/bash
#SBATCH --job-name=ecp-prearxiv
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=cpu
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-1
#SBATCH --output=cluster/logs/prearxiv_%A_%a.out
#SBATCH --error=cluster/logs/prearxiv_%A_%a.err

# ──────────────────────────────────────────────────────────────────
# Two pre-arXiv upgrades for the disentanglement paper (2026-09-05).
#   task 0  E10 null floor at 10 seeds (40 triples per null, was 12)
#           -> results/dominance_null.json
#   task 1  E11 second corpus: BERT-base and GPT-2 on DBpedia-14 (10 classes)
#           -> features/foundation/{bert_base,gpt2}__dbpedia.npz, then the
#              E11 protocol -> results/init_dominance/*__dbpedia_d05.json
# Before submitting, ON THE LOGIN NODE (compute nodes have no network):
#   DISENTANGLE_TEXT_DATASET=dbpedia $PY -m experiments.foundation --prefetch
#   (installs nothing; if it says MISSING, `$PY -m pip install datasets`)
# Task 0 also needs features/init/e3_resnet20w1_cifar10_wd0.0005_aug_s0_init.npz,
# which the init0 job of 2026-09-03 wrote.
# ──────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$HOME/disentanglement"
mkdir -p cluster/logs
module purge
module load anaconda/2023.03
PY="$HOME/.conda/envs/mixup-env/bin/python"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 DISENTANGLE_JOBS=${SLURM_CPUS_PER_TASK:-8}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

case "$SLURM_ARRAY_TASK_ID" in
  0) "$PY" -m experiments.dominance_null 10 ;;
  1) export DISENTANGLE_TEXT_DATASET=dbpedia
     "$PY" -m experiments.foundation 7          # bert_base
     "$PY" -m experiments.foundation 6          # gpt2
     for M in bert_base gpt2; do
       "$PY" - <<PYEOF
import numpy as np
z = np.load('features/foundation/${M}__dbpedia.npz')
_, i = np.unique(np.round(z['penult'].astype(np.float64), 6), axis=0, return_index=True); i = np.sort(i)
np.savez_compressed('features/foundation/${M}__dbpedia_dedup.npz', **{k: z[k][i] for k in z.files})
print('${M}', len(z['labels']), '->', len(i), 'per class', np.bincount(z['labels'][i]).tolist())
PYEOF
     done
     "$PY" -m experiments.init_dominance measure --npz features/foundation/bert_base__dbpedia_dedup.npz --name bert_base__dbpedia --d0 5 --layers penult,block6
     "$PY" -m experiments.init_dominance measure --npz features/foundation/gpt2__dbpedia_dedup.npz --name gpt2__dbpedia --d0 5 --layers penult
     "$PY" -m experiments.init_dominance report ;;
esac
echo "TASK $SLURM_ARRAY_TASK_ID DONE"
