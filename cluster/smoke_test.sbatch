#!/bin/bash
#SBATCH --job-name=ecp-smoke
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=nano
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=cluster/logs/smoke_%j.out
#SBATCH --error=cluster/logs/smoke_%j.err

# Verifies env + library before the arrays: ECP ground-truth checks,
# torch model round-trips, and dataset availability.

set -euo pipefail
cd "$HOME/disentanglement"

module purge
module load anaconda/2023.03
PYTHON="$HOME/.conda/envs/mixup-env/bin/python"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

"$PYTHON" -m experiments.smoke
"$PYTHON" - <<'EOF'
from experiments.data import torch_datasets
for name in ("cifar10", "cifar100s", "mnist"):
    tr, te, k = torch_datasets(name, augment=False)
    print(f"{name}: train={len(tr)} test={len(te)} classes={k}")
EOF
echo "smoke test passed"
