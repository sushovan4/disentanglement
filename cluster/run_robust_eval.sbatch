#!/bin/bash
#SBATCH --job-name=ecp-robust
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=cluster/logs/robust_%j.out
#SBATCH --error=cluster/logs/robust_%j.err
# Calibration (NLL, ECE) and on-the-fly corruption accuracy for the 96 E5
# population models at their final checkpoint -> results/robust/*.json.
# Pairs with cluster/run_pop_refold.sbatch; aggregate with
#   python -m experiments.disagreement
set -euo pipefail
cd "$HOME/disentanglement"
mkdir -p cluster/logs
module purge
module load anaconda/2023.03
PY="$HOME/.conda/envs/mixup-env/bin/python"
"$PY" -m experiments.robust_eval --tags pop
echo "ROBUST DONE"
