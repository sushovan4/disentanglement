# Cluster: disentangle paper experiment suite (E1–E5)

End-to-end pipeline for the experiments in `the paper`, run on GW
Pegasus (a Slurm cluster; written for GW Pegasus).

## What runs

| Stage | Script | Tasks | What it produces |
|---|---|---|---|
| smoke | `smoke_test.sbatch` | 1 (`tiny`) | library ground-truth checks + env + datasets |
| train | `run_train.sbatch` | 104 array | checkpoints + `results/train/*.json` (E5 targets) |
| measure | `run_measure.sbatch` | manifest-sized array | `results/measure/*.json`: pairwise quotient matrices, both one-sided p-values, guard, fold stats, baselines, E2 triple scans |
| e1 | `run_cpu_singles.sbatch e1` | 1 | MNIST-at-scale matrix + confusion correlation + stability sweep |
| e4 | `run_cpu_singles.sbatch e4` | 1 (after measure) | Type-I calibration (both tails + sign-flip) + guard audit |
| aggregate | inline `--wrap` | 1 (after measure) | `e3_dynamics.json` (paired epoch tests, BH) + `e5_predict.json` (LOO R², Spearman vs baselines) |

Model roster (`experiments/common.py`): 5× ResNet-20/CIFAR-10 seeds with dense
checkpoints (E3), ResNet-56 on CIFAR-10 + CIFAR-100-subset and a small ViT
(E2), and the 96-model population grid depth×width×wd×aug×seed (E5).

## One-time setup (login node)

```bash
git clone https://github.com/sushovan4/disentanglement.git
cd disentanglement
module load anaconda/2023.03
conda create -n mixup-env -y python=3.10
source activate mixup-env
# GPU route (recommended -- Pegasus has a 7-day `gpu` partition):
python -m pip install numpy scipy scikit-learn gudhi pandas torch torchvision
# CPU-only alternative:
#   python -m pip install numpy scipy scikit-learn gudhi pandas \
#          torch torchvision --index-url https://download.pytorch.org/whl/cpu \
#          --extra-index-url https://pypi.org/simple
python -c "import gudhi, torch, torchvision, sklearn, scipy; print('env OK')"
```

With the CUDA wheel, submit training on GPUs via
`bash cluster/submit_all.sh --gpu` (uses `run_train_gpu.sbatch`: ~25 min per
ResNet-20 vs ~6 h on 16 CPU threads); training auto-detects CUDA either way.

**Compute-node note** (same as pleb): `source activate` does not fix `PATH` on
compute nodes; the sbatch scripts call
`$HOME/.conda/envs/mixup-env/bin/python` by absolute path.

## Run

```bash
cd ~/disentanglement
git pull
bash cluster/submit_all.sh        # smoke → train → measure → e4/aggregate
bash cluster/monitor.sh
```

Partial reruns: `submit_all.sh --measure-only` re-submits measurement over
existing checkpoints; failed array indices can be resubmitted with
`sbatch --array=<idx> cluster/run_train.sbatch` (results are per-task files,
nothing is clobbered).

## Knobs

- `DISENTANGLE_B` (env): permutations per sweep test (default 199; E4 and
  triple-scan escalation use 999 internally).
- `experiments/common.py`: PCA_DIM=5, POINTS_PER_CLASS=200, N_FOLDS=8 — the
  paper's protocol constants, in one place.

## Total-mixup baseline (E5)

Wagner et al.'s total mixup is deliberately **not** re-implemented in
`baselines.py`; vendor their reference implementation into
`vendor/` and add its per-layer score to `measure.py` before the
final E5 run, so the baseline is theirs rather than our re-reading of it.

## After completion

```bash
bash cluster/monitor.sh                          # confirm arrays COMPLETED
git checkout -b disentangle-results
git add results/ && git commit -m "disentangle: cluster results E1-E5"
git push -u origin disentangle-results
```

Figures are generated locally from `results/*.json` (plotting scripts to come
with the figure pass on the paper).
