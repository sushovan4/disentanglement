# Certified topological interaction in neural representations

Code and measurement records for

> Sushovan Majhi. *Certified Topological Interaction in Neural Representations:
> Class Disentanglement Is Mostly Pairwise.* 2026. (arXiv identifier to follow.)

The paper measures class disentanglement in trained networks with the
Intersection Euler Characteristic Profile of Kawamura, Majhi and Mitra
([arXiv:2608.06180](https://arxiv.org/abs/2608.06180)), attaches an exact
permutation certificate to every number, and runs the protocol over 111
networks and 52,650 certified pairwise measurements. This repository holds
the experiment pipeline, the Slurm scripts that ran it, and every measurement
record behind the reported numbers, so that each figure and table rebuilds
from the records without retraining anything.

## Layout

| Path | Contents |
|---|---|
| `experiments/` | The pipeline, one module per stage; run as `python -m experiments.<module>` from the repository root. |
| `cluster/` | Slurm array scripts and drivers that produced the records (written for a GW Pegasus allocation; edit `--account` and the partition names). |
| `results/` | 893 measurement records (JSON) behind every number in the paper. |
| `figures/` | The paper's figures and tables, as rebuilt from `results/` by `experiments.figures`, `experiments.intro_figures`, and `experiments.factorial`; `concentration_table.tex` is transcribed from `results/ambient_concentration.json` and `results/depth_vs_dim.json`. |
| `vendor/` | Instructions for cloning the GPL-licensed reference implementation of the mixup barcode used as a baseline (not redistributed here). |
| `data/depth_strip.npz` | The one cached feature file needed by the introduction figure. |

Protocol constants live in one place, `experiments/common.py`: PCA dimension
`d0 = 5` (`4` for triple scans), `m = 200` points per class, `B = 199`
permutations per sweep test (`999` for escalations), `F = 8` folds for the
paired tests, and the scale-free fold statistic. Every trainable model in the
paper is a row of `CONFIGS` in the same file; cluster array tasks index into
it.

## Environment

Python 3.10 or later.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m experiments.smoke      # ground-truth checks on the ECP machinery
```

Only `numpy`, `scipy`, `scikit-learn`, `gudhi`, and `matplotlib` are needed
to rebuild figures and tables from the records or to run E1 and E4 on a
laptop. `torch` and `torchvision` are needed to retrain or re-extract
features, and `transformers` and `datasets` only for the frozen language
models of E10. The Alpha complexes come from GUDHI; if a recent GUDHI wheel
refuses NumPy 2 on your platform, pin `numpy<2`.

## Rebuild the paper's figures and tables from the records

```bash
python -m experiments.figures                 # E1, E2, E3, E4, E7 figures and the E5 table
python -m experiments.factorial               # E11 table (figures/factorial_table.tex)
python -m experiments.intro_figures all       # the three introduction figures
python -m experiments.init_dominance report   # E9 and E10 dominance cells
python -m experiments.e8_d0 --report          # E8 projection-dimension summary
python -m experiments.guard_audit --report    # E4 guard re-evaluation
python -m experiments.e9_grok --report        # Appendix E grokking numbers
```

Each command reads `results/` only and finishes in seconds to a few minutes.

## Experiments, scripts, and records

| Paper | Scripts | Records |
|---|---|---|
| E1 MNIST confusability | `e1_mnist.py`, `e1_adaptive.py` (density-offset variant), `cheap_baselines.py` | `results/e1_mnist.json`, `e1_adaptive.json`, `e1_cheap_baselines.json` |
| E2 depth, triple scans | `train.py`, `measure.py`, `cluster/run_followup_extras.sbatch` (deep rescans at `d0 = 5`), `dominance_search.py` (adversarial geometry), `dominance_null.py` (null floor) | `results/measure/`, `dominance_search.json`, `dominance_null.json` |
| E3 training dynamics, cost | `train.py`, `measure.py`, `refold.py` (scale-free fold statistic and cheap statistics on the same folds), `e3_dynamics.py`, `wagner_baseline.py` | `results/refold/`, `e3_dynamics*.json`, `results/wagner/` |
| E4 calibration, guard | `e4_calibration.py`, `guard_audit.py` | `e4_calibration.json`, `results/guard_audit/`, `guard_audit.json` |
| E5 generalization | `baselines.py` (inside `measure.py`), `e5_predict.py`, `disagreement.py`, `robust_eval.py` | `results/train/`, `results/robust/`, `e5_predict.json`, `disagreement.json` |
| E6 differentiable surrogate | `e6_regularize.py`, `train.py` with `dis_lambda` | `results/e6/`, `results/e6_detached/`, `e6_regularize.json` |
| E7 interaction spectrum | `e7_spectrum.py` | `results/e7/` |
| E8 projection dimension | `e8_d0.py` | `results/e8/`, `e8_d0_report.json` |
| E9 initialization, memorization | `init_dominance.py`, `dominance_null.py`, `train.py` with `random_labels` | `results/init_dominance/`, `init_dominance_report.json` |
| E10 frozen language models | `foundation.py` (feature extraction), `init_dominance.py measure --npz` | `results/init_dominance/{bert_base,gpt2}*.json` |
| E11 factorial | `factorial.py` | `factorial.json`, `figures/factorial_table.tex` |
| Appendix A.5 concentration | `ambient.py`, `hd.py`, `validate_hd.py`, `depth_vs_dim.py`, `foundation.py` | `ambient_concentration.json`, `results/ambient_onsets/`, `depth_vs_dim.json` |
| Appendix E grokking | `grok.py`, `e9_grok.py` | `results/grok/`, `results/e9_grok/`, `e9_grok_report.json` |

The core machinery is `experiments/ecp.py`: the diagonal Intersection ECP by
one sorted Alpha-complex sweep, both one-sided permutation tests, the
first-interaction-scale guard and its plateau form, the interaction quotient,
and the paired subsampled sign-flip test.

## Rerun the campaign

Training and measurement ran as Slurm arrays; `cluster/README.md` describes
the pipeline and `cluster/submit_all.sh` chains it (smoke test, 104 training
tasks, the measurement array, then E4 and the aggregations). Datasets are
fetched on a login node with `python -m experiments.data`, since compute
nodes have no network. Checkpoints and cached features are not distributed;
they regenerate from the seeded configurations in `experiments/common.py`.
The total-mixup baseline and the wall-clock comparison call the authors'
reference implementation, which is GPL-licensed and therefore cloned rather
than vendored; `vendor/README.md` has the two portability patches it needs.

## License

MIT for the code in this repository. The mixup-barcode reference
implementation referred to in `vendor/` is distributed by its authors under
GPL-3.0.
