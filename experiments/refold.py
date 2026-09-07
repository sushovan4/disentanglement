"""Re-derive the paired-test fold statistics on a SCALE-FREE functional, and
compute the cheap separability baselines on the same fold clouds.

Why: the campaign's fold statistic was the raw profile mass \\int|Dchi|dr,
which has length units.  The r-grid is data-adaptive, so the raw mass tracks
the clouds' scale, and a paired test on it across epochs certifies feature-norm
dynamics (x20 at epoch 0->1) as if they were interaction changes.  The fix is
the dimensionless functional \\int|Dchi|dr / r_max (ecp.profile_stat 'intnorm').

For each config (tags e3, ablr, e2 by default), each checkpoint epoch, each
layer, each class pair and each of the N_FOLDS disjoint folds, on the very
clouds measure.py used (same PCA, same fold cut), we record
    ecp     : \\int|Dchi|dr / r_max
    raw     : \\int|Dchi|dr                (reproduces the archived 'folds')
    fisher, meandist, knn, energy : cheap_baselines.pair_stats (no probe)
Output: results/refold/<name>_epoch<e>.json.  e3_dynamics.py picks these up.

Usage (cluster):  python -m experiments.refold --tags e3 ablr e2 --jobs 8
"""
import argparse
import json
import os
from itertools import combinations
from multiprocessing import get_context

import numpy as np

from .cheap_baselines import pair_stats
from .common import CONFIGS, N_FOLDS, PCA_DIM, RESULTS_DIR, checkpoint_epochs, dump_json
from .ecp import default_r_grid, mixup_ecp, profile_stat
from .extract import class_clouds, feature_path, load_features
from .measure import FOLD_M_MAX

OUT = os.path.join(RESULTS_DIR, "refold")


def _job(args):
    key, clouds_by_fold = args
    rec = {"ecp": [], "raw": [], "fisher": [], "meandist": [], "knn": [], "energy": []}
    for A, B in clouds_by_fold:
        g = default_r_grid([A, B]); d = mixup_ecp([A, B], g)
        rec["ecp"].append(profile_stat(d, g, "intnorm")); rec["raw"].append(profile_stat(d, g, "int"))
        for k, v in pair_stats(A, B, probe=False).items():
            rec[k].append(v)
    return key, rec


def refold_config_epoch(cfg, epoch, jobs):
    out_path = os.path.join(OUT, f"{cfg.name}_epoch{epoch}.json")
    if os.path.exists(out_path):
        return
    if not os.path.exists(feature_path(cfg, epoch)):
        print("no features for", cfg.name, epoch); return
    feats, labels = load_features(cfg, epoch)
    classes = sorted(np.unique(labels).tolist())
    fold_m = min(FOLD_M_MAX, min(int(np.sum(labels == c)) for c in classes) // N_FOLDS)
    work = []
    for layer, F in feats.items():
        for c1, c2 in combinations(classes, 2):
            cl = [class_clouds(F, labels, [c1, c2], m=fold_m, d0=PCA_DIM, fold=f) for f in range(N_FOLDS)]
            work.append(((layer, f"{c1},{c2}"), cl))
    res = {"name": cfg.name, "epoch": epoch, "fold_m": fold_m, "n_folds": N_FOLDS, "layers": {}}
    with get_context("fork").Pool(jobs) as pool:
        for (layer, pair), rec in pool.imap_unordered(_job, work, chunksize=4):
            res["layers"].setdefault(layer, {})[pair] = rec
    os.makedirs(OUT, exist_ok=True); dump_json(res, out_path)
    print("wrote", out_path, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["e3", "ablr", "e2"])
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--final-only", action="store_true", help="only the last checkpoint epoch of each config")
    ap.add_argument("--shard", default=None, help="K/N: take the K-th of N interleaved slices of the config list")
    a = ap.parse_args()
    cfgs = [c for c in CONFIGS if c.tag in a.tags]
    if a.shard:
        k, n = map(int, a.shard.split("/")); cfgs = cfgs[k::n]
    for cfg in cfgs:
        eps = checkpoint_epochs(cfg)
        for e in (eps[-1:] if a.final_only else eps):
            refold_config_epoch(cfg, e, a.jobs)


if __name__ == "__main__":
    main()
