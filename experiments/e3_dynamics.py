r"""E3 --- training-dynamics aggregation: paired epoch-to-epoch tests.

Reads the per-(config, epoch) fold statistics written by measure.py for the
five 'e3' seeds and, for every layer x class-pair x consecutive-checkpoint
step, runs the paired subsampled test (same folds through both epochs).
Applies Benjamini-Hochberg within each (seed, layer) family and writes the
trajectory table for the paper's E3 figure.

The fold statistic must be scale-free.  The archived 'folds' are the raw
profile mass \int|Dchi|dr (length units; the r-grid is data-adaptive), and a
paired test on them across epochs certifies feature-norm dynamics as if they
were interaction changes (at epoch 0->1 every significant step was an
"increase" while every quotient fell).  This script therefore uses, in order
of preference:
  1. results/refold/<name>_epoch<e>.json  (experiments.refold: the dimensionless
     \int|Dchi|dr / r_max on the same fold clouds, plus the cheap baselines) --
     one paired analysis per statistic, written to e3_dynamics_<stat>.json;
  2. otherwise the archived raw folds divided by the pair's permutation-null
     mean at that epoch (null_mean = r_max * const, so this is scale-free too),
     written to e3_dynamics_norm.json.
The raw analysis is kept in e3_dynamics.json for the record.

Usage: python -m experiments.e3_dynamics
"""

import glob
import json
import os

import numpy as np

from .common import CONFIGS, RESULTS_DIR, checkpoint_epochs, dump_json
from .ecp import paired_subsample_test


def bh(pvals, q=0.05):
    """Benjamini-Hochberg: boolean rejections at FDR q."""
    p = np.asarray(pvals)
    order = np.argsort(p)
    thresh = q * (np.arange(1, len(p) + 1)) / len(p)
    passed = p[order] <= thresh
    k = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    out = np.zeros(len(p), dtype=bool)
    out[order[:k]] = True
    return out


def _refold(cfg, epochs):
    out = {}
    for e in epochs:
        p = os.path.join(RESULTS_DIR, "refold", f"{cfg.name}_epoch{e}.json")
        if os.path.exists(p):
            out[e] = json.load(open(p))["layers"]
    return out


def paired_sweep(stat_of, label):
    """stat_of(cfg, e0, e1, layer, pair) -> (T0_folds, T1_folds) or None."""
    rows = []
    for cfg in [c for c in CONFIGS if c.tag in ("e3", "ablr")]:
        eps = checkpoint_epochs(cfg)
        per = {}
        for e in eps:
            path = os.path.join(RESULTS_DIR, "measure", f"{cfg.name}_epoch{e}.json")
            if os.path.exists(path):
                per[e] = json.load(open(path))
        avail = sorted(per)
        for e0, e1 in zip(avail, avail[1:]):
            for layer in per[e0]["layers"]:
                fam = []
                for pair in per[e0]["layers"][layer]["folds"]:
                    got = stat_of(cfg, per, e0, e1, layer, pair)
                    if got is None:
                        continue
                    T0, T1 = got
                    res = paired_subsample_test(T1, T0)  # D<0 = disentangling
                    fam.append({"config": cfg.name, "seed": cfg.seed, "sched": cfg.lr_schedule,
                                "layer": layer, "pair": pair, "epoch_from": e0, "epoch_to": e1,
                                **{k: res[k] for k in ("mean_diff", "p_sign", "ci95")}})
                if fam:
                    for r, rj in zip(fam, bh([r["p_sign"] for r in fam])):
                        r["bh_reject"] = bool(rj)
                    rows += fam
    n_sig = sum(1 for r in rows if r["bh_reject"]); n_dec = sum(1 for r in rows if r["bh_reject"] and r["mean_diff"] < 0)
    print(f"E3 [{label}]: {len(rows)} tests, {n_sig} significant, {n_dec} decreases")
    return rows


def main_scale_free():
    """The paired analyses the paper reports (see module docstring)."""
    refold = {cfg.name: _refold(cfg, checkpoint_epochs(cfg)) for cfg in CONFIGS if cfg.tag in ("e3", "ablr")}
    have = any(refold.values())
    if have:
        for stat in ("ecp", "raw", "fisher", "meandist", "knn", "energy"):
            def stat_of(cfg, per, e0, e1, layer, pair, stat=stat):
                R = refold[cfg.name]
                if e0 not in R or e1 not in R:
                    return None
                return R[e0][layer][pair][stat], R[e1][layer][pair][stat]
            rows = paired_sweep(stat_of, f"refold:{stat}")
            dump_json(rows, os.path.join(RESULTS_DIR, f"e3_dynamics_{stat}.json"))
    def stat_null(cfg, per, e0, e1, layer, pair):
        L0, L1 = per[e0]["layers"][layer], per[e1]["layers"][layer]
        if pair not in L1["folds"]:
            return None
        return (np.array(L0["folds"][pair]) / L0["pairs"][pair]["null_mean"],
                np.array(L1["folds"][pair]) / L1["pairs"][pair]["null_mean"])
    rows = paired_sweep(stat_null, "raw folds / permutation-null mean")
    dump_json(rows, os.path.join(RESULTS_DIR, "e3_dynamics_norm.json"))


def main():
    rows = []
    for cfg in [c for c in CONFIGS if c.tag in ("e3", "ablr")]:
        eps = checkpoint_epochs(cfg)
        per_epoch = {}
        for e in eps:
            path = os.path.join(RESULTS_DIR, "measure",
                                f"{cfg.name}_epoch{e}.json")
            if not os.path.exists(path):
                print(f"missing {path}; run measure first")
                continue
            per_epoch[e] = json.load(open(path))
        avail = sorted(per_epoch)
        for e0, e1 in zip(avail, avail[1:]):
            L0, L1 = per_epoch[e0]["layers"], per_epoch[e1]["layers"]
            for layer in L0:
                fam = []
                for pair, T0 in L0[layer]["folds"].items():
                    T1 = L1[layer]["folds"].get(pair)
                    if T1 is None:
                        continue
                    res = paired_subsample_test(T1, T0)  # D<0 = disentangling
                    fam.append({"config": cfg.name, "seed": cfg.seed, "layer": layer,
                                "pair": pair, "epoch_from": e0,
                                "epoch_to": e1, **{k: res[k] for k in
                                ("mean_diff", "p_sign", "ci95")}})
                if fam:
                    rej = bh([r["p_sign"] for r in fam])
                    for r, rj in zip(fam, rej):
                        r["bh_reject"] = bool(rj)
                    rows += fam
        # quotient trajectories for the figure
        for e in avail:
            for layer, rec in per_epoch[e]["layers"].items():
                quos = [v["quotient"] for v in rec["pairs"].values()]
                rows.append({"config": cfg.name, "seed": cfg.seed, "layer": layer, "epoch": e,
                             "mean_quotient": float(np.mean(quos)),
                             "kind": "trajectory"})
    dump_json(rows, os.path.join(RESULTS_DIR, "e3_dynamics.json"))
    n_sig = sum(1 for r in rows if r.get("bh_reject"))
    print(f"E3 aggregated: {len(rows)} rows, {n_sig} significant steps")


if __name__ == "__main__":
    main()              # the raw-mass analysis, kept for the record (e3_dynamics.json)
    main_scale_free()   # the analyses the paper reports
