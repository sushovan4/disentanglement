"""E6 --- differentiable disentanglement regularizer: does steering work?

The E6 configs train ResNet-20/CIFAR-10 at the E3 recipe with the
differentiable surrogate of train.disentangle_reg at strengths
dis_lambda in {0, 0.5, 2, 8} (dl=0 = matched control), 2 seeds each.

This script closes the loop with the paper's OWN certified measure: for
each model it computes the penultimate-layer mean interaction quotient with
the exact ECP permutation test (measure.measure_pair, the same protocol as
E1-E8), and tabulates it against clean test accuracy and the Gaussian-noise
robustness proxy recorded at training time.  The claim E6 can support is
narrow and honest: the differentiable surrogate provably reduces the
*certified* (non-differentiable) entanglement, at a measured accuracy cost.

Usage: python -m experiments.e6_regularize
Outputs results/e6_regularize.json.
"""

import json
import os
from itertools import combinations

import numpy as np

from .common import CONFIGS, PCA_DIM, POINTS_PER_CLASS, RESULTS_DIR, dump_json
from .common import checkpoint_epochs
from .extract import class_clouds, extract_checkpoint, load_features
from .measure import B_SWEEP, _run_jobs
from .measure import measure_pair  # noqa: F401  (kept explicit for provenance)
import zlib


def degeneracy(F):
    """Why a model's penultimate features cannot be measured, or None.

    The first corrected-campaign aggregation died inside np.quantile on an
    empty distance array: the lambda=8 model's features had collapsed, so
    every pairwise distance was zero (or non-finite) and the r-grid had
    nothing to span. Record that as a result rather than crashing on it."""
    if not np.isfinite(F).all():
        return "non-finite features"
    distinct = len(np.unique(np.round(F.astype(np.float64), 6), axis=0))
    if distinct < 50:
        return f"collapsed: {distinct} distinct feature vectors"
    if float(np.std(F)) < 1e-8:
        return "collapsed: zero variance"
    return None


def penult_quotient(cfg, epoch):
    """Certified mean pairwise interaction quotient at the penult layer.

    Returns (mean quotient or None, certified pairs, n pairs, degeneracy)."""
    extract_checkpoint(cfg, epoch)
    feats, labels = load_features(cfg, epoch)
    F = feats["penult"]
    why = degeneracy(F)
    if why is not None:
        n_pairs = len(list(combinations(sorted(np.unique(labels).tolist()), 2)))
        return None, 0, n_pairs, why
    classes = sorted(np.unique(labels).tolist())
    m = min(POINTS_PER_CLASS, min(int(np.sum(labels == c)) for c in classes))
    jobs = []
    for c1, c2 in combinations(classes, 2):
        clouds = class_clouds(F, labels, [c1, c2], m=m, d0=PCA_DIM)
        jobs.append((f"{c1},{c2}", clouds, B_SWEEP,
                     zlib.crc32(f"e6:{cfg.name}:{epoch}:penult:{c1},{c2}".encode())))
    pairs = _run_jobs(jobs)
    quos = [v["quotient"] for v in pairs.values() if v["quotient"] is not None]
    cert = sum(1 for v in pairs.values() if v["p_down"] <= 0.05)
    return float(np.mean(quos)), cert, len(pairs), None


def main():
    rows = []
    cache_dir = os.path.join(RESULTS_DIR, "e6")
    os.makedirs(cache_dir, exist_ok=True)
    for cfg in [c for c in CONFIGS if c.tag == "e6"]:
        tpath = os.path.join(RESULTS_DIR, "train", f"{cfg.name}.json")
        if not os.path.exists(tpath):
            print(f"missing train result: {cfg.name}")
            continue
        # per-model checkpoint: a crash late in the sweep no longer discards
        # every model measured before it, and reruns skip finished models
        cpath = os.path.join(cache_dir, f"{cfg.name}.json")
        if os.path.exists(cpath):
            row = json.load(open(cpath))
            rows.append(row)
            print("cached", row, flush=True)
            continue
        t = json.load(open(tpath))
        q, cert, npairs, why = penult_quotient(cfg, checkpoint_epochs(cfg)[-1])
        row = {"name": cfg.name, "dis_lambda": cfg.dis_lambda, "seed": cfg.seed,
               "test_acc": t["test_acc"], "robust_acc": t.get("robust_acc"),
               "penult_quotient": q, "penult_certified": cert,
               "n_pairs": npairs, "degenerate": why}
        rows.append(row)
        dump_json(row, cpath)
        print(row, flush=True)

    if not rows:
        print("no E6 models trained yet")
        return

    # aggregate by lambda: mean over seeds
    by_lambda = {}
    for r in rows:
        by_lambda.setdefault(r["dis_lambda"], []).append(r)
    summary = []
    for dl in sorted(by_lambda):
        g = by_lambda[dl]
        summary.append({
            "dis_lambda": dl, "n_seeds": len(g),
            "test_acc": float(np.mean([r["test_acc"] for r in g])),
            "robust_acc": float(np.mean([r["robust_acc"] for r in g
                                         if r["robust_acc"] is not None]))
            if any(r["robust_acc"] is not None for r in g) else None,
            "penult_quotient": float(np.mean(qs)) if (qs := [r["penult_quotient"] for r in g
                                                          if r["penult_quotient"] is not None]) else None,
            "n_degenerate": sum(1 for r in g if r.get("degenerate")),
        })
    for s in summary:
        print("SUMMARY", s)
    dump_json({"per_model": rows, "by_lambda": summary},
              os.path.join(RESULTS_DIR, "e6_regularize.json"))


if __name__ == "__main__":
    main()
