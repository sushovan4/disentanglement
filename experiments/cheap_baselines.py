"""Cheap pairwise separability statistics on the SAME clouds the ECP measures.

The reviewers' question: does the topological statistic carry information
beyond two-line alternatives?  For each class pair on the identical
PCA-d0 clouds we compute

  fisher   : ||mu_1 - mu_2||^2 / (tr S_1 + tr S_2)/2      (Fisher-type ratio)
  meandist : ||mu_1 - mu_2|| / pooled RMS radius
  knn      : cross-class fraction among the 5 nearest neighbours (higher = more mixed)
  energy   : energy distance 2E|X-Y| - E|X-X'| - E|Y-Y'|, scaled by E|X-Y|
  probe    : 5-fold CV error of a logistic probe on the d0 clouds (higher = more mixed)

`e1` reproduces the E1 MNIST clouds exactly (same rng stream as e1_mnist.py)
and correlates each statistic with the confusion matrix; `fold` computes the
statistics on a fold's clouds (used by refold.py for the paired tests).

Usage: python -m experiments.cheap_baselines e1
"""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

from .common import RESULTS_DIR, dump_json

KNN_K = 5


def pair_stats(A, B, probe=True, seed=0):
    A = np.asarray(A, float); B = np.asarray(B, float)
    mu = A.mean(0) - B.mean(0)
    within = 0.5 * (A.var(0, ddof=1).sum() + B.var(0, ddof=1).sum())
    out = {"fisher": float(mu @ mu / within),
           "meandist": float(np.linalg.norm(mu) / np.sqrt(within))}
    X = np.vstack([A, B]); y = np.r_[np.zeros(len(A)), np.ones(len(B))]
    _, idx = cKDTree(X).query(X, k=KNN_K + 1)
    out["knn"] = float((y[idx[:, 1:]] != y[:, None]).mean())
    dab = cdist(A, B).mean(); daa = cdist(A, A).mean(); dbb = cdist(B, B).mean()
    out["energy"] = float((2 * dab - daa - dbb) / dab)
    if probe:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        out["probe"] = float(1 - np.mean(cross_val_score(clf, X, y, cv=5)))
    return out


def e1():
    from .data import mnist_numpy
    from .e1_mnist import PAIRS, clouds_for
    Xtr, ytr = mnist_numpy(train=True)
    ref = json.load(open(os.path.join(RESULTS_DIR, "e1_mnist.json")))
    conf = np.array([ref["confusion_pairs"][f"{a},{b}"] for a, b in PAIRS], float)
    quo = np.array([ref["reference"][f"{a},{b}"]["quotient"] for a, b in PAIRS])
    rng = np.random.default_rng(0)             # the E1 stream: clouds then a seed, per pair
    stats = {}
    for c1, c2 in PAIRS:
        A, B = clouds_for(Xtr, ytr, c1, c2, 200, 5, rng); rng.integers(2**32)
        stats[f"{c1},{c2}"] = pair_stats(A, B)
    out = {"per_pair": stats, "spearman_vs_confusion": {}, "spearman_vs_quotient": {}}
    print(f"{'statistic':>10} {'rho vs confusion':>18} {'rho vs ECP quotient':>20}")
    for k in ("fisher", "meandist", "knn", "energy", "probe"):
        v = np.array([stats[f"{a},{b}"][k] for a, b in PAIRS])
        sgn = 1 if k in ("knn", "probe") else -1     # separation statistics anticorrelate with confusion
        r1, p1 = spearmanr(sgn * v, conf); r2, p2 = spearmanr(sgn * v, quo)
        out["spearman_vs_confusion"][k] = {"rho": float(r1), "p": float(p1)}
        out["spearman_vs_quotient"][k] = {"rho": float(r2), "p": float(p2)}
        print(f"{k:>10} {r1:>+18.3f} {r2:>+20.3f}")
    r, p = spearmanr(quo, conf); out["spearman_vs_confusion"]["ecp_quotient"] = {"rho": float(r), "p": float(p)}
    print(f"{'ECP':>10} {r:>+18.3f}")
    dump_json(out, os.path.join(RESULTS_DIR, "e1_cheap_baselines.json"))


if __name__ == "__main__":
    globals()[sys.argv[1] if len(sys.argv) > 1 else "e1"]()
