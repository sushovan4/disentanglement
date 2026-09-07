"""E1 --- MNIST pairwise entanglement at scale (torch-free, CPU).

(a) Reference pairwise interaction-quotient matrix over the 45 digit pairs
    (m=200/class, d0=5, B=199) with both one-sided p-values;
(b) Spearman correlation of the matrix with the test-set confusion matrix of
    a logistic classifier (symmetrized, off-diagonal entries);
(c) stability sweep: ranking stability of the raw-statistic matrix across
    d0 in {3,4,5,6}, m in {50,100,200,400}, 5 resamples.  (Alpha-complex
    cost caps d0 at 6: a 7-D complex on 400 points already exceeds 10^7
    simplices.)

Parallelizes over pairs via a fork pool (SLURM_CPUS_PER_TASK workers).

Usage: python -m experiments.e1_mnist
"""

import os
from itertools import combinations
from multiprocessing import get_context

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from .common import RESULTS_DIR, dump_json, ensure_dirs
from .data import mnist_numpy
from .ecp import default_r_grid, mixup_ecp, permutation_test, profile_stat

PAIRS = list(combinations(range(10), 2))
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", str(os.cpu_count() or 4)))
SWEEP_D0 = (3, 4, 5, 6)
SWEEP_M = (50, 100, 200, 400)


def clouds_for(X, y, c1, c2, m, d0, rng):
    idx1 = rng.choice(np.flatnonzero(y == c1), size=m, replace=False)
    idx2 = rng.choice(np.flatnonzero(y == c2), size=m, replace=False)
    pooled = np.vstack([X[idx1], X[idx2]])
    proj = PCA(n_components=d0, random_state=0).fit_transform(pooled)
    return [proj[:m], proj[m:]]


def _pair_job(args):
    (c1, c2), clouds, B, seed = args
    r_grid = default_r_grid(clouds)
    if B:
        res = permutation_test(clouds, r_grid, B=B, rng=seed)
        return (c1, c2), {"quotient": res["quotient"], "p_up": res["p_up"],
                          "p_down": res["p_down"], "T_obs": res["T_obs"]}
    return (c1, c2), {"T_obs": profile_stat(mixup_ecp(clouds, r_grid),
                                            r_grid)}


def pair_matrix(X, y, m, d0, rng, B=0):
    """45-entry dict of raw statistics (B=0) or full test results (B>0)."""
    jobs = [((c1, c2), clouds_for(X, y, c1, c2, m, d0, rng), B,
             int(rng.integers(2**32))) for c1, c2 in PAIRS]
    with get_context("fork").Pool(min(N_JOBS, len(jobs))) as pool:
        return dict(pool.imap_unordered(_pair_job, jobs, chunksize=1))


def main():
    ensure_dirs()
    rng = np.random.default_rng(0)
    Xtr, ytr = mnist_numpy(train=True)
    Xte, yte = mnist_numpy(train=False)

    # (a) reference matrix with tests
    ref = pair_matrix(Xtr, ytr, m=200, d0=5, rng=rng, B=199)

    # (b) confusion baseline: logistic on PCA-50
    pca = PCA(n_components=50, random_state=0).fit(Xtr)
    clf = LogisticRegression(max_iter=1000).fit(pca.transform(Xtr), ytr)
    C = confusion_matrix(yte, clf.predict(pca.transform(Xte)))
    Cs = C + C.T
    conf = np.array([Cs[a, b] for a, b in PAIRS], dtype=float)
    quo = np.array([ref[p]["quotient"] for p in PAIRS])
    rho, pval = spearmanr(quo, conf)

    # (c) stability sweep on raw statistics
    stability = []
    ref_T = np.array([ref[p]["T_obs"] for p in PAIRS])
    for d0 in SWEEP_D0:
        for m in SWEEP_M:
            for rep in range(5):
                mat = pair_matrix(Xtr, ytr, m=m, d0=d0,
                                  rng=np.random.default_rng(1000 + rep))
                T = np.array([mat[p]["T_obs"] for p in PAIRS])
                stability.append({"d0": d0, "m": m, "rep": rep,
                                  "spearman_vs_ref": float(spearmanr(T, ref_T)[0])})

    out = {
        "reference": {f"{a},{b}": v for (a, b), v in ref.items()},
        "confusion_spearman": {"rho": float(rho), "p": float(pval)},
        "confusion_pairs": {f"{a},{b}": float(c)
                            for (a, b), c in zip(PAIRS, conf)},
        "stability": stability,
    }
    dump_json(out, os.path.join(RESULTS_DIR, "e1_mnist.json"))
    print(f"E1 done. Spearman(ECP, confusion) = {rho:.3f} (p={pval:.2g})")


if __name__ == "__main__":
    main()
