"""E4 --- Type-I calibration of the tests + cancellation-guard audit.

(a) One-sided permutation tests: 500 null replicates (two clouds drawn from
    the SAME pooled MNIST-feature distribution); empirical rejection rate at
    alpha in {0.01, 0.05, 0.1} must match alpha, both tails.
(b) Sign-flip paired test: two identical conditions measured on disjoint
    folds; rejection rate at alpha must match alpha.
(c) Guard audit: aggregate guard_fired / rstar over every measure.py output
    present in results/measure/.

Usage: python -m experiments.e4_calibration
"""

import glob
import json
import os

import numpy as np
from sklearn.decomposition import PCA

from .common import RESULTS_DIR, dump_json, ensure_dirs
from .data import mnist_numpy
from .ecp import default_r_grid, mixup_ecp, paired_subsample_test, \
    permutation_test, profile_stat

ALPHAS = (0.01, 0.05, 0.10)
N_NULL = 500
M = 100
B = 199


def null_calibration(rng):
    X, y = mnist_numpy(train=True)
    proj = PCA(n_components=5, random_state=0).fit_transform(
        X[rng.choice(len(X), 20000, replace=False)])
    p_up, p_down = [], []
    for i in range(N_NULL):
        idx = rng.choice(len(proj), size=2 * M, replace=False)
        clouds = [proj[idx[:M]], proj[idx[M:]]]
        res = permutation_test(clouds, default_r_grid(clouds), B=B,
                               rng=rng.integers(2**32))
        p_up.append(res["p_up"])
        p_down.append(res["p_down"])
        if (i + 1) % 50 == 0:
            print(f"null replicate {i + 1}/{N_NULL}", flush=True)
    return {f"reject_up@{a}": float(np.mean(np.array(p_up) <= a))
            for a in ALPHAS} | \
           {f"reject_down@{a}": float(np.mean(np.array(p_down) <= a))
            for a in ALPHAS} | \
           {"p_up": p_up, "p_down": p_down}


def signflip_calibration(rng):
    """Identical conditions: per-fold stats from two disjoint half-samples of
    the same distribution; E[D]=0 and D is symmetric, so the sign-flip test
    is under its exact null."""
    X, y = mnist_numpy(train=True)
    proj = PCA(n_components=5, random_state=0).fit_transform(
        X[rng.choice(len(X), 20000, replace=False)])
    ps = []
    for i in range(N_NULL):
        Ta, Tb = [], []
        for fold in range(8):
            idx = rng.choice(len(proj), size=4 * M, replace=False)
            for T, sl in ((Ta, slice(0, 2 * M)), (Tb, slice(2 * M, 4 * M))):
                cl = [proj[idx[sl]][:M], proj[idx[sl]][M:]]
                rg = default_r_grid(cl)
                T.append(profile_stat(mixup_ecp(cl, rg), rg))
        ps.append(paired_subsample_test(Ta, Tb)["p_sign"])
    return {f"reject_sign@{a}": float(np.mean(np.array(ps) <= a))
            for a in ALPHAS} | {"p_sign": ps}


def guard_audit():
    total = fired = 0
    for path in glob.glob(os.path.join(RESULTS_DIR, "measure", "*.json")):
        rec = json.load(open(path))
        for layer in rec.get("layers", {}).values():
            for pair in layer.get("pairs", {}).values():
                total += 1
                fired += bool(pair.get("guard_fired"))
    return {"n_measurements": total, "n_guard_fired": fired,
            "rate": fired / total if total else None}


def main():
    ensure_dirs()
    rng = np.random.default_rng(4)
    out = {"permutation": null_calibration(rng),
           "signflip": signflip_calibration(rng),
           "guard": guard_audit()}
    dump_json(out, os.path.join(RESULTS_DIR, "e4_calibration.json"))
    print({k: v for k, v in out["permutation"].items() if k.startswith("reject")})
    print({k: v for k, v in out["signflip"].items() if k.startswith("reject")})
    print("guard:", out["guard"])


if __name__ == "__main__":
    main()
