"""E1 on the density-offset diagonal: re-run the MNIST reference matrix
(m=200/class, d0=5, B=199, seed 0) with per-cloud density offsets
(ecp.cloud_offsets) and compare with the diagonal reference in
results/e1_mnist.json: confusability Spearman, certified pairs, and the
agreement of the two quotient matrices.  Writes results/e1_adaptive.json.

Usage: python -m experiments.e1_adaptive
"""
import json
import os

import numpy as np
from scipy.stats import spearmanr

from .common import RESULTS_DIR, dump_json
from .data import mnist_numpy
from .e1_mnist import PAIRS, clouds_for
from .ecp import cloud_offsets, default_r_grid, permutation_test


def _job(args):
    (c1, c2), clouds, seed = args
    r_grid = default_r_grid(clouds)
    res = permutation_test(clouds, r_grid, B=199, rng=seed, adaptive=True)
    w = cloud_offsets(clouds)
    return f"{c1},{c2}", {"quotient": res["quotient"], "p_up": res["p_up"], "p_down": res["p_down"],
                          "T_obs": res["T_obs"], "offset": float(max(w[0][0], w[1][0]))}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default=None, help="K/N: run the K-th of N interleaved slices of the 45 pairs serially")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    parts = os.path.join(RESULTS_DIR, "e1_adaptive_parts"); os.makedirs(parts, exist_ok=True)
    if a.merge:
        ada = {}
        for f in sorted(os.listdir(parts)):
            ada.update(json.load(open(os.path.join(parts, f))))
        assert len(ada) == len(PAIRS), len(ada)
    else:
        rng = np.random.default_rng(0)
        Xtr, ytr = mnist_numpy(train=True)
        jobs = [((c1, c2), clouds_for(Xtr, ytr, c1, c2, 200, 5, rng), int(rng.integers(2**32))) for c1, c2 in PAIRS]
        k, n = map(int, a.shard.split("/")) if a.shard else (0, 1)
        ada = dict(_job(j) for j in jobs[k::n])
        dump_json(ada, os.path.join(parts, f"{k}.json")); print("shard", k, "done", len(ada), "pairs", flush=True)
        return
    ref = json.load(open(os.path.join(RESULTS_DIR, "e1_mnist.json")))
    conf = ref["confusion_pairs"]; keys = [f"{a},{b}" for a, b in PAIRS]
    qa = np.array([ada[k]["quotient"] for k in keys]); qr = np.array([ref["reference"][k]["quotient"] for k in keys])
    cf = np.array([conf[k] for k in keys])
    out = {"adaptive": ada,
           "spearman_vs_confusion": float(spearmanr(qa, cf)[0]), "reference_spearman_vs_confusion": float(spearmanr(qr, cf)[0]),
           "spearman_adaptive_vs_reference": float(spearmanr(qa, qr)[0]),
           "certified_adaptive": int(sum(ada[k]["p_down"] <= 0.05 for k in keys)),
           "certified_reference": int(sum(ref["reference"][k]["p_down"] <= 0.05 for k in keys)),
           "mean_quotient_adaptive": float(qa.mean()), "mean_quotient_reference": float(qr.mean())}
    dump_json(out, os.path.join(RESULTS_DIR, "e1_adaptive.json"))
    for k, v in out.items():
        if k != "adaptive": print(k, v)


if __name__ == "__main__":
    main()
