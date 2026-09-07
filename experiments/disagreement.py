"""Does the topological quotient carry information the cheap statistics do
not?  Score, per (model, layer) cell of the E5 population, how far the
interaction quotient departs from what the 5-NN cross-class rate (and the
Fisher ratio) predict, and test the departure as a predictor of test accuracy,
generalization gap, calibration (ECE, NLL) and corruption robustness, alone and
on top of the E5 baselines (linear probe, NC1, CKA).

Inputs: results/measure/pop_*_epoch80.json (quotients, probe_acc, nc1, cka),
        results/refold/pop_*_epoch80.json (scale-free ecp, knn, fisher, energy
        per fold), results/train/pop_*.json, results/robust/pop_*.json.
Output: results/disagreement.json and a printed table.

Usage: python -m experiments.disagreement
"""
import glob
import json
import os

import numpy as np
from scipy.stats import spearmanr

from .common import RESULTS_DIR, dump_json
from .e5_predict import evaluate_family

LAYERS = ["stem", "stage1", "stage2", "stage3"]   # penult == stage3 clouds


def cell_means(name):
    M = json.load(open(os.path.join(RESULTS_DIR, "measure", f"{name}_epoch80.json")))["layers"]
    R = json.load(open(os.path.join(RESULTS_DIR, "refold", f"{name}_epoch80.json")))["layers"]
    out = {}
    for l in LAYERS:
        pairs = R[l]
        out[l] = {"q": float(np.mean([M[l]["pairs"][p]["quotient"] for p in pairs])),
                  "ecp": float(np.mean([np.mean(pairs[p]["ecp"]) for p in pairs])),
                  "knn": float(np.mean([np.mean(pairs[p]["knn"]) for p in pairs])),
                  "fisher": float(np.mean([np.mean(pairs[p]["fisher"]) for p in pairs])),
                  "energy": float(np.mean([np.mean(pairs[p]["energy"]) for p in pairs])),
                  "probe_acc": M[l].get("probe_acc"), "nc1": M[l].get("nc1"), "cka": M[l].get("cka_to_penult")}
    return out


def main():
    names = sorted(os.path.basename(f)[:-len("_epoch80.json")] for f in glob.glob(os.path.join(RESULTS_DIR, "refold", "pop_*_epoch80.json")))
    rows = []
    for n in names:
        tr = json.load(open(os.path.join(RESULTS_DIR, "train", f"{n}.json")))
        rb_path = os.path.join(RESULTS_DIR, "robust", f"{n}.json")
        rb = json.load(open(rb_path)) if os.path.exists(rb_path) else {}
        rows.append({"name": n, "cells": cell_means(n), "test_acc": tr["test_acc"], "gen_gap": tr["gen_gap"],
                     "ece": rb.get("ece"), "nll": rb.get("nll"), "acc_corrupt": rb.get("acc_corrupt_mean")})
    print(f"{len(rows)} models")
    # robustness and calibration residualized on clean accuracy: the part of each
    # target that clean accuracy (hence the linear probe) does not already explain
    acc = np.array([r["test_acc"] for r in rows])
    for t in ("acc_corrupt", "ece"):
        if all(r[t] is not None for r in rows):
            v = np.array([r[t] for r in rows]); res = v - np.polyval(np.polyfit(acc, v, 1), acc)
            for r, x in zip(rows, res):
                r[f"{t}_res"] = float(x)
    # residual of the quotient given the neighbour rate, per layer, across models
    for l in LAYERS:
        q = np.array([r["cells"][l]["q"] for r in rows]); k = np.array([r["cells"][l]["knn"] for r in rows])
        f = np.array([r["cells"][l]["fisher"] for r in rows])
        for key, x in (("knn", k), ("fisher", np.log(f + 1e-9))):
            A = np.vstack([x, np.ones_like(x)]).T
            coef, *_ = np.linalg.lstsq(A, q, rcond=None); res = q - A @ coef
            for r, v in zip(rows, res):
                r["cells"][l][f"res_{key}"] = float(v)
        print(f"{l}: corr(q, knn) = {spearmanr(q, k)[0]:+.2f}, corr(q, fisher) = {spearmanr(q, f)[0]:+.2f}, sd(res_knn) = {np.std(q - np.polyval(np.polyfit(k, q, 1), k)):.3f}")
    # correlations of residuals and raw cell statistics with every target
    targets = [t for t in ("test_acc", "gen_gap", "ece", "nll", "acc_corrupt", "acc_corrupt_res", "ece_res") if all(r.get(t) is not None for r in rows)]
    table = {}
    print("\nSpearman of per-layer statistic with target (n = %d):" % len(rows))
    print(f"{'stat':22s}" + "".join(f"{t:>12s}" for t in targets))
    for l in LAYERS:
        for key in ("q", "knn", "fisher", "res_knn", "res_fisher", "probe_acc"):
            x = [r["cells"][l][key] for r in rows]
            if any(v is None for v in x):
                continue
            cs = {t: float(spearmanr(x, [r[t] for r in rows])[0]) for t in targets}
            table[f"{l}/{key}"] = cs
            print(f"{l+'/'+key:22s}" + "".join(f"{cs[t]:+12.2f}" for t in targets))
    # leave-one-out ridge: do the residuals add to the E5 baselines?
    fams = {}
    def feats(keys):
        return [{f"{l}/{k}": r["cells"][l][k] for l in LAYERS for k in keys if r["cells"][l].get(k) is not None} for r in rows]
    base = feats(("probe_acc", "nc1", "cka"))
    resid = feats(("res_knn", "res_fisher"))
    cheap = feats(("knn", "fisher", "energy"))
    quot = feats(("q",))
    print("\nLeave-one-out ridge R^2:")
    for t in targets:
        y = [r[t] for r in rows]
        for lab, X in (("baselines", base), ("cheap", cheap), ("quotient", quot), ("residuals", resid),
                       ("baselines+residuals", [a | b for a, b in zip(base, resid)]),
                       ("cheap+quotient", [a | b for a, b in zip(cheap, quot)])):
            r2 = evaluate_family(X, y, lab)
            fams[f"{t}/{lab}"] = r2
            print(f"  {t:12s} {lab:22s} R2 = {r2['loo_r2']:+.3f}  rho = {r2['spearman']:+.2f}")
    dump_json({"rows": rows, "spearman": table, "ridge": fams}, os.path.join(RESULTS_DIR, "disagreement.json"))


if __name__ == "__main__":
    main()
