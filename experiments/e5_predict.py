"""E5 --- exploratory generalization prediction over the model population.

Features per model (96 'pop' configs, final epoch): per-layer mean/max
pairwise interaction quotient, count of significantly entangled pairs
(p_up, BH), depth-slope of the mean quotient; baselines: per-layer linear
probe accuracy, NC1, CKA-to-penult.  Target: held-out test accuracy (and
generalization gap).  Model: ridge regression, leave-one-out R^2 +
Spearman; feature families compared separately and jointly.

Usage: python -m experiments.e5_predict
"""

import glob
import json
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler

from .common import CONFIGS, RESULTS_DIR, checkpoint_epochs, dump_json


def wagner_features(name, epoch):
    """Total-mixup baseline features from the vendored reference run (G1)."""
    path = os.path.join(RESULTS_DIR, "wagner", f"{name}_epoch{epoch}.json")
    if not os.path.exists(path):
        return None
    rec = json.load(open(path))
    wag = {}
    for layer in sorted(rec["layers"]):
        L = rec["layers"][layer]
        wag[f"{layer}_wag_meanperc"] = L["mean_mixup_perc_sym"]
        tm = [np.mean([p["fwd"]["dim0"]["total_mixup_perc"],
                       p["bwd"]["dim0"]["total_mixup_perc"]])
              for p in L["pairs"].values()]
        wag[f"{layer}_wag_totperc0"] = float(np.mean(tm))
    return wag


def model_features(name, epoch):
    path = os.path.join(RESULTS_DIR, "measure", f"{name}_epoch{epoch}.json")
    if not os.path.exists(path):
        return None
    rec = json.load(open(path))
    ecp, base, raw = {}, {}, {}
    layers = sorted(rec["layers"])
    means = []
    for layer in layers:
        L = rec["layers"][layer]
        quos = np.array([v["quotient"] for v in L["pairs"].values()])
        pups = np.array([v["p_up"] for v in L["pairs"].values()])
        ecp[f"{layer}_mean_q"] = float(np.mean(quos))
        ecp[f"{layer}_max_q"] = float(np.max(quos))
        ecp[f"{layer}_n_sig"] = int(np.sum(pups <= 0.05))
        means.append(np.mean(quos))
        # Unnormalized profile mass and first-interaction scale: the absolute
        # overlap geometry the quotient divides out. Tests the explanation
        # offered for total mixup out-predicting the quotient.
        tobs = np.array([v["T_obs"] for v in L["pairs"].values()])
        rstar = np.array([v["rstar"] for v in L["pairs"].values()])
        raw[f"{layer}_mean_T"] = float(np.mean(tobs))
        raw[f"{layer}_max_T"] = float(np.max(tobs))
        raw[f"{layer}_mean_rstar"] = float(np.mean(rstar))
        for k_src, k_dst in (("probe_acc", "probe"), ("nc1", "nc1"),
                             ("cka_to_penult", "cka")):
            if k_src in L:
                base[f"{layer}_{k_dst}"] = L[k_src]
    if len(means) >= 2:
        ecp["depth_slope"] = float(np.polyfit(range(len(means)), means, 1)[0])
    return ecp, base, raw


def evaluate_family(Xd, yv, label):
    keys = sorted(set(k for d in Xd for k in d))
    X = np.array([[d.get(k, np.nan) for k in keys] for d in Xd])
    X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
    X = StandardScaler().fit_transform(X)
    y = np.asarray(yv)
    pred = cross_val_predict(RidgeCV(alphas=np.logspace(-3, 3, 13)), X, y,
                             cv=LeaveOneOut())
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return {"family": label, "n": len(y), "loo_r2": 1 - ss_res / ss_tot,
            "spearman": float(spearmanr(pred, y)[0]), "n_features": len(keys)}


def main():
    rows_ecp, rows_base, rows_both, acc, gap = [], [], [], [], []
    rows_wag, rows_raw = [], []
    for cfg in [c for c in CONFIGS if c.tag == "pop"]:
        tpath = os.path.join(RESULTS_DIR, "train", f"{cfg.name}.json")
        if not os.path.exists(tpath):
            continue
        ep = checkpoint_epochs(cfg)[-1]
        feats = model_features(cfg.name, ep)
        if feats is None:
            continue
        ecp, base, raw = feats
        t = json.load(open(tpath))
        rows_ecp.append(ecp)
        rows_base.append(base)
        rows_both.append(ecp | base)
        rows_raw.append(raw)
        rows_wag.append(wagner_features(cfg.name, ep))
        acc.append(t["test_acc"])
        gap.append(t["gen_gap"])

    if len(acc) < 10:
        print(f"only {len(acc)} models measured; run train + measure first")
        return
    families = [("ecp", rows_ecp), ("baselines", rows_base),
                ("ecp+baselines", rows_both),
                ("ecp-raw", rows_raw),
                ("ecp+ecp-raw", [e | r for e, r in zip(rows_ecp, rows_raw)]),
                ("baselines+ecp-raw", [b | r for b, r in zip(rows_base, rows_raw)])]
    if all(w is not None for w in rows_wag):
        families += [("total-mixup", rows_wag),
                     ("baselines+total-mixup",
                      [b | w for b, w in zip(rows_base, rows_wag)])]
    else:
        n_missing = sum(w is None for w in rows_wag)
        print(f"wagner baseline missing for {n_missing}/{len(rows_wag)} models; "
              "total-mixup families skipped (run experiments.wagner_baseline)")
    out = {"n_models": len(acc), "results": []}
    for target, tv in (("test_acc", acc), ("gen_gap", gap)):
        for fam, X in families:
            r = evaluate_family(X, tv, fam) | {"target": target}
            out["results"].append(r)
            print(r)
    dump_json(out, os.path.join(RESULTS_DIR, "e5_predict.json"))


if __name__ == "__main__":
    main()
