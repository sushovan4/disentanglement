"""What changes certified disentanglement: depth, width, weight decay and
augmentation as factors on the E5 population (4 x 2 x 2 x 2 x 3 seeds = 96
ResNets on CIFAR-10, final epoch).

Matched pairs: two models that differ in exactly one factor share the same test
images in the same folds, so the paired sign-flip test of Section 3 applies to
them exactly as it does to two checkpoints.  Statistic: the scale-free fold
profile mass (results/refold), decrease = disentangling.  BH within each
(matched pair, layer) family of 45 class pairs.

Output: results/factorial.json and figures/factorial_table.tex.
Usage: python -m experiments.factorial
"""
import json
import os
from collections import defaultdict
from itertools import combinations

import numpy as np

from .common import CONFIGS, RESULTS_DIR, dump_json
from .e3_dynamics import bh
from .ecp import paired_subsample_test

LAYERS = ["stem", "stage1", "stage2", "stage3"]
FIG = os.path.join(os.path.dirname(RESULTS_DIR), "figures")


def load():
    pop = {}
    for c in CONFIGS:
        if c.tag != "pop":
            continue
        rf = os.path.join(RESULTS_DIR, "refold", f"{c.name}_epoch80.json")
        mf = os.path.join(RESULTS_DIR, "measure", f"{c.name}_epoch80.json")
        tf = os.path.join(RESULTS_DIR, "train", f"{c.name}.json")
        if not (os.path.exists(rf) and os.path.exists(mf)):
            continue
        R = json.load(open(rf))["layers"]; M = json.load(open(mf))["layers"]; T = json.load(open(tf))
        key = (c.depth, c.width, c.weight_decay, c.augment, c.seed)
        pop[key] = {"folds": {l: {p: R[l][p]["ecp"] for p in R[l]} for l in LAYERS},
                    "q": {l: {p: M[l]["pairs"][p]["quotient"] for p in M[l]["pairs"]} for l in LAYERS},
                    "test_acc": T["test_acc"], "name": c.name}
    return pop


def contrast(pop, factor, lo, hi):
    """All matched pairs (lo -> hi in `factor`, everything else equal); per layer:
    certified decreases / increases (BH within the 45-pair family), mean change of
    the scale-free statistic and of the quotient, and the mean test-accuracy change."""
    idx = {"depth": 0, "width": 1, "wd": 2, "aug": 3}[factor]
    out = {l: {"dec": 0, "inc": 0, "tests": 0, "dstat": [], "dq": []} for l in LAYERS}; dacc = []; n = 0
    for key, a in pop.items():
        if key[idx] != lo:
            continue
        kb = list(key); kb[idx] = hi; kb = tuple(kb)
        if kb not in pop:
            continue
        b = pop[kb]; n += 1; dacc.append(b["test_acc"] - a["test_acc"])
        for l in LAYERS:
            fam = []
            for p in a["folds"][l]:
                if p not in b["folds"][l]:
                    continue
                r = paired_subsample_test(b["folds"][l][p], a["folds"][l][p])   # D = hi - lo
                fam.append((r["mean_diff"], r["p_sign"]))
                out[l]["dstat"].append(r["mean_diff"]); out[l]["dq"].append(b["q"][l][p] - a["q"][l][p])
            rej = bh([p for _, p in fam])
            for (d, _), rj in zip(fam, rej):
                out[l]["tests"] += 1
                if rj:
                    out[l]["dec" if d < 0 else "inc"] += 1
    for l in LAYERS:
        out[l]["mean_dstat"] = float(np.mean(out[l]["dstat"])); out[l]["mean_dq"] = float(np.mean(out[l]["dq"]))
        del out[l]["dstat"]; del out[l]["dq"]
    # second test, on the quotient: for each class pair and layer, the matched-model
    # differences q_hi - q_lo (one per matched pair, independent networks) under a
    # Monte-Carlo sign-flip test; BH within the 45-pair family per layer
    rng = np.random.default_rng(0); B = 19999
    for l in LAYERS:
        D = defaultdict(list)
        for key, a in pop.items():
            if key[idx] != lo:
                continue
            kb = list(key); kb[idx] = hi; kb = tuple(kb)
            if kb in pop:
                for p in a["q"][l]:
                    D[p].append(pop[kb]["q"][l][p] - a["q"][l][p])
        ps, means = [], []
        for p, d in D.items():
            d = np.asarray(d); obs = abs(d.mean())
            flips = rng.choice([-1.0, 1.0], size=(B, len(d))); null = np.abs((flips * d).mean(1))
            ps.append((1 + np.sum(null >= obs)) / (1 + B)); means.append(d.mean())
        rej = bh(ps)
        out[l]["q_dec"] = int(sum(1 for m, r in zip(means, rej) if r and m < 0))
        out[l]["q_inc"] = int(sum(1 for m, r in zip(means, rej) if r and m > 0))
        out[l]["q_tests"] = len(ps)
    return {"factor": factor, "lo": lo, "hi": hi, "n_pairs": n, "mean_dacc": float(np.mean(dacc)) if dacc else None, "layers": out}


def main():
    pop = load(); print(len(pop), "models")
    # descriptive: mean quotient per level per layer
    levels = {"depth": [20, 32, 44, 56], "width": [1, 2], "wd": [0.0, 5e-4], "aug": [False, True]}
    desc = {}
    for f, lv in levels.items():
        idx = {"depth": 0, "width": 1, "wd": 2, "aug": 3}[f]
        for v in lv:
            ms = [m for k, m in pop.items() if k[idx] == v]
            desc[f"{f}={v}"] = {l: float(np.mean([np.mean(list(m["q"][l].values())) for m in ms])) for l in LAYERS} | {"acc": float(np.mean([m["test_acc"] for m in ms])), "n": len(ms)}
    contrasts = [contrast(pop, "depth", 20, 32), contrast(pop, "depth", 32, 44), contrast(pop, "depth", 44, 56), contrast(pop, "depth", 20, 56),
                 contrast(pop, "width", 1, 2), contrast(pop, "wd", 0.0, 5e-4), contrast(pop, "aug", False, True)]
    dump_json({"descriptive": desc, "contrasts": contrasts}, os.path.join(RESULTS_DIR, "factorial.json"))
    print("\nmean quotient by level:"); print(f"{'level':12s}" + "".join(f"{l:>9s}" for l in LAYERS) + "      acc")
    for k, v in desc.items():
        print(f"{k:12s}" + "".join(f"{v[l]:9.3f}" for l in LAYERS) + f"{v['acc']:9.4f}  (n={v['n']})")
    print("\ncontrasts (hi - lo): certified decreases/increases of the scale-free statistic out of tests; mean dq = change in quotient")
    lab = {"depth": "depth", "width": "width", "wd": "weight decay", "aug": "augmentation"}
    rows = [r"\begin{tabular}{@{}lrr" + "rr" * len(LAYERS) + "@{}}", r"\toprule",
            r"contrast & pairs & $\Delta$acc & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{l.replace('stage', 'stage ')}}}" for l in LAYERS) + r" \\",
            r" & & & " + " & ".join(r"mass $\downarrow/\uparrow$ & $\widehat E$ $\downarrow/\uparrow$" for _ in LAYERS) + r" \\", r"\midrule"]
    for c in contrasts:
        line = f"{lab[c['factor']]} {c['lo']}$\\to${c['hi']} & {c['n_pairs']} & {c['mean_dacc']:+.3f}"
        print(f"{c['factor']} {c['lo']}->{c['hi']}  n={c['n_pairs']}  dacc {c['mean_dacc']:+.4f}")
        for l in LAYERS:
            o = c["layers"][l]
            print(f"    {l:7s} mass dec {o['dec']:4d} inc {o['inc']:4d} / {o['tests']}   quotient: pairs dec {o['q_dec']:2d} inc {o['q_inc']:2d} / {o['q_tests']}   mean dq {o['mean_dq']:+.3f}")
            line += f" & {o['dec']}/{o['inc']} & {o['q_dec']}/{o['q_inc']}"
        rows.append(line + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    os.makedirs(FIG, exist_ok=True); open(os.path.join(FIG, "factorial_table.tex"), "w").write("\n".join(rows) + "\n")
    print("wrote figures/factorial_table.tex")


if __name__ == "__main__":
    main()
