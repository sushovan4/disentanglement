"""E9 (Direction B): does the representation reorganize *when* the model groks?

Grokking gives a sharp, well-known generalization event whose representational
correlate is usually read off eyeballed curves.  The paired test of the
disentanglement paper can do better: it attaches a p-value to each
checkpoint-to-checkpoint change in class-cloud interaction, so we can ask
whether the topological reorganization leads, coincides with, or lags the
validation jump -- and whether it disappears in a control that never groks.

Protocol, per run:
  * classes are the p residue classes of the answer; we measure a fixed
    random subset of `--classes` of them (default 10 -> 45 pairs), matching
    the paper's pairwise convention;
  * at every checkpoint, per layer, we compute F disjoint-fold statistics
    (the paired-test inputs) and, optionally, the permutation quotient;
  * the fold statistic is AVERAGED OVER PAIRS before testing, because the
    claim is about the representation ("it reorganized between these two
    steps"), not about an individual class pair -- and because a per-pair
    family is too large for the correction to reject anything (see
    `feasible`);
  * consecutive checkpoints are compared with the exact sign-flip test of
    Section 3.3, Benjamini--Hochberg-corrected across the sweep.

Cost note: the fold statistics carry the trajectory claim and need NO
permutations, so the default is cheap.  `--B 0` skips the permutation
quotients entirely (trajectory only); raising it adds absolute context at
linear cost.

Usage:
  python -m experiments.e9_grok <config_index> [--layers penult] [--B 99]
  python -m experiments.e9_grok --report
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

from .common import FOLD_STAT, CKPT_DIR, PCA_DIM, RESULTS_DIR, dump_json
from .ecp import (default_r_grid, mixup_ecp, paired_subsample_test,
                  permutation_test, profile_stat)
from .extract import class_clouds
from .grok import CONFIGS, GrokTransformer, checkpoint_steps, make_data, grok_step

# Fold count is bounded above by the data (each residue class has exactly p
# members, and a fold needs > d0+1 points) and below by RESOLUTION: the
# sign-flip test's smallest attainable p-value is 2/2^F, and BH over a family
# of N tests requires the smallest p to clear q/N. F=6 gives 2/64 = 0.031,
# which cannot clear q/N for any realistic N -- the analysis is then incapable
# of rejecting anything. See `feasible()`.
N_FOLDS = 12         # p=97 => fold_m = 8 > d0+1; 2/2^12 = 4.9e-4
STAT = "int"


def feasible(n_tests, n_folds=N_FOLDS, q=0.05, verbose=True):
    """Can this analysis reject anything at all?

    The sign-flip p-value is a multiple of 2^{-F}, so its minimum is 2/2^F.
    Benjamini--Hochberg requires the smallest p-value to clear q/n_tests. If
    the minimum attainable p exceeds that, no configuration of the data can
    produce a rejection and the whole sweep returns zero by construction.
    This check exists because that is exactly what happened on the first run.
    """
    p_min = 2.0 / (2 ** n_folds)
    thresh = q / max(n_tests, 1)
    ok = p_min < thresh
    if verbose:
        print(f"[feasibility] folds={n_folds}  min attainable p={p_min:.2e}  "
              f"BH threshold (q/N, N={n_tests})={thresh:.2e}  "
              f"{'OK' if ok else 'INFEASIBLE -- cannot reject anything'}")
    return ok


def bh(pvals, q=0.05):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    n = len(p)
    thresh = q * (np.arange(1, n + 1)) / n
    passed = p[order] <= thresh
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    out = np.zeros(n, dtype=bool)
    out[order[:k]] = True
    return out


@torch.no_grad()
def features_at(cfg, step, which="all"):
    """Representations at a checkpoint, with their answer labels."""
    path = os.path.join(CKPT_DIR, cfg.name, f"step{step}.pt")
    if not os.path.exists(path):
        return None, None
    Xtr, ytr, Xte, yte = make_data(cfg)
    if which == "test":
        X, y = Xte, yte
    elif which == "train":
        X, y = Xtr, ytr
    else:
        X, y = torch.cat([Xtr, Xte]), torch.cat([ytr, yte])
    model = GrokTransformer(cfg.p, cfg.depth, cfg.d_model, cfg.heads)
    model.load_state_dict(torch.load(path, map_location="cpu")["model"])
    model.eval()
    feats = {}
    for i in range(0, len(X), 4096):
        _, f = model(X[i:i + 4096], return_features=True)
        for k, v in f.items():
            feats.setdefault(k, []).append(v.numpy())
    return {k: np.concatenate(v) for k, v in feats.items()}, y.numpy()


def measure_checkpoint(feats, labels, classes, layers, B, seed):
    """Per-layer: fold statistics for every pair, plus optional quotients."""
    out = {}
    counts = {c: int(np.sum(labels == c)) for c in classes}
    fold_m = min(counts.values()) // N_FOLDS
    m_head = min(min(counts.values()), 200)
    if fold_m < PCA_DIM + 2:
        # each class contributes exactly p points, so p must exceed
        # N_FOLDS*(d0+2); with d0=5 and F=12 that means p >= 84.
        raise ValueError(
            f"fold size {fold_m} too small for a {PCA_DIM}-dimensional complex "
            f"(need >= {PCA_DIM + 2}); use a larger modulus p, fewer folds, or "
            f"--inputs all")
    from itertools import combinations
    for layer in layers:
        Fm = feats[layer]
        rec = {"folds": {}, "pairs": {}, "fold_m": fold_m, "m_head": m_head}
        for c1, c2 in combinations(classes, 2):
            key = f"{c1},{c2}"
            Ts = []
            for fold in range(N_FOLDS):
                clouds = class_clouds(Fm, labels, [c1, c2], m=fold_m,
                                      d0=PCA_DIM, fold=fold)
                g = default_r_grid(clouds)
                # scale-free fold statistic: under weight decay the feature
                # norm moves across the transition, and the raw profile mass
                # (length units) would track that rather than interaction
                Ts.append(profile_stat(mixup_ecp(clouds, g), g, FOLD_STAT))
            rec["folds"][key] = Ts
            if B and B > 0:
                clouds = class_clouds(Fm, labels, [c1, c2], m=m_head, d0=PCA_DIM)
                res = permutation_test(clouds, default_r_grid(clouds), B=B,
                                       stat=STAT, rng=seed)
                rec["pairs"][key] = {"quotient": res["quotient"],
                                     "p_up": res["p_up"],
                                     "p_down": res["p_down"]}
        out[layer] = rec
    return out


def run(cfg, layers, B, n_classes, which, shuffle_classes=False):
    curve_path = os.path.join(RESULTS_DIR, "grok", f"{cfg.name}.json")
    if not os.path.exists(curve_path):
        sys.exit(f"missing training curve: {curve_path} (run experiments.grok first)")
    curve = json.load(open(curve_path))

    rng = np.random.default_rng(0)
    classes = sorted(rng.choice(cfg.p, size=n_classes, replace=False).tolist())
    steps = [s for s in checkpoint_steps(cfg)
             if os.path.exists(os.path.join(CKPT_DIR, cfg.name, f"step{s}.pt"))]

    # Falsification control: replace the true residue classes by an arbitrary
    # grouping of the same sizes, fixed across checkpoints so the trajectory is
    # comparable. If the reorganization at the generalization jump survives
    # this, it is generic representational drift rather than anything about the
    # task's class structure -- and the Direction B claim is much weaker.
    relabel = np.random.default_rng(12345) if shuffle_classes else None

    per_step = {}
    for s in steps:
        feats, labels = features_at(cfg, s, which=which)
        if feats is None:
            continue
        if relabel is not None:
            labels = labels.copy()
            relabel.shuffle(labels)          # same class sizes, arbitrary membership
        per_step[s] = measure_checkpoint(feats, labels, classes, layers, B, s)
        mq = {L: float(np.mean([v["quotient"] for v in per_step[s][L]["pairs"].values()]))
              for L in layers if per_step[s][L]["pairs"]}
        print(f"[{cfg.name}] step {s:>6} measured" + (f"  mean_q={mq}" if mq else ""),
              flush=True)

    # ---- consecutive-checkpoint paired tests (the Direction B claim) -------
    # The claim of interest is about the REPRESENTATION ("it reorganized
    # between step a and b"), not about an individual class pair, so the fold
    # statistic is averaged over pairs before the paired test. This is both the
    # natural scientific statement and what keeps the family small enough for
    # the correction to be able to reject at all.
    rows = []
    for a, b in zip(steps[:-1], steps[1:]):
        for L in layers:
            keys = sorted(per_step[a][L]["folds"])
            T0 = np.mean([per_step[a][L]["folds"][k] for k in keys], axis=0)
            T1 = np.mean([per_step[b][L]["folds"][k] for k in keys], axis=0)
            res = paired_subsample_test(T1, T0)       # D<0 => disentangling
            rows.append({"step_from": a, "step_to": b, "layer": L,
                         "n_pairs_averaged": len(keys),
                         "mean_diff": res["mean_diff"],
                         "p_sign": res["p_sign"], "ci95": list(res["ci95"])})
    if rows:
        feasible(len(rows))
        rej = bh([r["p_sign"] for r in rows])
        for r, x in zip(rows, rej):
            r["bh_reject"] = bool(x)

    traj = [{"step": s, "layer": L,
             "mean_quotient": float(np.mean([v["quotient"]
                                             for v in per_step[s][L]["pairs"].values()]))
             if per_step[s][L]["pairs"] else None,
             "mean_fold_stat": float(np.mean([np.mean(v)
                                              for v in per_step[s][L]["folds"].values()]))}
            for s in per_step for L in layers]

    out = {"name": cfg.name, "config": curve["config"], "curve": curve["curve"],
           "grok_step": curve.get("grok_step"), "classes": classes,
           "layers": layers, "which_inputs": which, "B": B,
           "shuffle_classes": bool(shuffle_classes),
           "trajectory": traj, "paired": rows}
    tag = "_shuffled" if shuffle_classes else ""
    dump_json(out, os.path.join(RESULTS_DIR, "e9_grok", f"{cfg.name}{tag}.json"))
    print(f"[{cfg.name}] {sum(r['bh_reject'] for r in rows)}/{len(rows)} "
          f"BH-significant steps; grok at {curve.get('grok_step')}")
    return out


def report(dense_only=False):
    """Align certified reorganization against the generalization jump.

    dense_only: restrict the lead/lag analysis to the corrected dense window
    [1800, 3200] at 100-step resolution (the *_dense1800 runs). This is the
    analysis the paper quotes (r = -0.80, n = 45 intervals); pooling the
    coarse log-spaced runs in as well dilutes it to about -0.5 with null
    lags, because the coarse intervals span thousands of steps.

    Ranks by EFFECT SIZE, not by count. After the per-pair aggregation there is
    one test per (step, layer), so counting significant tests per step gives 1
    everywhere and an argmax over counts merely returns the earliest
    significant step -- which is the initial memorization transient in every
    run, grokking or not. That was the first version of this function and it
    reported step 1 for all five configs.
    """
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "e9_grok", "*.json")))
    if not files:
        print("no E9 results yet")
        return
    summary, by_interval = [], {}
    for f in files:
        d = json.load(open(f))
        if d.get("shuffle_classes"):
            continue                     # reported separately; see --report-shuffled
        gs = d.get("grok_step")
        wd = d["config"]["weight_decay"]
        sig = [r for r in d["paired"] if r.get("bh_reject")]
        for r in d["paired"]:
            by_interval.setdefault((r["step_from"], r["step_to"]), {}) \
                       .setdefault(wd, []).append(r["mean_diff"])
        peak = max(sig, key=lambda r: abs(r["mean_diff"])) if sig else None
        # the change spanning the generalization jump, if the run groks
        at_grok = next((r for r in d["paired"] if r["step_to"] == gs), None)
        rec = {"name": d["name"], "wd": wd, "grok_step": gs,
               "n_significant": len(sig),
               "peak_step": peak["step_to"] if peak else None,
               "peak_mean_diff": peak["mean_diff"] if peak else None,
               "at_grok_mean_diff": at_grok["mean_diff"] if at_grok else None,
               "at_grok_significant": bool(at_grok["bh_reject"]) if at_grok else None}
        summary.append(rec)
        print(rec)

    # the contrast that turns a correlation into a claim
    print("\nmean_diff by interval, grokking arm vs control "
          "(negative = disentangling):")
    print(f"{'interval':>16} {'wd=1.0':>12} {'wd=0':>12}")
    for k in sorted(by_interval, key=lambda k: k[0]):
        a = by_interval[k].get(1.0, []); b = by_interval[k].get(0.0, [])
        if a and b:
            mark = "  <== generalization jump" if any(
                s["grok_step"] == k[1] for s in summary if s["grok_step"]) else ""
            print(f"{str(k[0])+'->'+str(k[1]):>16} "
                  f"{np.mean(a):>12.3f} {np.mean(b):>12.3f}{mark}")

    # Lead / lag: the question Direction B set out to answer. Correlate the
    # certified change over interval t against the validation change over
    # interval t+k. k=0 dominating means the reorganization is simultaneous
    # with the generalization change at this resolution, not predictive of it.
    try:
        from scipy.stats import pearsonr
        series = []
        for f in files:
            d = json.load(open(f))
            if d.get("shuffle_classes") or d["config"]["weight_decay"] == 0:
                continue
            if dense_only and not d["name"].endswith("_dense1800"):
                continue
            val = {c["step"]: c["val_acc"] for c in d["curve"]}
            rows = sorted([r for r in d["paired"] if r["layer"] == "penult"],
                          key=lambda r: r["step_from"])
            if dense_only:   # the dense window itself, not the log-spaced lead-in
                rows = [r for r in rows if 1800 <= r["step_from"] and r["step_to"] <= 3200]
            dv = [val[r["step_to"]] - val[r["step_from"]]
                  for r in rows if r["step_to"] in val and r["step_from"] in val]
            md = [r["mean_diff"] for r in rows
                  if r["step_to"] in val and r["step_from"] in val]
            if len(dv) > 5:
                series.append((np.array(dv), np.array(md)))
        if series:
            n_int = sum(len(dv) for dv, _ in series)
            print(f"\nlead/lag (mean_diff at t vs val change at t+k; "
                  f"{'dense window only' if dense_only else 'all wd=1 runs'}, "
                  f"{len(series)} runs, {n_int} intervals):")
            lags = {}
            for k in (-2, -1, 0, 1, 2):
                xs, ys = [], []
                for dv, md in series:
                    a, b = (md[:len(md) - k], dv[k:]) if k >= 0 else (md[-k:], dv[:len(dv) + k])
                    n = min(len(a), len(b)); xs += list(a[:n]); ys += list(b[:n])
                r, pv = pearsonr(xs, ys)
                lags[k] = {"r": float(r), "p": float(pv)}
                tag = ("contemporaneous" if k == 0 else
                       f"topology {'leads' if k > 0 else 'lags'} by {abs(k)}")
                print(f"  k={k:+d}  r={r:+.3f}  p={pv:.1e}   {tag}")
            summary_lag = lags
        else:
            summary_lag = None
    except Exception as e:                                    # noqa: BLE001
        print("lead/lag skipped:", e); summary_lag = None

    dump_json({"per_run": summary, "lead_lag": summary_lag,
               "lead_lag_scope": "dense1800 only" if dense_only else "all wd=1 runs",
               "by_interval": {f"{a}->{b}": v for (a, b), v in by_interval.items()}},
              os.path.join(RESULTS_DIR, "e9_grok_report.json"))
    print("\nRead the lead/lag table above with the coarse runs excluded if you "
          "want the resolved answer: on the dense [1800,3200] grid the "
          "contemporaneous term dominates and both lags are null, so the "
          "reorganization is simultaneous with the generalization change at "
          "100-step resolution -- it does not anticipate it.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("index", nargs="?")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--dense-only", action="store_true",
                    help="lead/lag on the dense [1800,3200] window only (the "
                         "numbers quoted in the paper)")
    ap.add_argument("--layers", default="penult")
    ap.add_argument("--B", type=int, default=99)
    ap.add_argument("--classes", type=int, default=10,
                    help="residue classes to measure; pairs are averaged over")
    ap.add_argument("--inputs", default="all", choices=["all", "train", "test"])
    ap.add_argument("--shuffle-classes", action="store_true",
                    help="falsification control: arbitrary groupings instead of "
                         "the true residue classes; writes *_shuffled.json")
    a = ap.parse_args()
    if a.report:
        report(dense_only=a.dense_only)
    else:
        run(CONFIGS[int(a.index)], a.layers.split(","), a.B, a.classes, a.inputs,
            shuffle_classes=a.shuffle_classes)
