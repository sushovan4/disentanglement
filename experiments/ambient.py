"""Direction A rescue: measure the interaction profile WITHOUT projecting.

Motivation. On real foundation-model features a d0=5 PCA retains only 8-26% of
the ambient first-interaction scale (`results/foundation_regime.json`), and the
projection choice is not the cause -- PCA, discriminant and max-margin
projections all give the same tightness. So a projected certificate on a wide
representation is valid but covers only small scales.

The escape route is the cross-subset expansion proved for the null-moment work:

    Dchi(r) = sum_{i>=2} (-1)^i * #{ MIXED i-subsets with circumradius <= r },

where "mixed" means the subset contains points from both clouds. Circumradius
is computable in ANY dimension, so this evaluates Dchi exactly in the ambient
space with no Alpha complex and no projection. Only mixed subsets contribute,
which is what makes it plausibly tractable: near r* few cross-class subsets are
active.

WHAT CAN GO WRONG, and why `concentration_report` runs first. In high dimension
pairwise distances concentrate, so the overlap may go from empty to complete
over a very narrow range of r. If so the mixed complex explodes exactly where
the profile becomes interesting, and no ambient method helps -- the obstruction
is statistical, not computational. Run the diagnostic before trusting or even
building on the profile.

Usage:
    python -m experiments.ambient --diagnose        # step zero: is there room?
    python -m experiments.ambient --profile <model> <layer>   # Dchi (may blow up)
    python -m experiments.ambient --onsets <model> <layer>    # r*_j (search, not sum)
"""

import argparse
import glob
import itertools
import json
import os

import numpy as np

from .common import RESULTS_DIR, ROOT, dump_json

FEAT_DIR = os.path.join(ROOT, "features", "foundation")


# --------------------------------------------------------------------------- #
# Step zero: is there a usable scale window at all?
# --------------------------------------------------------------------------- #
def cross_distance_spread(X, Y, qs=(0.0, 0.001, 0.01, 0.05, 0.25)):
    """Quantiles of the cross-class distance distribution.

    The profile lives between the first contact (min distance) and the scale at
    which most pairs are active. `window` reports how far the 1st-percentile
    distance is above the minimum, as a fraction: a spike (window ~ 0) means
    the overlap switches on all at once and no ambient method will resolve it.
    """
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    d2 = (np.einsum("ij,ij->i", X, X)[:, None]
          + np.einsum("ij,ij->i", Y, Y)[None, :] - 2.0 * X @ Y.T)
    d = np.sqrt(np.maximum(d2, 0)).ravel()
    q = {f"q{p}": float(np.quantile(d, p)) for p in qs}
    dmin = float(d.min())
    # A window is a RATIO to the closest pair, so duplicate feature vectors
    # (dmin = 0) or near-duplicates make it nan or astronomically large.  That
    # is a broken extraction, not a wide window, and the two must not look
    # alike in the output -- report the degeneracy instead of a bare nan.
    scale = float(np.median(d))
    dup = float(np.mean(d <= 1e-9 * max(scale, 1e-12)))
    ok = dmin > 1e-9 * max(scale, 1e-12)
    # window_1pct divides by the MINIMUM, an extreme order statistic: it keeps
    # falling as more points are drawn, so the ratio grows with sample size and
    # converges to nothing.  Absolute values are therefore comparable only at a
    # fixed cap, and any threshold quoted against them is cap-specific.
    # window_stable replaces the minimum with the 0.1% quantile -- same
    # geometric reading, "how fast does the cross-distance distribution lift
    # off first contact", but both ends are consistent estimators, so it can be
    # compared across sample sizes.
    q001 = q["q0.001"]
    return {**q, "min": dmin, "median": scale, "dup_frac": dup,
            "degenerate": not ok,
            "window_1pct": float(q["q0.01"] / dmin - 1.0) if ok else np.nan,
            "window_25pct": float(q["q0.25"] / dmin - 1.0) if ok else np.nan,
            "window_stable": float(q["q0.01"] / q001 - 1.0)
            if ok and q001 > 0 else np.nan}


def concentration_report(n_pairs=6, cap=200, seed=0):
    """Run the diagnostic on every extracted foundation representation."""
    rows = []
    for path in sorted(glob.glob(os.path.join(FEAT_DIR, "*.npz"))):
        z = np.load(path)
        model = os.path.basename(path)[:-4]
        labels = z["labels"]
        classes = sorted(np.unique(labels).tolist())
        rng = np.random.default_rng(seed)
        pairs = [tuple(rng.choice(classes, 2, replace=False)) for _ in range(n_pairs)]
        for layer in [k for k in z.files if k != "labels"]:
            F = z[layer]
            w1, w25, ws, deg = [], [], [], []
            for c1, c2 in pairs:
                s = cross_distance_spread(F[labels == c1][:cap], F[labels == c2][:cap])
                w1.append(s["window_1pct"]); w25.append(s["window_25pct"])
                ws.append(s["window_stable"]); deg.append(s["degenerate"])
            bad = float(np.mean(deg))
            rec = {"model": model, "layer": layer, "ambient_dim": int(F.shape[1]),
                   "degenerate_frac": bad,
                   "cap": cap,
                   "window_1pct": float(np.nanmedian(w1)) if bad < 1 else float("nan"),
                   "window_25pct": float(np.nanmedian(w25)) if bad < 1 else float("nan"),
                   "window_stable": float(np.nanmedian(ws)) if bad < 1 else float("nan")}
            rows.append(rec)
            head = (f"{rec['model']:>16} {rec['layer']:>9} "
                    f"D={rec['ambient_dim']:>5}  ")
            if bad > 0:
                print(head + f"DEGENERATE ({bad:.0%} of pairs have duplicate "
                             f"features) -- extraction is broken, not concentrated")
            else:
                print(head + f"1%-window={rec['window_1pct']:+.3f}  "
                             f"25%-window={rec['window_25pct']:+.3f}  "
                             f"stable={rec['window_stable']:+.3f}")
    # A missing model is invisible in a globbed report: the table still prints
    # a full-looking set of rows and reads as complete.  convnext_tiny failed
    # three submissions in a row before anyone noticed it was absent.  Check the
    # extracted set against the declared one and say so loudly.
    try:
        from .foundation import MODELS
        want = {m.name for m in MODELS}
        have = {r["model"].split("__")[0] for r in rows}
        missing = sorted(want - have)
        if missing:
            print(f"\n!! MISSING {len(missing)} of {len(want)} declared models: "
                  f"{', '.join(missing)}\n   These never extracted -- the table "
                  f"above is INCOMPLETE, not a full sweep.")
    except Exception:                                   # noqa: BLE001
        pass
    if rows:
        dump_json(rows, os.path.join(RESULTS_DIR, "ambient_concentration.json"))
        print(f"\n(cap={cap} points per class. window_1pct and window_25pct "
              f"divide by the MINIMUM\ncross distance, so they grow with cap "
              f"and are comparable only at fixed cap;\n`stable` divides by the "
              f"0.1% quantile instead and is cap-robust. Ratios between\n"
              f"layers are stable under either.)\n")
        print("\nHow to read this. window_1pct is how far the 1st-percentile "
              "cross distance sits above the very closest pair, as a fraction.\n"
              "  >~ 0.10  : a real window exists; the mixed complex grows "
              "gradually and the ambient profile is worth computing.\n"
              "  <~ 0.02  : distances are concentrated into a spike; the "
              "overlap switches on all at once, the complex explodes exactly\n"
              "             where the profile matters, and no ambient method "
              "rescues this. Take the reframing route instead.")
    return rows


# --------------------------------------------------------------------------- #
# Route 2: the ambient profile itself
# --------------------------------------------------------------------------- #
def _circumradius_le(P, r):
    """Do the r-balls around rows of P share a point? (circumradius <= r)"""
    P = np.asarray(P, np.float64)
    if len(P) == 1:
        return True
    if len(P) == 2:
        return float(np.linalg.norm(P[0] - P[1])) <= 2 * r + 1e-12
    from scipy.optimize import minimize
    f = lambda c: np.max(np.linalg.norm(P - c, axis=1))
    return minimize(f, P.mean(0), method="Nelder-Mead",
                    options={"xatol": 1e-9, "fatol": 1e-11,
                             "maxiter": 5000}).fun <= r + 1e-9


def ambient_dchi(X, Y, r, max_order=None, budget=200000):
    """Dchi(r) in the AMBIENT space via the mixed-subset expansion.

    Enumerates mixed subsets incrementally: a subset can have circumradius <= r
    only if all of its sub-pairs do, so we grow from the close cross pairs.

    MUST RUN TO EXHAUSTION. The alternating sum has large intermediate terms
    that cancel -- on a 6+6 point example the counts run 22, 48, 44, 23, 7, 1, 0
    and only the complete sum equals the true value of 1. Stopping early is not
    an approximation, it is a wrong answer (truncating that example at order 4
    gives 18 instead of 1). `max_order=None` therefore means "until no subsets
    survive", and any truncation is reported in the returned info dict.

    Returns (None, info) if the enumeration exceeds `budget`, which is the
    signal that concentration has made this scale intractable.
    """
    X = np.asarray(X, np.float64); Y = np.asarray(Y, np.float64)
    nx, ny = len(X), len(Y)
    P = np.vstack([X, Y])
    is_y = np.arange(len(P)) >= nx

    d2 = (np.einsum("ij,ij->i", P, P)[:, None]
          + np.einsum("ij,ij->i", P, P)[None, :] - 2.0 * P @ P.T)
    close = np.sqrt(np.maximum(d2, 0)) <= 2 * r + 1e-12
    np.fill_diagonal(close, False)

    # order 2: mixed close pairs
    pairs = [(i, j) for i in range(len(P)) for j in range(i + 1, len(P))
             if close[i, j] and (is_y[i] != is_y[j])]
    total = len(pairs)                      # (-1)^2 * count
    counts = {2: len(pairs)}
    current = [frozenset(p) for p in pairs]

    order = 2
    while max_order is None or order < max_order:
        order += 1
        nxt = set()
        for S in current:
            cand = set(range(len(P)))
            for v in S:
                cand &= set(np.flatnonzero(close[v]))
            for v in cand - S:
                nxt.add(S | {v})
            if len(nxt) > budget:
                return None, {"aborted_at_order": order, "counts": counts}
        keep = [S for S in nxt
                if any(is_y[v] for v in S) and any(not is_y[v] for v in S)
                and _circumradius_le(P[sorted(S)], r)]
        counts[order] = len(keep)
        total += (-1) ** order * len(keep)
        current = keep
        if not keep:
            return total, {"counts": counts, "truncated": False}
    return total, {"counts": counts, "truncated": True,
                   "warning": f"stopped at max_order={max_order} with "
                              f"{len(current)} subsets still live; the "
                              f"alternating sum is NOT converged"}


# --------------------------------------------------------------------------- #
# Route 3: the ONSET vector, exactly, in ambient dimension
# --------------------------------------------------------------------------- #
def ambient_onsets(clouds, max_order=None, budget=2000000, verbose=False):
    """r*_j for j = 2..k, exactly, with no projection and no chi.

    r*_j is the smallest r at which some j of the clouds share a point,
    equivalently the minimum circumradius over j-subsets that span j distinct
    clouds. That is a MINIMISATION, so it admits branch and bound: for any
    subset S, circumradius(S) >= max_{u,v in S} d(u,v) / 2, and the bound is
    monotone under adding points. Processing candidates in increasing order of
    that bound and pruning against the incumbent means nothing past the
    minimiser is ever examined.

    (An earlier version binary-searched r and called a routine that enumerated
    ALL subsets at each step, roughly seventy times over -- far slower than the
    Dchi enumeration it was meant to beat. That was the bug behind the 16-hour
    stage1 run, not a property of the data.)
    """
    k = len(clouds)
    max_order = max_order or k
    P = np.vstack([np.asarray(c, np.float64) for c in clouds])
    owner = np.concatenate([np.full(len(c), i) for i, c in enumerate(clouds)])
    sq = np.einsum("ij,ij->i", P, P)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * P @ P.T, 0.0))
    np.fill_diagonal(D, np.inf)

    cross = owner[:, None] != owner[None, :]
    out = {2: 0.5 * float(D[cross].min())}
    info = {"n_points": int(len(P)), "evaluated": {}}
    if verbose:
        print(f"  r*_2 = {out[2]:.6f}")

    # candidate seeds: cross pairs in increasing distance
    iu = np.triu_indices(len(P), 1)
    mask = cross[iu]
    pi, pj, pd = iu[0][mask], iu[1][mask], D[iu][mask]
    order_idx = np.argsort(pd)
    pi, pj, pd = pi[order_idx], pj[order_idx], pd[order_idx]

    for order in range(3, max_order + 1):
        best = np.inf
        evaluated = 0
        for a, b, dab in zip(pi, pj, pd):
            if 0.5 * dab >= best:
                break                       # every later seed is worse
            base = frozenset((int(a), int(b)))
            stack = [(base, {int(owner[a]), int(owner[b])})]
            while stack:
                S, clouds_hit = stack.pop()
                if len(S) == order:
                    evaluated += 1
                    if evaluated > budget:
                        info["evaluated"][order] = evaluated
                        info[f"aborted_order_{order}"] = True
                        return out, info
                    r = _circumradius(P[sorted(S)])
                    if r < best:
                        best = r
                    continue
                # extend by a point from a cloud not yet represented
                for v in range(len(P)):
                    if v in S or int(owner[v]) in clouds_hit:
                        continue
                    lb = 0.5 * max(D[v, u] for u in S)
                    if lb >= best or lb >= np.inf:
                        continue
                    if 0.5 * max(dab, max(D[v, u] for u in S)) >= best:
                        continue
                    stack.append((S | {v}, clouds_hit | {int(owner[v])}))
        if not np.isfinite(best):
            info[f"no_witness_order_{order}"] = True
            return out, info
        out[order] = float(best)
        info["evaluated"][order] = evaluated
        if verbose:
            print(f"  r*_{order} = {best:.6f}   (ratio to r*_2: {best/out[2]:.3f})"
                  f"   [{evaluated} subsets evaluated]")
    return out, info


def _circumradius(P):
    """Radius of the smallest enclosing ball of the rows of P."""
    P = np.asarray(P, np.float64)
    if len(P) == 1:
        return 0.0
    if len(P) == 2:
        return float(np.linalg.norm(P[0] - P[1]) / 2.0)
    from scipy.optimize import minimize
    f = lambda c: np.max(np.linalg.norm(P - c, axis=1))
    return float(minimize(f, P.mean(0), method="Nelder-Mead",
                          options={"xatol": 1e-10, "fatol": 1e-12,
                                   "maxiter": 10000}).fun)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--profile", nargs=2, metavar=("MODEL", "LAYER"))
    ap.add_argument("--onsets", nargs=2, metavar=("MODEL", "LAYER"))
    ap.add_argument("--classes", type=int, default=5)
    ap.add_argument("--max-order", type=int, default=4)
    ap.add_argument("--cap", type=int, default=200)  # match the function default
    a = ap.parse_args()
    if a.diagnose:
        concentration_report(cap=a.cap)
    elif a.profile:
        model, layer = a.profile
        z = np.load(os.path.join(FEAT_DIR, f"{model}.npz"))
        lab = z["labels"]; F = z[layer]
        cs = sorted(np.unique(lab).tolist())[:2]
        X = F[lab == cs[0]][:a.cap].astype(np.float64)
        Y = F[lab == cs[1]][:a.cap].astype(np.float64)
        rs = float(0.5 * np.min(np.linalg.norm(X[:, None] - Y[None], axis=-1)))
        print(f"r* = {rs:.4f}; profile just above first contact:")
        for mult in (1.0, 1.02, 1.05, 1.10, 1.25):
            r = rs * mult
            v, info = ambient_dchi(X, Y, r)
            print(f"  r={r:.4f} ({mult:.2f}x r*)  Dchi={v}  {info}")
    elif a.onsets:
        model, layer = a.onsets
        z = np.load(os.path.join(FEAT_DIR, f"{model}.npz"))
        lab = z["labels"]; F = z[layer]
        cs = sorted(np.unique(lab).tolist())[:a.classes]
        clouds = [F[lab == c][:a.cap].astype(np.float64) for c in cs]
        print(f"{model}/{layer}: D={F.shape[1]}, {len(cs)} classes x {a.cap} pts")
        onsets, info = ambient_onsets(clouds, max_order=a.max_order, verbose=True)
        if len(onsets) >= 2:
            base = onsets[2]
            print("\nonset vector, ambient, exact (no projection):")
            for j in sorted(onsets):
                print(f"   r*_{j} = {onsets[j]:.6f}   ratio to r*_2 = {onsets[j]/base:.3f}")
            print("\nRising ratios are the occupancy form of pairwise dominance: "
                  "higher-order overlaps begin at disproportionately larger scales.")
        print("info:", info)
        dump_json({"model": model, "layer": layer, "classes": cs, "cap": a.cap,
                   "onsets": {str(k): v for k, v in onsets.items()}, "info": info},
                  os.path.join(RESULTS_DIR, "ambient_onsets",
                               f"{model}_{layer}.json"))
    else:
        ap.print_help()
