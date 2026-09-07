"""Path 1: adversarially hunt counterexamples to the pairwise-dominance conjecture.

THE CONJECTURE (E2, main.tex).  In the guard-certified regime, the triple
interaction quotient is dominated by the maximal pairwise one,

    E_triple  <=  C * max_{pairs} E_pair,      C = 1 empirically,

holding in 834/840 measured cells (median ratio 0.47, max 1.07).

WHY IT IS NOT OBVIOUS, and therefore worth attacking before anyone invests in
a proof.  Witness-region nesting gives the SET inclusion
cap_{i in T} U_i subset cap_{i in S} U_i for S subset T, but chi is not
monotone under inclusion: cutting a blob by a third union of balls can CREATE
components, so nesting alone yields nothing.  The normalization cuts the wrong
way too -- the triple's permutation null is far smaller than a pair's, which
should INFLATE triple quotients rather than suppress them.  Yet the measured
median ratio is 0.47.  So the empirical fact has no mechanism yet, and the six
exceedances may be permutation noise or may be the real thing.

WHAT THIS DOES.  Searches a parameterized family of three-cloud configurations
for the largest achievable

    ratio = E_triple / max_{pairs} E_pair

subject to the guard not firing on any pair (outside that regime the
conjecture claims nothing).  A robust ratio > 1 means the conjecture needs a
hypothesis it does not currently have; failure to find one after a wide search
is the evidence that justifies spending theorist time.

Configurations are anisotropic Gaussians -- centres and per-axis log-scales --
which is the smallest family rich enough to produce the shapes that break
naive intersection arguments: thin slabs meeting at a common region, one cloud
straddling two others, and so on.

Usage:
    python experiments/dominance_search.py [--iters N] [--d 3] [--n 30]
    python experiments/dominance_search.py --replay results/dominance_search.json
"""

import argparse
import json
import os
from itertools import combinations

import numpy as np

from .common import RESULTS_DIR, dump_json
from .ecp import cancellation_guard, default_r_grid, permutation_test


def sample_config(rng, d, n, theta):
    """Three anisotropic Gaussian clouds from a flat parameter vector."""
    centres = theta[: 3 * d].reshape(3, d)
    logs = theta[3 * d:].reshape(3, d)
    return [rng.normal(size=(n, d)) * np.exp(logs[i]) + centres[i]
            for i in range(3)]


def evaluate(clouds, B, rng):
    """(ratio, detail) for one configuration; ratio is NaN if inadmissible."""
    pair_q, guard_any = [], False
    for a, b in combinations(range(3), 2):
        sub = [clouds[a], clouds[b]]
        grid = default_r_grid(sub)
        res = permutation_test(sub, grid, B=B, stat="int", rng=rng)
        fired, _ = cancellation_guard(res["dchi_obs"], grid, sub)
        guard_any |= fired
        pair_q.append(res["quotient"])
    grid3 = default_r_grid(clouds)
    r3 = permutation_test(clouds, grid3, B=B, stat="int", rng=rng)
    q3 = r3["quotient"]
    pair_q = np.array(pair_q, dtype=float)
    if guard_any or not np.isfinite(q3) or not np.all(np.isfinite(pair_q)):
        return np.nan, None                    # outside the claimed regime
    mx = float(np.nanmax(pair_q))
    if mx <= 0:
        return np.nan, None
    return q3 / mx, {"q_triple": float(q3), "q_pairs": pair_q.tolist(),
                     "max_pair": mx}


def search(iters, d, n, B, seed, sigma0=1.2):
    """Random restarts plus Gaussian hill-climbing on the ratio."""
    rng = np.random.default_rng(seed)
    dim = 6 * d
    best, best_theta, best_detail = -np.inf, None, None
    theta = np.concatenate([rng.normal(scale=sigma0, size=3 * d),
                            rng.normal(scale=0.4, size=3 * d)])
    cur, step, stall = -np.inf, 0.5, 0
    for it in range(iters):
        cand = theta + rng.normal(scale=step, size=dim)
        clouds = sample_config(rng, d, n, cand)
        ratio, detail = evaluate(clouds, B, rng)
        if np.isfinite(ratio) and ratio > cur:
            theta, cur, stall = cand, ratio, 0
            if ratio > best:
                best, best_theta, best_detail = ratio, cand.copy(), detail
        else:
            stall += 1
        if stall >= 25:                        # restart: the surface is rugged
            theta = np.concatenate([rng.normal(scale=sigma0, size=3 * d),
                                    rng.normal(scale=0.4, size=3 * d)])
            cur, stall = -np.inf, 0
        if (it + 1) % 50 == 0:
            print(f"  iter {it+1:>5}  best ratio {best:.3f}", flush=True)
    return best, best_theta, best_detail


def confirm(theta, d, n, B, reps, seed):
    """Re-evaluate the winner on fresh draws -- a single lucky draw is noise."""
    out = []
    for k in range(reps):
        rng = np.random.default_rng(seed + 9973 * k)
        ratio, _ = evaluate(sample_config(rng, d, n, theta), B, rng)
        if np.isfinite(ratio):
            out.append(ratio)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--B", type=int, default=99)
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    print(f"searching: d={a.d} n={a.n} B={a.B} iters={a.iters}", flush=True)
    best, theta, detail = search(a.iters, a.d, a.n, a.B, a.seed)
    if theta is None:
        print("no admissible configuration found")
        return
    print(f"\nbest ratio during search: {best:.3f}")
    print(f"  {detail}")

    fresh = confirm(theta, a.d, a.n, a.B, a.reps, a.seed + 1)
    print(f"\nre-evaluated on {len(fresh)} fresh draws of the same geometry:")
    print(f"  mean {fresh.mean():.3f}   median {np.median(fresh):.3f}   "
          f"max {fresh.max():.3f}   P[ratio>1] = {np.mean(fresh > 1):.3f}")
    verdict = ("COUNTEREXAMPLE: dominance fails robustly, not by a lucky draw"
               if np.median(fresh) > 1 else
               "no robust counterexample: the search peak did not reproduce")
    print(f"\n{verdict}")

    dump_json({"best_search_ratio": float(best), "theta": theta.tolist(),
               "detail": detail, "fresh_ratios": fresh.tolist(),
               "d": a.d, "n": a.n, "B": a.B, "iters": a.iters,
               "median_fresh": float(np.median(fresh)),
               "verdict": verdict},
              os.path.join(RESULTS_DIR, "dominance_search.json"))
    print("wrote results/dominance_search.json")


if __name__ == "__main__":
    main()
