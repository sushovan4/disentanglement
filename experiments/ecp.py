r"""Core Mixup ECP machinery for the disentanglement experiments.

Implements, on the diagonal scale t = (r, ..., r):

  * ecc_alpha        -- Euler characteristic curve chi(U(X; r)) via an Alpha
                        complex sweep (signed simplex counts, no reduction).
  * mixup_ecp        -- k-fold Mixup ECP  Dchi(r) = chi( \cap_i U(X_i; r) )
                        by dual inclusion-exclusion over pooled sub-clouds
                        (Appendix `Inclusion-Exclusion for Valuations' of the
                        foundations paper): for same-radius unions,
                        U(X;r) u U(Y;r) = U(X u Y; r).
  * first_interaction_scale -- the guard r* against Euler cancellation.
  * permutation_test -- both one-sided exact tests (p_up: excess interaction,
                        p_down: separation) + the interaction quotient.
  * paired_subsample_test -- sign-flip + t-interval for comparative claims.

Conventions: clouds are (n_i, d) float64 arrays; r_grid is an increasing 1-D
array of ball radii r (NOT alpha filtration values, which are r^2).
"""

from itertools import combinations

import numpy as np
from scipy.spatial import cKDTree

import gudhi as gd


# --------------------------------------------------------------------------- #
# Euler characteristic curves
# --------------------------------------------------------------------------- #
def ecc_alpha(points, r_grid, weights=None):
    """chi(U(points; r)) for each r in r_grid, via the Alpha complex nerve.

    With per-point `weights` w >= 0 the ball of point i at grid value r has
    radius sqrt(r^2 + w_i) (weighted Alpha complex / power distance): the
    density-offset variant, in which sparser clouds are thickened ahead of
    denser ones by a fixed squared offset.  weights=None is the diagonal.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return np.zeros(len(r_grid), dtype=np.int64)
    if weights is None:
        ac = gd.AlphaComplex(points=points, precision="fast")
    else:
        ac = gd.AlphaComplex(points=points, weights=np.asarray(weights, dtype=np.float64), precision="fast")
    st = ac.create_simplex_tree()
    filt = list(st.get_filtration())
    vals = np.fromiter((f for _, f in filt), dtype=np.float64, count=len(filt))
    signs = np.fromiter(((-1.0) ** (len(s) - 1) for s, _ in filt),
                        dtype=np.float64, count=len(filt))
    order = np.argsort(vals, kind="stable")
    vals, signs = vals[order], signs[order]
    cum = np.cumsum(signs)
    # GUDHI Alpha filtration values are squared radii.
    idx = np.searchsorted(vals, np.asarray(r_grid) ** 2, side="right") - 1
    chi = np.where(idx >= 0, cum[np.clip(idx, 0, None)], 0.0)
    return np.rint(chi).astype(np.int64)


def cloud_offsets(clouds):
    """Density offsets w_i = s_i^2 - min_j s_j^2 (s_i = median within-cloud
    nearest-neighbour spacing), one scalar per cloud, so that at every grid
    value the sparser clouds carry a fixed squared head start."""
    s2 = []
    for c in clouds:
        d, _ = cKDTree(c).query(c, k=2)
        s2.append(float(np.median(d[:, 1])) ** 2)
    m = min(s2)
    return [np.full(len(c), v - m) for c, v in zip(clouds, s2)]


def mixup_ecp(clouds, r_grid, _cache=None, weights=None):
    r"""k-fold Mixup ECP on the diagonal: Dchi(r) = chi( \cap_i U(X_i; r) ).

    Dual inclusion-exclusion over nonempty S in [k]:
        chi(\cap_i A_i) = sum_S (-1)^{|S|+1} chi(\cup_{i in S} A_i),
    with each union term the ECC of the pooled cloud \cup_{i in S} X_i.

    `weights` (optional, one array per cloud) gives the density-offset
    variant: ball radii sqrt(r^2 + w_i), computed by the weighted Alpha
    complex of the pooled weighted points; the inclusion-exclusion is unchanged.

    `_cache` (optional dict) memoizes ECC terms by frozenset key across calls
    -- used by the permutation test to reuse the permutation-invariant full
    pooled term.
    """
    k = len(clouds)
    out = np.zeros(len(r_grid), dtype=np.int64)
    for size in range(1, k + 1):
        for S in combinations(range(k), size):
            if _cache is not None and S in _cache:
                term = _cache[S]
            else:
                pooled = np.vstack([clouds[i] for i in S])
                w = None if weights is None else np.concatenate([weights[i] for i in S])
                term = ecc_alpha(pooled, r_grid, weights=w)
                if _cache is not None:
                    _cache[S] = term
            out += ((-1) ** (size + 1)) * term
    return out


# --------------------------------------------------------------------------- #
# Scalar summaries and the cancellation guard
# --------------------------------------------------------------------------- #
_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy 2.x rename


def profile_stat(dchi, r_grid, kind="int"):
    r"""Phi(Dchi): 'max' -> max_r |Dchi|;  'int' -> \int |Dchi| dr."""
    a = np.abs(dchi)
    if kind == "max":
        return float(a.max())
    if kind == "int":
        return float(_trapz(a, r_grid))
    if kind == "intnorm":
        # \int |Dchi| dr / r_max: dimensionless.  'int' has length units and
        # tracks the clouds' scale (the r-grid is data-adaptive), so a paired
        # test on 'int' across epochs confounds feature-norm dynamics with
        # interaction; 'intnorm' is invariant to rescaling the clouds.
        return float(_trapz(a, r_grid) / max(float(r_grid[-1]), 1e-12))
    raise ValueError(f"unknown stat kind: {kind}")


def first_interaction_scale(clouds):
    r"""r* = inf{ r : \cap_i U(X_i; r) nonempty }.

    Exact for k = 2 (half the minimum cross-cloud distance).  For k > 2 it is
    the 1-center value  min_z max_i d(z, X_i)  minimized over a candidate set
    (all points + cross-pair midpoints of each cloud pair's closest pairs),
    hence an upper bound; the guard stays conservative.
    """
    k = len(clouds)
    trees = [cKDTree(c) for c in clouds]
    if k == 2:
        d, _ = trees[1].query(clouds[0], k=1)
        return float(d.min()) / 2.0
    cands = [np.vstack(clouds)]
    for i, j in combinations(range(k), 2):
        d, idx = trees[j].query(clouds[i], k=1)
        best = np.argsort(d)[:32]
        cands.append((clouds[i][best] + clouds[j][idx[best]]) / 2.0)
    Z = np.vstack(cands)
    worst = np.max(np.stack([t.query(Z, k=1)[0] for t in trees]), axis=0)
    return float(worst.min())


def cancellation_guard(dchi, r_grid, clouds, tol=0, skip=0, mode="window"):
    """True when Dchi is ~0 on scales where the overlap is provably nonempty.

    Fires on the structural-cancellation failure mode (annulus / torus-shell
    overlaps): r* certifies interaction, Dchi stays silent.  Let L be a quarter
    of the grid points at or beyond r*.
      mode="window":  Dchi == 0 on the L points starting `skip` points after
                      first contact (skip=0 is the campaign's archived window).
      mode="plateau": sampled clouds first meet in isolated contact components,
                      a transient of a few grid points with Dchi > 0; find the
                      first return of Dchi to 0 after contact, require it within
                      L points of r*, and require Dchi == 0 on at least 90% of
                      the L points from there (a blob has Dchi = 1 throughout).
                      Robust to the transient's width and to stray +-1 values.
    """
    rstar = first_interaction_scale(clouds)
    d = np.abs(np.asarray(dchi)); live = np.flatnonzero(np.asarray(r_grid) >= rstar)
    if len(live) == 0:
        return False, rstar
    L = max(1, len(live) // 4)
    if mode == "window":
        win = live[skip: skip + L]
        return (bool(np.all(d[win] <= tol)) if len(win) else False), rstar
    zeros = live[1:][d[live[1:]] <= tol]
    if len(zeros) == 0 or zeros[0] - live[0] > L:
        return False, rstar
    win = np.arange(zeros[0], min(zeros[0] + L, len(d)))
    # a sampled annulus keeps a few stray +-1 values on its plateau; a blob has
    # Dchi = 1 throughout, so "at most 10% nonzero on the window" separates them
    return bool(len(win) == L and np.mean(d[win] > tol) <= 0.1), rstar


# --------------------------------------------------------------------------- #
# Exact one-sided permutation tests + interaction quotient
# --------------------------------------------------------------------------- #
def permutation_test(clouds, r_grid, B=999, stat="int", rng=None, adaptive=False):
    """Both one-sided exact permutation tests and the interaction quotient.

    Returns dict with T_obs, p_up (excess interaction), p_down (separation),
    quotient = T_obs / mean(T_perm), and the permutation replicates.
    The full pooled ECC term is permutation invariant and computed once
    (on the diagonal; with `adaptive=True` the density offsets are recomputed
    from each relabeling, so every term is recomputed).
    """
    rng = np.random.default_rng(rng)
    k = len(clouds)
    sizes = [len(c) for c in clouds]
    pooled = np.vstack(clouds)
    full_key = tuple(range(k))
    w_obs = cloud_offsets(clouds) if adaptive else None
    cache_obs = {} if adaptive else {full_key: ecc_alpha(pooled, r_grid)}
    dchi_obs = mixup_ecp(clouds, r_grid, _cache=cache_obs, weights=w_obs)
    T_obs = profile_stat(dchi_obs, r_grid, stat)

    bounds = np.cumsum([0] + sizes)
    T_perm = np.empty(B)
    for b in range(B):
        perm = rng.permutation(len(pooled))
        parts = [pooled[perm[bounds[i]:bounds[i + 1]]] for i in range(k)]
        if adaptive:
            dchi = mixup_ecp(parts, r_grid, weights=cloud_offsets(parts))
        else:
            cache = {full_key: cache_obs[full_key]}  # invariant term reused
            dchi = mixup_ecp(parts, r_grid, _cache=cache)
        T_perm[b] = profile_stat(dchi, r_grid, stat)

    p_up = (1 + np.sum(T_perm >= T_obs)) / (1 + B)
    p_down = (1 + np.sum(T_perm <= T_obs)) / (1 + B)
    null_mean = float(T_perm.mean())
    return {
        "T_obs": T_obs,
        "p_up": float(p_up),
        "p_down": float(p_down),
        "null_mean": null_mean,
        "quotient": T_obs / null_mean if null_mean > 0 else np.nan,
        "T_perm": T_perm,
        "dchi_obs": dchi_obs,
    }


# --------------------------------------------------------------------------- #
# Paired subsampled comparison (Section `Comparing entanglement across ...')
# --------------------------------------------------------------------------- #
def paired_subsample_test(T_a, T_b):
    """Compare entanglement between conditions a and b from paired fold stats.

    T_a, T_b: arrays of per-fold statistics (same folds through both
    conditions).  Returns the sign-flip p-value (exact for the symmetric
    null; enumerated when F <= 16, else 20000 Monte-Carlo flips), a 95%
    t-interval for E[D], and the mean difference.
    """
    D = np.asarray(T_a, dtype=float) - np.asarray(T_b, dtype=float)
    F = len(D)
    obs = D.mean()
    if F <= 16:
        signs = np.array(np.meshgrid(*([[1, -1]] * F))).T.reshape(-1, F)
        flips = (signs * D).mean(axis=1)
        p_sign = np.mean(np.abs(flips) >= abs(obs) - 1e-12)
    else:
        rng = np.random.default_rng(0)
        signs = rng.choice([1, -1], size=(20000, F))
        flips = (signs * D).mean(axis=1)
        p_sign = (1 + np.sum(np.abs(flips) >= abs(obs) - 1e-12)) / (1 + 20000)
    se = D.std(ddof=1) / np.sqrt(F) if F > 1 else np.inf
    from scipy import stats as sps
    tcrit = sps.t.ppf(0.975, df=max(F - 1, 1))
    return {
        "mean_diff": float(obs),
        "p_sign": float(p_sign),
        "ci95": (float(obs - tcrit * se), float(obs + tcrit * se)),
        "D": D,
    }


def default_r_grid(clouds, num=200, upper_quantile=0.5):
    """Scale grid from 0 to the pooled cloud's diameter quantile / 2."""
    pooled = np.vstack(clouds)
    n = len(pooled)
    idx = np.random.default_rng(0).choice(n, size=min(n, 500), replace=False)
    sub = pooled[idx]
    d = np.sqrt(((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1))
    rmax = np.quantile(d[d > 0], upper_quantile) / 2.0
    return np.linspace(0.0, rmax, num)
