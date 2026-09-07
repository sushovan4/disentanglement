"""High-dimensional interaction protocol (Direction A).

The Alpha complex costs O(n^{ceil(d/2)}), so the profile Dchi is only
computable after projecting to small d0.  Proposition (projection is
conservative) says an orthogonal projection is 1-Lipschitz, hence

    r*(PX, PY)  <=  r*(X, Y),

so a projection can inflate apparent overlap but never manufacture
separation.  That asymmetry organizes this module into two tiers:

  AMBIENT TIER  -- exact, no projection, no topology.
      The first-interaction scale r*(X,Y) = 1/2 min ||x-y|| is computable
      directly in R^D at O(n^2 D).  `rstar_test` turns it into an exact
      permutation test for separation, valid at any dimension.

      CAVEAT, measured (see `validate_hd`): this test is exactly calibrated
      but has essentially NO POWER once D is large and the noise is close to
      isotropic, because pairwise distances concentrate: at D=768 a class
      gap of 4 leaves r* indistinguishable from the gap-0 value (ratio
      0.99), and p_sep is non-monotone in the true separation.  Ambient r*
      is then dominated by the noise directions, not by class structure.
      Use this tier as a diagnostic at modest D, not as the high-dimensional
      test.  The projected tier is where class structure is legible.

  PROJECTED TIER -- the topological profile, on an ensemble of projections.
      Dchi carries structure r* cannot see (how the overlap is shaped, not
      merely when it starts), but needs d0 small.  `ensemble_profile` runs
      the profile over K label-independent projections and aggregates.
      Because the ensemble is drawn without reference to labels (random
      projections from a fixed seed; PCA fitted on the pooled sample), any
      aggregate of it remains a valid permutation statistic -- the exact
      tests survive (`ensemble_test`).

`certified_rstar_bound` reports how tight the projected lower bound on the
ambient r* actually is.  Because ambient r* is cheap, this is MEASURED per
representation rather than assumed -- the protocol self-certifies.

What governs that tightness is whether the class separation survives the
projection -- NOT spectral decay, and not ambient dimension.  Synthetic spectra
suggested decay was enough; measured features say otherwise:

    ImageNet ViT-B/16, blocks 3..penult (D=768)    tightness 0.08 - 0.22
    ImageNet ResNet-50, stages 1..4  (D=256..2048) tightness 0.13 - 0.26

so pooled PCA at d0=5 retains under a quarter of the ambient first-interaction
scale on real foundation-model features.  The certificate stays valid -- the
1-Lipschitz argument is unconditional -- but it covers only the small scales.

The natural diagnosis -- that PCA optimizes the wrong objective, maximizing
variance when r* is set by the closest cross-class pair -- turns out NOT to be
the fix.  On synthetic features with the class signal deliberately away from the
top principal directions, three projections aimed at three different objectives
give essentially the same tightness:

    pooled PCA (variance)            0.165
    discriminant (mean separation)   0.139
    max-margin / SVM (margin)        0.136      medians over 6 class pairs

`discriminant_projection` is kept -- it is fitted on a held-out split so the
permutation tests stay exact, and it may be preferable for other purposes --
but it does not rescue the certificate.

OPEN, and worth stating plainly: we do not know why tightness is low, nor what
the best achievable value is over all rank-d0 orthogonal projections.  A
plausible dimensional-accounting story (tightness ~ sqrt(d0/D_eff)) does not
survive testing.  Until that is settled, the honest position is that a d0=5
certificate on a wide representation is VALID but covers only the small scales,
and that no projection choice we have tried changes this.  Always report
`spectral_profile` and the realized `certified_rstar_bound` next to any
high-dimensional measurement.
"""

import numpy as np

from .ecp import default_r_grid, mixup_ecp, profile_stat


# --------------------------------------------------------------------------- #
# Ambient tier: exact, dimension-free
# --------------------------------------------------------------------------- #
def _cross_min_sq(X, Y, chunk=512):
    """min_{x,y} ||x-y||^2 without materializing the (n,m,D) broadcast."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    yy = np.einsum("ij,ij->i", Y, Y)
    best = np.inf
    for i in range(0, len(X), chunk):
        Xc = X[i:i + chunk]
        xx = np.einsum("ij,ij->i", Xc, Xc)
        d2 = xx[:, None] + yy[None, :] - 2.0 * (Xc @ Y.T)
        m = d2.min()
        if m < best:
            best = m
    return max(float(best), 0.0)


def rstar_ambient(X, Y):
    """First interaction scale in the AMBIENT space, exactly, at O(n^2 D).

    U(X;r) and U(Y;r) are disjoint for every r < rstar_ambient(X, Y), in the
    full-dimensional representation -- no projection, no Alpha complex.
    """
    return 0.5 * np.sqrt(_cross_min_sq(X, Y))


def rstar_test(X, Y, B=999, rng=None):
    """Exact permutation test for separation, in ambient dimension.

    Statistic: r* itself.  Under exchangeability the pooled sample re-splits
    into interpenetrating halves, so a genuinely separated pair sits in the
    UPPER tail of the permutation distribution of r*.  Exactness is the same
    rank argument as the profile tests: the statistic is a fixed function of
    the labeled split and the permutations are drawn independently.

    Returns p_sep (separation), and a scale-free separation index
    r*_obs / mean(r*_perm) analogous to the interaction quotient.
    """
    rng = np.random.default_rng(rng)
    n = len(X)
    pooled = np.vstack([X, Y])
    obs = rstar_ambient(X, Y)
    perm = np.empty(B)
    for b in range(B):
        idx = rng.permutation(len(pooled))
        perm[b] = rstar_ambient(pooled[idx[:n]], pooled[idx[n:]])
    p_sep = (1.0 + np.sum(perm >= obs)) / (1.0 + B)
    null_mean = float(perm.mean())
    return {"rstar_obs": float(obs), "p_sep": float(p_sep),
            "null_mean": null_mean,
            "separation_index": float(obs / null_mean) if null_mean > 0 else np.nan,
            "rstar_perm": perm}


# --------------------------------------------------------------------------- #
# Projected tier: label-independent projection ensemble
# --------------------------------------------------------------------------- #
def random_projections(D, d0, K, seed=0):
    """K orthonormal projections R^D -> R^d0, drawn WITHOUT seeing labels.

    Label-independence is what keeps the downstream permutation tests exact:
    the ensemble is a fixed feature of the ambient space, not of the split.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(K):
        Q, _ = np.linalg.qr(rng.normal(size=(D, d0)))
        out.append(Q[:, :d0])
    return out


def pca_projection(pooled, d0):
    """PCA fitted on the POOLED sample: label-invariant, hence permutation-safe."""
    Z = np.asarray(pooled, dtype=np.float64)
    Zc = Z - Z.mean(0)
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    return Vt[:d0].T


def discriminant_projection(fit_X, fit_y, d0, shrink=1e-3):
    """Projection aimed at class SEPARATION rather than total variance.

    PCA maximizes variance, but r* is driven by the closest cross-class pair,
    so PCA can spend its budget on directions along which the classes overlap.
    Measured on foundation-model features, pooled PCA at d0=5 retains only
    8-26% of the ambient first-interaction scale. This fits the top
    generalized-eigenvector directions of the between-class scatter relative
    to the within-class scatter -- the classical discriminant subspace -- which
    is aimed at exactly the quantity r* depends on.

    EXACTNESS. The projection uses labels, so it must not see the labels the
    test will permute. Fit it on a disjoint split (`fit_X`, `fit_y`) and
    measure on the rest: the returned P is then a fixed map, independent of
    the test split's labels, and the permutation argument is unaffected -- the
    same reasoning that licenses pooled PCA, which is label-free.

    Returns a (D, d0) matrix with orthonormal columns, so the 1-Lipschitz
    argument of the projection proposition still applies.
    """
    X = np.asarray(fit_X, dtype=np.float64)
    y = np.asarray(fit_y)
    D = X.shape[1]
    classes = np.unique(y)
    mu = X.mean(0)
    Sw = np.zeros((D, D))
    Sb = np.zeros((D, D))
    for c in classes:
        Xc = X[y == c]
        if len(Xc) < 2:
            continue
        dc = Xc - Xc.mean(0)
        Sw += dc.T @ dc
        m = (Xc.mean(0) - mu)[:, None]
        Sb += len(Xc) * (m @ m.T)
    Sw /= max(len(X) - len(classes), 1)
    Sb /= max(len(X), 1)
    Sw += shrink * np.trace(Sw) / D * np.eye(D)      # ridge: D may exceed n
    # symmetric solve of Sb v = lambda Sw v
    L = np.linalg.cholesky(Sw)
    Li = np.linalg.inv(L)
    M = Li @ Sb @ Li.T
    M = (M + M.T) / 2
    w, V = np.linalg.eigh(M)
    W = Li.T @ V[:, np.argsort(w)[::-1][:d0]]        # top d0 discriminants
    Q, _ = np.linalg.qr(W)                           # orthonormalize: 1-Lipschitz
    return Q[:, :d0]


def spectral_profile(features, d0=5):
    """Is this representation in the regime where a d0 projection is safe?

    Returns the fraction of variance retained at d0, the participation-ratio
    effective rank, and the stable rank.  Rapidly decaying spectra (small
    effective rank, high retained variance) are the regime in which the
    projected first-interaction scale nearly matches the ambient one; close
    to isotropic spectra are the regime in which it does not.
    """
    Z = np.asarray(features, dtype=np.float64)
    Zc = Z - Z.mean(0)
    s = np.linalg.svd(Zc, compute_uv=False)
    ev = s ** 2
    tot = ev.sum()
    if tot <= 0:
        return {"retained": np.nan, "eff_rank": np.nan, "stable_rank": np.nan}
    p = ev / tot
    return {"ambient_dim": int(Z.shape[1]),
            "retained_at_d0": float(p[:d0].sum()),
            "eff_rank": float(1.0 / np.sum(p ** 2)),      # participation ratio
            "stable_rank": float(tot / ev.max())}


def certified_rstar_bound(X, Y, projections):
    """Tightness diagnostic: max_k r*(P_k X, P_k Y) <= r*(X, Y).

    The gap is exactly what the projected tier loses.  Reports the ambient
    truth, the best projected bound, and their ratio.
    """
    amb = rstar_ambient(X, Y)
    per = [rstar_ambient(X @ P, Y @ P) for P in projections]
    best = max(per) if per else 0.0
    return {"rstar_ambient": float(amb), "rstar_bound": float(best),
            "tightness": float(best / amb) if amb > 0 else np.nan,
            "per_projection": [float(v) for v in per]}


def ensemble_profile(clouds, projections, r_grid=None, stat="int",
                     aggregate="mean"):
    """Profile functional aggregated over the projection ensemble.

    Each projection yields its own Dchi and its own scalar Phi(Dchi); the
    ensemble aggregate is the reported statistic.  'min' is the conservative
    choice when the claim is that interaction is PRESENT (the least
    entangled view still shows it); 'mean' is the stable default.
    """
    vals, grids = [], []
    for P in projections:
        proj = [c @ P for c in clouds]
        g = default_r_grid(proj) if r_grid is None else r_grid
        vals.append(profile_stat(mixup_ecp(proj, g), g, stat))
        grids.append(g)
    vals = np.asarray(vals, dtype=float)
    agg = {"mean": vals.mean, "min": vals.min, "max": vals.max}[aggregate]()
    return {"value": float(agg), "per_projection": vals.tolist()}


def ensemble_test(clouds, projections, B=199, stat="int", aggregate="mean",
                  rng=None):
    """Exact permutation test on the ensemble-aggregated profile statistic.

    Valid because `projections` is fixed and label-independent: permuting the
    labels permutes the split but not the ensemble, so the aggregate is an
    ordinary permutation statistic and the rank argument applies unchanged.
    """
    rng = np.random.default_rng(rng)
    sizes = [len(c) for c in clouds]
    bounds = np.cumsum([0] + sizes)
    pooled = np.vstack(clouds)

    obs = ensemble_profile(clouds, projections, stat=stat, aggregate=aggregate)
    T_obs = obs["value"]
    T_perm = np.empty(B)
    for b in range(B):
        idx = rng.permutation(len(pooled))
        parts = [pooled[idx[bounds[i]:bounds[i + 1]]] for i in range(len(clouds))]
        T_perm[b] = ensemble_profile(parts, projections, stat=stat,
                                     aggregate=aggregate)["value"]
    p_up = (1.0 + np.sum(T_perm >= T_obs)) / (1.0 + B)
    p_down = (1.0 + np.sum(T_perm <= T_obs)) / (1.0 + B)
    null_mean = float(T_perm.mean())
    return {"T_obs": float(T_obs), "p_up": float(p_up), "p_down": float(p_down),
            "null_mean": null_mean,
            "quotient": float(T_obs / null_mean) if null_mean > 0 else np.nan,
            "per_projection": obs["per_projection"]}
