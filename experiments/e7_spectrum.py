"""E7 --- the interaction spectrum: floors chi(C_j) = 'at least j classes meet'.

Foundations paper, Remark on the interaction spectrum: the descending chain
C_j(r) = union_{|S|=j} cap_{i in S} U(X_i; r) has pointwise-Euler floors
chi(C_j) = int phi_j dchi, phi_j = max_{|S|=j} prod_{i in S} eps_i.

Computation is exact and torch-free, via Jordan's sieve on subset profiles:
    1[cov >= j] = sum_{m>=j} (-1)^{m-j} C(m-1, j-1) * C(cov, m)
so, integrating against dchi,
    chi(C_j) = sum_{m>=j} (-1)^{m-j} C(m-1, j-1) * S_m,
    S_m      = sum_{|S|=m} Dchi_S,
and every Dchi_S expands over pooled-cloud ECCs (one shared cache of
2^k - 1 Alpha complexes; the all-classes ECC is permutation-invariant).

Per (model, layer) this driver reports, with permutation quotients:
  * the full spectrum j = 2..5 on three FIXED 5-class subsets (B=99);
  * the top floors j in {8, 9, 10} of ALL ten classes (B=19 -- each
    permutation costs 2^10 - 2 ECC recomputations);
  * the onset vector r*_j = inf{ r : C_j nonempty }, j = 2..10, computed
    exactly-enough from the j-th smallest cloud-distance over candidates.

Usage:  python -m experiments.e7_spectrum <config_index> <epoch>
        (intended: the three e2 finals + e3 seed 0 final; see run_e7.sbatch)
"""

import os
import sys
import zlib
from itertools import combinations
from math import comb
from multiprocessing import get_context

import numpy as np
from scipy.spatial import cKDTree

from .common import (CONFIGS, PCA_DIM, POINTS_PER_CLASS, RESULTS_DIR, STAT,
                     dump_json)
from .ecp import default_r_grid, ecc_alpha, profile_stat
from .extract import class_clouds, load_features

N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", str(os.cpu_count() or 4)))
SUBSETS_5 = [(2, 3, 4, 5, 7),   # bird cat deer dog horse (animals)
             (0, 1, 8, 9, 6),   # airplane auto ship truck + frog
             (1, 3, 5, 7, 9)]   # odd classes
B_K5 = int(os.environ.get("DISENTANGLE_E7_B5", "99"))
B_K10 = int(os.environ.get("DISENTANGLE_E7_B10", "19"))


# --------------------------------------------------------------------------- #
# floors from pooled ECCs (bitmask algebra)
# --------------------------------------------------------------------------- #
def _ecc_job(args):
    mask, pts, r_grid = args
    return mask, ecc_alpha(pts, r_grid)


def pooled_eccs(clouds, r_grid, pool, fixed=None):
    """ECC of the pooled cloud for every nonempty subset mask (cached full)."""
    k = len(clouds)
    jobs = []
    out = dict(fixed or {})
    for mask in range(1, 2 ** k):
        if mask in out:
            continue
        pts = np.vstack([clouds[i] for i in range(k) if mask >> i & 1])
        jobs.append((mask, pts, r_grid))
    if pool is None:
        out.update(_ecc_job(j) for j in jobs)
    else:
        out.update(pool.imap_unordered(_ecc_job, jobs, chunksize=4))
    return out


def floors_from_eccs(eccs, k, n_grid):
    """chi(C_j) for j = 2..k via subset-sum + Jordan's sieve."""
    dchi = {}
    for mask in range(1, 2 ** k):
        # Dchi_mask = sum over nonempty submasks R: (-1)^{|R|+1} ECC_R
        total = np.zeros(n_grid, dtype=np.int64)
        sub = mask
        while sub:
            total += ((-1) ** (bin(sub).count("1") + 1)) * eccs[sub]
            sub = (sub - 1) & mask
        dchi[mask] = total
    S = {m: np.zeros(n_grid, dtype=np.int64) for m in range(1, k + 1)}
    for mask, v in dchi.items():
        S[bin(mask).count("1")] += v
    floors = {}
    for j in range(2, k + 1):
        f = np.zeros(n_grid, dtype=np.int64)
        for m in range(j, k + 1):
            f += ((-1) ** (m - j)) * comb(m - 1, j - 1) * S[m]
        floors[j] = f
    return floors


def spectrum_with_quotients(clouds, r_grid, B, rng, pool, js=None):
    """Observed floors + permutation-mean-normalized quotients per floor."""
    k = len(clouds)
    full = (1 << k) - 1
    eccs = pooled_eccs(clouds, r_grid, pool)
    obs = floors_from_eccs(eccs, k, len(r_grid))
    js = js or list(range(2, k + 1))
    T_obs = {j: profile_stat(obs[j], r_grid, STAT) for j in js}

    pooled = np.vstack(clouds)
    sizes = [len(c) for c in clouds]
    bounds = np.cumsum([0] + sizes)
    null_sum = {j: 0.0 for j in js}
    ge = {j: 0 for j in js}
    le = {j: 0 for j in js}
    for _ in range(B):
        perm = rng.permutation(len(pooled))
        parts = [pooled[perm[bounds[i]:bounds[i + 1]]] for i in range(k)]
        eccs_p = pooled_eccs(parts, r_grid, pool, fixed={full: eccs[full]})
        fl = floors_from_eccs(eccs_p, k, len(r_grid))
        for j in js:
            t = profile_stat(fl[j], r_grid, STAT)
            null_sum[j] += t
            ge[j] += t >= T_obs[j]
            le[j] += t <= T_obs[j]
    out = {}
    for j in js:
        nm = null_sum[j] / max(B, 1)
        out[j] = {"T_obs": T_obs[j],
                  "quotient": T_obs[j] / nm if nm > 0 else np.nan,
                  "null_mean": nm,
                  "p_up": (1 + ge[j]) / (1 + B),
                  "p_down": (1 + le[j]) / (1 + B)}
    return out


def onset_vector(clouds):
    """r*_j = inf{r : some point within r of >= j clouds}, j = 2..k.

    Evaluated over candidates (all points + closest cross-pair midpoints):
    at each candidate z take the j-th smallest of d(z, X_i); minimize over z.
    Upper bound; exact for j=2 up to the candidate set.
    """
    k = len(clouds)
    trees = [cKDTree(c) for c in clouds]
    cands = [np.vstack(clouds)]
    for i, jdx in combinations(range(k), 2):
        d, idx = trees[jdx].query(clouds[i], k=1)
        best = np.argsort(d)[:16]
        cands.append((clouds[i][best] + clouds[jdx][idx[best]]) / 2.0)
    Z = np.vstack(cands)
    D = np.stack([t.query(Z, k=1)[0] for t in trees], axis=1)
    D.sort(axis=1)
    return {j: float(D[:, j - 1].min()) for j in range(2, k + 1)}


# --------------------------------------------------------------------------- #
def main(cfg, epoch):
    feats, labels = load_features(cfg, epoch)
    classes = sorted(np.unique(labels).tolist())
    m = min(POINTS_PER_CLASS, min(int(np.sum(labels == c)) for c in classes))
    rng = np.random.default_rng(zlib.crc32(f"e7:{cfg.name}:{epoch}".encode()))
    out = {"name": cfg.name, "epoch": epoch, "layers": {}}
    ctx = get_context("fork")
    with ctx.Pool(N_JOBS) as pool:
        for layer, F in feats.items():
            rec = {"subsets5": {}, "k10": None, "onsets": None}
            # full spectrum on fixed 5-class subsets
            for sub in SUBSETS_5:
                if not set(sub) <= set(classes):
                    continue
                clouds = class_clouds(F, labels, list(sub), m=m, d0=PCA_DIM)
                r_grid = default_r_grid(clouds)
                rec["subsets5"][",".join(map(str, sub))] = \
                    spectrum_with_quotients(clouds, r_grid, B_K5, rng, pool)
            # top floors of all ten classes
            clouds10 = class_clouds(F, labels, classes, m=m, d0=PCA_DIM)
            r_grid = default_r_grid(clouds10)
            rec["k10"] = spectrum_with_quotients(
                clouds10, r_grid, B_K10, rng, pool, js=[8, 9, 10])
            rec["onsets"] = onset_vector(clouds10)
            out["layers"][layer] = rec
            print(f"[{cfg.name} ep{epoch}] e7 {layer} done", flush=True)
    dump_json(out, os.path.join(RESULTS_DIR, "e7",
                                f"{cfg.name}_epoch{epoch}.json"))


if __name__ == "__main__":
    cfg = CONFIGS[int(sys.argv[1])]
    main(cfg, int(sys.argv[2]))
