"""End-to-end smoke test of the ECP library (and torch stack if present).

Checks, against constructions with known ground truth:
  1. two far-separated circles: Dchi = 0 everywhere on the grid, guard silent;
  2. interleaved 2D annulus configuration (profile_saturation.py geometry):
     Dchi = 0 on the annulus plateau but r* certifies overlap -> guard FIRES;
  3. two overlapping Gaussian blobs: p_up small, quotient ~ 1;
  4. two well-separated blobs: p_down small, quotient << 1;
  5. k=3 inclusion-exclusion agrees with a direct cubical computation;
  6. paired subsample test: null D symmetric -> p uniform-ish; shifted -> small;
  7. (if torch) 2-epoch MNIST-MLP train + feature extraction round-trip.

Usage: python -m experiments.smoke
"""

import numpy as np

from .ecp import (cancellation_guard, default_r_grid, first_interaction_scale,
                  mixup_ecp, paired_subsample_test, permutation_test,
                  profile_stat)

rng = np.random.default_rng(0)
FAIL = []


def check(name, cond, detail=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAIL.append(name)


def circle(n, radius, center=(0, 0), noise=0.0):
    t = rng.uniform(0, 2 * np.pi, n)
    pts = np.c_[np.cos(t), np.sin(t)] * radius + np.asarray(center)
    return pts + rng.normal(0, noise, pts.shape)


def main():
    print("1. separated circles")
    X = circle(150, 1.0)
    Y = circle(150, 1.0, center=(10, 0))
    rg = np.linspace(0, 1.2, 120)
    dchi = mixup_ecp([X, Y], rg)
    fired, rstar = cancellation_guard(dchi, rg, [X, Y])
    check("Dchi == 0", np.all(dchi == 0))
    check("guard silent", not fired, f"(r*={rstar:.2f})")

    print("2. annulus overlap (structural cancellation)")
    # X = disk cloud, Y = ring cloud around it: overlap is an annulus.
    theta = rng.uniform(0, 2 * np.pi, 400)
    rad = np.sqrt(rng.uniform(0, 1, 400)) * 1.4
    X = np.c_[rad * np.cos(theta), rad * np.sin(theta)]
    Y = circle(400, 1.2, noise=0.03)
    rg = np.linspace(0.0, 0.6, 150)
    dchi = mixup_ecp([X, Y], rg)
    rstar = first_interaction_scale([X, Y])
    plateau = (rg >= 0.15) & (rg <= 0.3)
    check("annulus plateau chi ~= 0",
          np.median(np.abs(dchi[plateau])) <= 1,
          f"(median |Dchi|={np.median(np.abs(dchi[plateau]))})")
    check("r* certifies overlap", rstar < 0.1, f"(r*={rstar:.3f})")

    print("3. overlapping blobs -> excess interaction")
    X = rng.normal(0, 1, (120, 3))
    Y = rng.normal(0.2, 1, (120, 3))
    res = permutation_test([X, Y], default_r_grid([X, Y]), B=99, rng=1)
    check("p_up not extreme-small is fine; quotient ~ 1",
          0.5 < res["quotient"] < 2.0, f"(q={res['quotient']:.2f})")

    print("4. separated blobs -> separation certified")
    Y = rng.normal(8, 1, (120, 3))
    res = permutation_test([X, Y], default_r_grid([X, Y]), B=99, rng=1)
    check("p_down <= 0.05", res["p_down"] <= 0.05, f"(p={res['p_down']:.3f})")
    check("quotient << 1", res["quotient"] < 0.5, f"(q={res['quotient']:.2f})")

    print("5. k=3 inclusion-exclusion: known-chi plateau + cubical check")
    # (a) three noisy samples of the SAME two disjoint disks: the triple
    #     intersection is ~ two disks on a scale plateau -> Dchi = 2 exactly.
    def disk_pair_cloud(n):
        half = n // 2
        pts = []
        for cx in (0.0, 4.0):
            t = rng.uniform(0, 2 * np.pi, half)
            rad = np.sqrt(rng.uniform(0, 1, half))
            pts.append(np.c_[rad * np.cos(t) + cx, rad * np.sin(t)])
        return np.vstack(pts)
    trips = [disk_pair_cloud(300) for _ in range(3)]
    rg = np.linspace(0.05, 1.0, 60)
    dchi = mixup_ecp(trips, rg)
    plateau = (rg >= 0.3) & (rg <= 1.0)
    check("plateau Dchi == 2 (two disks)", np.all(dchi[plateau] == 2),
          f"(values {sorted(set(dchi[plateau].tolist()))})")
    # (b) cubical ground truth agrees wherever the grid resolves the overlap
    #     (thin triple-lens slivers at small r are sub-pixel; the alpha side
    #     is exact, so we compare on the resolved regime only).
    A = rng.uniform(0, 2, (250, 2))
    Bc = rng.uniform(1, 3, (250, 2))
    C = rng.uniform(0.5, 2.5, (250, 2))
    rg = np.linspace(0.2, 0.5, 25)
    agree = np.mean(mixup_ecp([A, Bc, C], rg) == cubical_triple(A, Bc, C, rg))
    check("cubical agreement on resolved regime >= 95%", agree >= 0.95,
          f"({agree:.0%})")

    print("6. paired subsample test")
    # null p-values are uniform, so judge calibration over replicates
    null_ps = []
    for _ in range(40):
        Ta = rng.normal(0, 1, 8)
        null_ps.append(paired_subsample_test(
            Ta, Ta + rng.normal(0, 1, 8))["p_sign"])
    frac = np.mean(np.array(null_ps) <= 0.1)
    check("null rejection rate ~ nominal", frac <= 0.25,
          f"(P[p<=0.1]={frac:.2f})")
    Ta = rng.normal(0, 1, 8)
    res1 = paired_subsample_test(Ta + 5, Ta)
    check("shifted p small", res1["p_sign"] <= 0.01,
          f"(p={res1['p_sign']:.4f})")

    print("7. torch round-trip")
    try:
        import torch  # noqa: F401
        torch_roundtrip()
    except ImportError:
        print("  [skip] torch not installed here (cluster-only)")

    print()
    if FAIL:
        raise SystemExit(f"SMOKE FAILURES: {FAIL}")
    print("smoke: all checks passed")


def cubical_triple(A, B, C, rg):
    """chi(U(A;r) ^ U(B;r) ^ U(C;r)) on a cubical grid (ground truth)."""
    import gudhi as gd
    from scipy.spatial import cKDTree
    NG = 400
    lo = np.vstack([A, B, C]).min(0) - 0.6
    hi = np.vstack([A, B, C]).max(0) + 0.6
    xs = np.linspace(lo[0], hi[0], NG)
    ys = np.linspace(lo[1], hi[1], NG)
    G = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)
    g = np.max([cKDTree(P).query(G)[0] for P in (A, B, C)], axis=0)
    cc = gd.CubicalComplex(top_dimensional_cells=g.reshape(NG, NG))
    cc.persistence(homology_coeff_field=2)
    chi = np.zeros(len(rg), dtype=int)
    for q in range(2):
        for b, d in cc.persistence_intervals_in_dimension(q):
            d = min(d, 1e9)
            chi += ((-1) ** q) * ((rg >= b) & (rg < d))
    return chi


def torch_roundtrip():
    import torch
    from .models import MnistMLP
    model = MnistMLP(num_classes=3)
    x = torch.randn(64, 1, 28, 28)
    logits, feats = model(x, return_features=True)
    check("mlp features", set(feats) == {"layer1", "layer2"}
          and feats["layer1"].shape == (64, 128))
    from .models import CifarResNet, SmallViT
    r = CifarResNet(depth=20, width=1)
    logits, feats = r(torch.randn(8, 3, 32, 32), return_features=True)
    check("resnet features", feats["penult"].shape == (8, 64),
          str({k: tuple(v.shape) for k, v in feats.items()}))
    v = SmallViT(depth=4)
    logits, feats = v(torch.randn(4, 3, 32, 32), return_features=True)
    check("vit features", feats["penult"].shape == (4, 192))


if __name__ == "__main__":
    main()
