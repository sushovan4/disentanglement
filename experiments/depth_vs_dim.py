"""Is the scale window narrowed by DEPTH, or merely by ambient DIMENSION?

The confound.  ResNet-50 shows the cleanest depth gradient (0.607 -> 0.177),
but its width grows with depth too: 256, 512, 1024, 2048.  ViT holds width
fixed across blocks and shows the weakest gradient.  So in the model with the
strongest result the two explanations are perfectly entangled, and in the model
where dimension is controlled the effect is smallest -- exactly the pattern a
pure dimension artifact would produce.

The test.  Project every layer to a COMMON dimension and re-measure.  If the
gradient survives at matched dimension it is depth; if it flattens, the
original curve was tracking width.  This needs no new extraction -- it reuses
the features already on disk.

PCA rather than a random projection, deliberately.  A Johnson-Lindenstrauss
projection approximately preserves pairwise distances, so it would preserve the
window almost by construction and test nothing.  PCA to a fixed k asks the
question that matters: once each layer is given the same number of directions
to express itself in, is the late layer still the concentrated one?

Also reports effective rank (participation ratio of the covariance spectrum),
which is the honest measure of how much dimension a layer actually uses -- the
ambient D overstates it badly for anisotropic features.

Usage:  python -m experiments.depth_vs_dim [--k 32] [--cap 200]
"""

import argparse
import glob
import os

import numpy as np

from .ambient import cross_distance_spread
from .common import RESULTS_DIR, dump_json
from .foundation import FEAT_DIR


def eff_rank(F):
    """Participation ratio of the covariance spectrum."""
    X = F - F.mean(0, keepdims=True)
    ev = np.linalg.eigvalsh(np.cov(X, rowvar=False)) if X.shape[1] <= 1500 \
        else np.linalg.svd(X, compute_uv=False) ** 2 / max(len(X) - 1, 1)
    ev = np.clip(ev, 0, None)
    s = ev.sum()
    return float(s * s / np.sum(ev * ev)) if s > 0 else float("nan")


def pca_to(F, k):
    """Project onto the top-k principal directions of F."""
    X = F - F.mean(0, keepdims=True)
    if X.shape[1] <= k:
        return X
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[:k].T


def window(X, Y):
    s = cross_distance_spread(X, Y)
    return np.nan if s["degenerate"] else s["window_1pct"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--cap", type=int, default=200)
    ap.add_argument("--pairs", type=int, default=6)
    a = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(FEAT_DIR, "*.npz"))):
        z = np.load(path)
        model = os.path.basename(path)[:-4]
        labels = z["labels"]
        classes = sorted(np.unique(labels).tolist())
        rng = np.random.default_rng(0)
        pairs = [tuple(rng.choice(classes, 2, replace=False))
                 for _ in range(a.pairs)]
        layers = [k for k in z.files if k != "labels"]
        for li, layer in enumerate(layers):
            F = z[layer].astype(np.float64)
            full, proj = [], []
            for c1, c2 in pairs:
                X, Y = F[labels == c1][:a.cap], F[labels == c2][:a.cap]
                full.append(window(X, Y))
                P = pca_to(np.vstack([X, Y]), a.k)
                proj.append(window(P[:len(X)], P[len(X):]))
            rows.append({
                "model": model, "layer": layer, "depth_idx": li,
                "n_layers": len(layers), "D": int(F.shape[1]),
                "eff_rank": eff_rank(F[:1000]),
                "window_full": float(np.nanmedian(full)),
                "window_pca": float(np.nanmedian(proj))})
            r = rows[-1]
            print(f"{r['model']:>22} {r['layer']:>13} D={r['D']:>5} "
                  f"eff={r['eff_rank']:>7.1f}  full={r['window_full']:+.3f}  "
                  f"pca{a.k}={r['window_pca']:+.3f}", flush=True)

    dump_json(rows, os.path.join(RESULTS_DIR, "depth_vs_dim.json"))

    print("\n" + "=" * 74)
    print(f"first -> last layer, per model (window at PCA-{a.k}, matched dimension)")
    print("=" * 74)
    print(f"{'model':>22} {'full: first->last':>22} {'pca: first->last':>22}")
    for m in sorted({r["model"] for r in rows}):
        g = [r for r in rows if r["model"] == m and not r["layer"].endswith("_mean")]
        if len(g) < 2:
            continue
        g.sort(key=lambda r: r["depth_idx"])
        f0, f1 = g[0]["window_full"], g[-1]["window_full"]
        p0, p1 = g[0]["window_pca"], g[-1]["window_pca"]
        print(f"{m:>22} {f0:>9.3f} ->{f1:>7.3f} ({f0/f1 if f1 else np.nan:>4.1f}x)"
              f" {p0:>9.3f} ->{p1:>7.3f} ({p0/p1 if p1 else np.nan:>4.1f}x)")
    print("\nIf the pca ratios collapse toward 1.0 while the full ratios stay "
          "large,\nthe gradient was ambient dimension.  If they track each "
          "other, it is depth.")


if __name__ == "__main__":
    main()
