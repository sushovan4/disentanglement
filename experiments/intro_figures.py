"""The pictures a reader needs before the definitions.

  intro      two interlocking clouds, their ball unions at three scales with the
             intersection shaded, and the profile Dchi(r) beneath with the
             first-interaction scale r* and the permutation-null band
  blindspot  two concentric rings: the overlap is an annulus, Dchi is zero on
             it, the unguarded test would certify separation; r* says otherwise
  strip      one class pair through the stages of a trained ResNet (2-D shadow
             of the d0=5 clouds), with the certified quotient under each panel;
             needs data/depth_strip.npz from `extract-strip` (run where the
             features are)

Usage: python -m experiments.intro_figures [intro|blindspot|strip|extract-strip|all]
"""
import os
import sys

import numpy as np

from .common import DATA_DIR, RESULTS_DIR
from .ecp import default_r_grid, first_interaction_scale, mixup_ecp, permutation_test, profile_stat

FIGDIR = os.path.join(os.path.dirname(RESULTS_DIR), "figures")
BLUE, ORANGE, DARK = "#3b6fb6", "#e08a2e", "#4a2f5c"


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), bbox_inches="tight", dpi=200)
    print("wrote", os.path.join(FIGDIR, name + ".pdf"))


def _union_masks(A, B, r, xx, yy):
    from scipy.spatial import cKDTree
    P = np.c_[xx.ravel(), yy.ravel()]
    inA = cKDTree(A).query(P, k=1)[0] <= r
    inB = cKDTree(B).query(P, k=1)[0] <= r
    return inA.reshape(xx.shape), inB.reshape(xx.shape)


def _draw_scales(axes, A, B, radii, lim, labels):
    xs = np.linspace(-lim[0], lim[0], 400); ys = np.linspace(-lim[1], lim[1], 400)
    xx, yy = np.meshgrid(xs, ys)
    for ax, r, lab in zip(axes, radii, labels):
        inA, inB = _union_masks(A, B, r, xx, yy)
        ax.contourf(xx, yy, inA.astype(float), levels=[0.5, 1.5], colors=[BLUE], alpha=0.25)
        ax.contourf(xx, yy, inB.astype(float), levels=[0.5, 1.5], colors=[ORANGE], alpha=0.25)
        ax.contourf(xx, yy, (inA & inB).astype(float), levels=[0.5, 1.5], colors=[DARK], alpha=0.85)
        ax.scatter(*A.T, s=3, color=BLUE, lw=0); ax.scatter(*B.T, s=3, color=ORANGE, lw=0)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(lab, fontsize=8)
        for s in ax.spines.values():
            s.set_visible(False)


def _profile_panel(ax, clouds, B=199, seed=0, stat="int"):
    g = default_r_grid(clouds)
    res = permutation_test(clouds, g, B=B, stat=stat, rng=seed)
    d = res["dchi_obs"]; rstar = first_interaction_scale(clouds)
    # null band: profiles of the relabelings
    rng = np.random.default_rng(seed); pooled = np.vstack(clouds); n = len(clouds[0]); prof = []
    for _ in range(B):
        p = rng.permutation(len(pooled)); prof.append(mixup_ecp([pooled[p[:n]], pooled[p[n:]]], g))
    prof = np.array(prof)
    ax.fill_between(g, np.quantile(prof, 0.05, 0), np.quantile(prof, 0.95, 0), color="grey", alpha=0.25, lw=0, label="random relabelings (90% band)")
    ax.plot(g, prof.mean(0), color="grey", lw=0.8, ls="--", label="relabeling mean")
    ax.plot(g, d, color=DARK, lw=1.4, label=r"observed $\Delta\chi(r)$")
    ax.axvline(rstar, color="k", lw=0.7, ls=":"); ax.text(rstar, ax.get_ylim()[0], r" $r^\ast$", fontsize=8, va="bottom")
    ax.set_xlabel("scale $r$"); ax.set_ylabel(r"$\Delta\chi(r)=\chi(\mathcal{U}(A;r)\cap\mathcal{U}(B;r))$")
    return g, d, res, rstar


def intro():
    plt = _mpl()
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=300, noise=0.06, random_state=0)
    A, B = X[y == 0], X[y == 1]
    clouds = [A, B]; rstar = first_interaction_scale(clouds)
    fig = plt.figure(figsize=(6.6, 3.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.08, wspace=0.05)
    top = [fig.add_subplot(gs[0, i]) for i in range(3)]
    radii = [0.6 * rstar, 0.16, 0.32]
    _draw_scales(top, A, B, radii, (1.9, 1.4), [rf"$r={radii[0]:.2f}<r^\ast$: offsets disjoint", rf"$r={radii[1]:.2f}$: overlap is two arcs", rf"$r={radii[2]:.2f}$: overlap is one band"])
    ax = fig.add_subplot(gs[1, :])
    g, d, res, _ = _profile_panel(ax, clouds)
    for r in radii:
        ax.axvline(r, color=DARK, lw=0.5, alpha=0.4)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    for r, lab in zip(radii[1:], (r"$\Delta\chi=2$", r"$\Delta\chi=1$")):
        ax.annotate(lab, (r, float(np.interp(r, g, d))), xytext=(4, 10), textcoords="offset points", fontsize=7, color=DARK)
    ax.set_title(rf"interaction quotient $\widehat E={res['quotient']:.2f}$, separation $p_\downarrow={res['p_down']:.3f}$, excess $p_\uparrow={res['p_up']:.2f}$ ($B=199$)", fontsize=8)
    _save(fig, "intro_profile")


def blindspot():
    plt = _mpl()
    rng = np.random.default_rng(0)
    th = rng.uniform(0, 2 * np.pi, 400)
    A = np.c_[np.cos(th), np.sin(th)]
    th2 = rng.uniform(0, 2 * np.pi, 400)
    B = 1.08 * np.c_[np.cos(th2), np.sin(th2)]
    clouds = [A, B]; rstar = first_interaction_scale(clouds)
    fig = plt.figure(figsize=(6.6, 3.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.08, wspace=0.05)
    top = [fig.add_subplot(gs[0, i]) for i in range(3)]
    radii = [0.6 * rstar, 0.12, 0.3]
    _draw_scales(top, A, B, radii, (1.5, 1.5), [rf"$r={radii[0]:.3f}<r^\ast$: disjoint", rf"$r={radii[1]:.2f}$: annulus, $\chi=0$", rf"$r={radii[2]:.2f}$: annulus, $\chi=0$"])
    ax = fig.add_subplot(gs[1, :])
    g, d, res, _ = _profile_panel(ax, clouds)
    for r in radii:
        ax.axvline(r, color=DARK, lw=0.5, alpha=0.4)
    live = np.flatnonzero(g >= rstar); L = max(1, len(live) // 4)
    z0 = live[1:][np.abs(d[live[1:]]) == 0][0]
    ax.axvspan(g[z0], g[min(z0 + L, len(g) - 1)], color="red", alpha=0.08, lw=0)
    ax.text(g[z0], ax.get_ylim()[1] * 0.72, "  guard window: overlap nonempty,\n  $\\Delta\\chi\\equiv0$ after the contact transient\n  → certificate withheld", fontsize=7, color="darkred", va="top")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.set_title(rf"unguarded test: $p_\downarrow={res['p_down']:.3f}$ would certify separation of two clouds that meet at every $r\geq r^\ast={rstar:.3f}$", fontsize=8)
    _save(fig, "intro_blindspot")


def extract_strip(name="e2_resnet56w1_cifar10_wd0.0005_aug_s0", epoch=80, classes=(3, 5), m=200):
    """Run where features exist: 2-D shadow (PCA on the pair's d0=5 clouds) per layer."""
    import json
    from sklearn.decomposition import PCA
    from .common import CONFIGS, PCA_DIM
    from .extract import class_clouds, load_features
    cfg = {c.name: c for c in CONFIGS}[name]
    feats, labels = load_features(cfg, epoch)
    M = json.load(open(os.path.join(RESULTS_DIR, "measure", f"{name}_epoch{epoch}.json")))["layers"]
    out = {}
    for layer, F in feats.items():
        cl = class_clouds(F, labels, list(classes), m=m, d0=PCA_DIM)
        P = PCA(n_components=2, random_state=0).fit(np.vstack(cl))
        out[f"{layer}_A"] = P.transform(cl[0]); out[f"{layer}_B"] = P.transform(cl[1])
        p = M[layer]["pairs"][f"{classes[0]},{classes[1]}"]
        out[f"{layer}_q"] = np.array([p["quotient"], p["p_down"], p["rstar"]])
    out["layers"] = np.array(list(feats))
    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez(os.path.join(DATA_DIR, "depth_strip.npz"), **out); print("wrote", os.path.join(DATA_DIR, "depth_strip.npz"))


def strip():
    plt = _mpl()
    z = np.load(os.path.join(DATA_DIR, "depth_strip.npz"))
    layers = [l for l in z["layers"] if l != "penult"]
    fig, axes = plt.subplots(1, len(layers), figsize=(6.6, 1.9))
    for ax, l in zip(axes, layers):
        A, B = z[f"{l}_A"], z[f"{l}_B"]; q, p, rs = z[f"{l}_q"]
        ax.scatter(*A.T, s=2.5, color=BLUE, lw=0, alpha=0.8); ax.scatter(*B.T, s=2.5, color=ORANGE, lw=0, alpha=0.8)
        P = np.vstack([A, B]); lo, hi = np.percentile(P, 1, axis=0), np.percentile(P, 99, axis=0)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal", adjustable="box"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(l.replace("stage", "stage "), fontsize=8)
        ax.set_xlabel(rf"$\widehat E={q:.2f}$, $p_\downarrow={p:.3f}$", fontsize=7)
    _save(fig, "intro_strip")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "extract-strip":
        extract_strip()
    else:
        for name in (["intro", "blindspot", "strip"] if which == "all" else [which]):
            try:
                globals()[name]()
            except FileNotFoundError as e:
                print("skipped", name, e)
