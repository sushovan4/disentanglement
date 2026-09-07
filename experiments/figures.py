"""Generate the paper's figures from results/*.json (whatever exists).

Each figure is guarded by input existence, so this can run mid-campaign
and will only produce what the data allows.  Outputs land in figures/.

Usage: python -m experiments.figures
"""

import glob
import json
import os
from itertools import combinations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import CONFIGS, RESULTS_DIR, ROOT, checkpoint_epochs

FIGDIR = os.path.join(ROOT, "figures")
os.makedirs(FIGDIR, exist_ok=True)


def _load(name):
    path = os.path.join(RESULTS_DIR, name)
    return json.load(open(path)) if os.path.exists(path) else None


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"),
                    bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote figures/{name}.pdf")


# --------------------------------------------------------------------------- #
def e1_heatmap(r):
    M = np.full((10, 10), np.nan)
    for key, v in r["reference"].items():
        a, b = map(int, key.split(","))
        M[a, b] = M[b, a] = v["quotient"]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(M, cmap="viridis")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("digit"); ax.set_ylabel("digit")
    for i, j in combinations(range(10), 2):
        if M[i, j] >= 0.25:  # annotate the entangled pairs
            for (a, b) in ((i, j), (j, i)):
                ax.text(b, a, f"{M[a, b]:.2f}", ha="center", va="center",
                        color="white", fontsize=7)
    fig.colorbar(im, ax=ax, label=r"interaction quotient $\widehat{E}$")
    rho = r["confusion_spearman"]["rho"]
    ax.set_title(rf"MNIST pairwise Intersection ECP  ($\rho_{{conf}}={rho:.2f}$)")
    _save(fig, "e1_heatmap")


def e1_stability(r):
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    d0s = sorted({s["d0"] for s in r["stability"]})
    ms = sorted({s["m"] for s in r["stability"]})
    for d0 in d0s:
        xs, ys, es = [], [], []
        for m in ms:
            vals = [s["spearman_vs_ref"] for s in r["stability"]
                    if s["d0"] == d0 and s["m"] == m]
            if vals:
                xs.append(m); ys.append(np.mean(vals)); es.append(np.std(vals))
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3,
                    label=rf"$d_0={d0}$")
    ax.set_xscale("log"); ax.set_xticks(ms); ax.set_xticklabels(ms)
    ax.set_xlabel("points per class $m$")
    ax.set_ylabel("Spearman vs. reference ranking")
    ax.axhline(1.0, color="gray", lw=0.6, ls=":")
    ax.legend(ncol=2, fontsize=8)
    _save(fig, "e1_stability")


# --------------------------------------------------------------------------- #
LAYER_ORDER = ["input", "stem", "stage1", "stage2", "stage3",
               "block2", "block3", "block4", "block6", "block8", "block9", "penult"]


def _layer_key(l):
    return (LAYER_ORDER.index(l) if l in LAYER_ORDER else 99, l)


def _short(name):
    arch = ("ViT-Tiny" if "vit8" in name else "ViT-B/16" if "vit_b16" in name
            else "ResNet-56" if "resnet56" in name else "ResNet-20")
    arch = arch.replace("ResNet-", "R").replace("ViT-Tiny", "ViT")
    return arch + ("/C100" if "cifar100s" in name else "/C10")


def _pretty(name):
    """Run-name -> legend label."""
    arch = ("ViT-Tiny" if "vit8" in name else "ViT-B/16" if "vit_b16" in name
            else "ResNet-56" if "resnet56" in name else "ResNet-20")
    data = "CIFAR-100 (10-class subset)" if "cifar100s" in name else "CIFAR-10"
    return f"{arch}, {data}"


def _measure_files(tag):
    out = {}
    for cfg in [c for c in CONFIGS if c.tag == tag]:
        for e in checkpoint_epochs(cfg):
            p = os.path.join(RESULTS_DIR, "measure",
                             f"{cfg.name}_epoch{e}.json")
            if os.path.exists(p):
                out[(cfg.name, e)] = json.load(open(p))
    return out


def e2_depth():
    data = _measure_files("e2")
    finals = {}
    for cfg in [c for c in CONFIGS if c.tag == "e2"]:
        last = checkpoint_epochs(cfg)[-1]
        if (cfg.name, last) in data:
            finals[cfg.name] = data[(cfg.name, last)]
    if not finals:
        return False
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for name, rec in finals.items():
        layers = list(rec["layers"])
        ys = [np.mean([p["quotient"] for p in rec["layers"][l]["pairs"].values()])
              for l in layers]
        ax.plot(range(len(layers)), ys, marker="o", label=_pretty(name))
    ax.set_xticks(range(5)); ax.set_xticklabels(["stem / block 2", "stage 1 / block 4", "stage 2 / block 6", "stage 3 / block 8", "penultimate"], fontsize=7)
    ax.set_xlabel("depth $\\rightarrow$ (ResNet stage / ViT block)")
    ax.set_ylabel(r"mean pairwise $\widehat{E}_\ell$")
    ax.legend(fontsize=7)
    _save(fig, "e2_depth")

    # triple report: joint vs max-pairwise, escalated values only
    rows = []
    for name, rec in finals.items():
        for layer, trips in rec.get("triples", {}).items():
            pairs = rec["layers"][layer]["pairs"]
            for t in trips:
                if "headline_quotient" not in t:
                    continue
                cs = t["triple"].split(",")
                pq = [pairs[k]["quotient"] for k in
                      (f"{a},{b}" for a, b in combinations(sorted(map(int, cs)), 2))
                      if k in pairs]
                rows.append((name, layer, t["triple"],
                             t["headline_quotient"], max(pq) if pq else np.nan))
    if rows:
        rows.sort(key=lambda r: -(r[3] - r[4]))
        with open(os.path.join(FIGDIR, "e2_triples.txt"), "w") as f:
            f.write("model layer triple joint_quotient max_pairwise\n")
            for r in rows[:40]:   # the listing keeps the 40 largest gaps; the figure uses all
                f.write(f"{r[0]} {r[1]} ({r[2]}) {r[3]:.3f} {r[4]:.3f}\n")
        print("wrote figures/e2_triples.txt (top joint-vs-pairwise gaps)")
        # the full ratio distribution, one strip per (model, layer) cell,
        # against the measured null floor of E9 (dominance_null.json)
        cells = {}
        for r in rows:
            if np.isfinite(r[4]) and r[4] > 0:
                cells.setdefault((r[0], r[1]), []).append(r[3] / r[4])
        # a ResNet's penultimate clouds are its stage-3 clouds: show once
        for k in [k for k in cells if k[1] == "penult" and (k[0], "stage3") in cells]:
            del cells[k]
        keys = sorted(cells, key=lambda k: (k[0], _layer_key(k[1])))
        null = _load("dominance_null.json")
        fig, ax = plt.subplots(figsize=(6.4, 3.0))
        rng = np.random.default_rng(0)
        ax.axvspan(-0.5, 3.5, color="C1", alpha=0.04, lw=0); ax.axvspan(8.5, 13.5, color="C3", alpha=0.04, lw=0)
        for x, k in enumerate(keys):
            v = np.array(cells[k])
            ax.scatter(x + rng.uniform(-0.18, 0.18, len(v)), v, s=5, alpha=0.5,
                       color="C3" if "vit" in k[0] else ("C1" if "cifar100s" in k[0] else "C0"))
            ax.plot([x - 0.25, x + 0.25], [np.median(v)] * 2, color="k", lw=1.2)
        if null:
            meds = [c["median_ratio"] for c in null["gauss"] + null["shuffle"]]
            ax.axhspan(min(meds), max(meds), color="grey", alpha=0.2, lw=0)
            ax.text(len(keys) - 0.5, max(meds) + 0.02, "null-floor medians", fontsize=6, ha="right", color="dimgrey")
        ax.axhline(1.0, color="k", lw=0.6, ls=":")
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([k[1].replace("stage", "stage ").replace("block", "block ").replace("penult", "penult.") for k in keys], fontsize=6, rotation=45, ha="right")
        ytop = ax.get_ylim()[1]
        for x0, x1, lab in ((-0.5, 3.5, "ResNet-56 / CIFAR-100 subset"), (3.5, 8.5, "ResNet-56 / CIFAR-10"), (8.5, 13.5, "ViT-Tiny / CIFAR-10")):
            ax.text((x0 + x1) / 2, ytop, lab, fontsize=6.5, ha="center", va="bottom", color="dimgrey")
        ax.set_ylabel(r"$\widehat E_{\rm triple}\,/\,\max_{\rm pairs}\widehat E$")
        _save(fig, "e2_triple")
    return True


def e3_trajectories():
    rows = _load("e3_dynamics.json")
    if not rows:
        return False
    # E3 proper is the five cosine-schedule seeds. The LR-schedule ablation
    # (tag 'ablr') shares seeds 0 and 1 with them and lives in the same file;
    # grouping by seed alone stitched its checkpoints into the E3 curves, which
    # drew lines that doubled back in epoch. Group by config, E3 only.
    e3 = lambda r: r.get("config", "").startswith("e3_")
    traj = [r for r in rows if r.get("kind") == "trajectory" and e3(r)]
    # significance marks come from the scale-free paired analysis when present
    norm = _load("e3_dynamics_ecp.json") or _load("e3_dynamics_norm.json")
    sig = [r for r in (norm or rows) if r.get("bh_reject") and e3(r)]
    layers = sorted({r["layer"] for r in traj}, key=_layer_key)
    if "stage3" in layers and "penult" in layers:   # same clouds: show once
        layers.remove("penult")
    fig, axes = plt.subplots(1, len(layers), figsize=(2.6 * len(layers), 3.0),
                             sharey=True)
    for ax, layer in zip(np.atleast_1d(axes), layers):
        for config in sorted({r["config"] for r in traj}):
            pts = sorted([(r["epoch"], r["mean_quotient"]) for r in traj
                          if r["layer"] == layer and r["config"] == config])
            if pts:
                ax.plot(*zip(*pts), lw=0.9, alpha=0.8)
        # one mark per BH-significant step, weighted by how many of the 45x5
        # (pair, seed) tests fired: decreases red, increases grey
        for e in sorted({r["epoch_to"] for r in sig if r["layer"] == layer}):
            n_dn = sum(1 for r in sig if r["layer"] == layer and r["epoch_to"] == e and r["mean_diff"] < 0)
            n_up = sum(1 for r in sig if r["layer"] == layer and r["epoch_to"] == e and r["mean_diff"] > 0)
            if n_dn:
                ax.axvline(e, color="red", lw=0.4 + 2.0 * n_dn / 225, alpha=0.35)
            if n_up:
                ax.axvline(e, color="grey", lw=0.4 + 2.0 * n_up / 225, alpha=0.35, ls="--")
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xticks([0, 1, 2, 4, 8, 16, 32, 80]); ax.set_xticklabels(["0", "1", "2", "4", "8", "16", "32", "80"], fontsize=7)
        ax.minorticks_off()
        ax.set_title(layer, fontsize=9)
        ax.set_xlabel("epoch")
    np.atleast_1d(axes)[0].set_ylabel(r"mean pairwise $\widehat{E}_\ell$")
    _save(fig, "e3_trajectories")
    return True


def e4_calibration():
    r = _load("e4_calibration.json")
    if not r:
        return False
    alphas = np.linspace(0.005, 0.25, 50)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    for block, ps_key, label in (("permutation", "p_up", r"$p_\uparrow$"),
                                 ("permutation", "p_down", r"$p_\downarrow$"),
                                 ("signflip", "p_sign", "sign-flip")):
        ps = np.array(r[block][ps_key])
        ax.plot(alphas, [(ps <= a).mean() for a in alphas], label=label)
    ax.plot([0, 0.25], [0, 0.25], color="gray", lw=0.7, ls=":")
    ax.set_xlabel(r"nominal level $\alpha$")
    ax.set_ylabel("empirical rejection rate (null)")
    ax.legend(fontsize=8)
    g = r.get("guard", {})
    if g.get("rate") is not None:
        ax.set_title(f"guard fired {g['n_guard_fired']}/{g['n_measurements']}"
                     f" ({100 * g['rate']:.2f}%)", fontsize=9)
    _save(fig, "e4_calibration")
    return True


def e5_table():
    r = _load("e5_predict.json")
    if not r:
        return False
    lines = [r"\begin{tabular}{llrrr}", r"\toprule",
             r"target & features & $n_{\mathrm{feat}}$ & LOO $R^2$ & Spearman \\",
             r"\midrule"]
    for row in r["results"]:
        tgt = row["target"].replace("_", r"\_")
        fam = row["family"].replace("_", r"\_")
        lines.append(f"{tgt} & {fam} & "
                     f"{row['n_features']} & {row['loo_r2']:.3f} & "
                     f"{row['spearman']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(FIGDIR, "e5_table.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote figures/e5_table.tex")
    return True


def _e7_cells():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "e7", "*.json")))
    return [json.load(open(f)) for f in files]


_E7_SHORT = {"e3_resnet20w1_cifar10_wd0.0005_aug_s0": "ResNet-20 / C10",
             "e2_resnet56w1_cifar10_wd0.0005_aug_s0": "ResNet-56 / C10",
             "e2_resnet56w1_cifar100s_wd0.0005_aug_s0": "ResNet-56 / C100s",
             "e2_vit8w1_cifar10_wd0.05_aug_s0": "ViT-8 / C10"}


def e7_spectrum():
    """Quotient of the interaction floors C_j vs j, per layer, per model.

    Solid lines: j=2..5 from the 5-class subsets (mean over the three
    subsets).  Isolated markers: j=8,9,10 top floors of all ten classes
    (B=19 -- magnitudes, not certifications).  Log y: geometric decay
    shows as straight lines.
    """
    cells = _e7_cells()
    if not cells:
        return False
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.4), sharex=True, sharey=True)
    for ax, d in zip(axes.flat, cells):
        layers = list(d["layers"])
        colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(layers)))
        for col, lay in zip(colors, layers):
            rec = d["layers"][lay]
            byj = {}
            for floors in rec["subsets5"].values():
                for j, r in floors.items():
                    if r.get("quotient") is not None:
                        byj.setdefault(int(j), []).append(r["quotient"])
            js = sorted(byj)
            ax.plot(js, [np.mean(byj[j]) for j in js], "o-", color=col,
                    label=lay, ms=3.5, lw=1.4)
            k10 = {int(j): r["quotient"] for j, r in rec.get("k10", {}).items()
                   if isinstance(r, dict) and r.get("quotient") is not None}
            if k10:
                js10 = sorted(k10)
                ax.plot(js10, [k10[j] for j in js10], "^", color=col,
                        ms=4.5, mfc="none")
        ax.set_yscale("log")
        ax.set_title(_E7_SHORT.get(d["name"], d["name"]), fontsize=9)
        ax.grid(alpha=0.25, lw=0.4)
        ax.legend(fontsize=6, frameon=False, ncol=2)
    for ax in axes[1]:
        ax.set_xlabel(r"floor order $j$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"quotient $\widehat{E}(C_j)$")
    fig.suptitle("E7: the interaction spectrum decays geometrically in $j$, "
                 "fastest in deep layers", fontsize=10)
    _save(fig, "e7_spectrum")
    return True


def e7_onsets():
    """Onset delay profiles: r*_j normalized by the pairwise onset r*_2."""
    cells = _e7_cells()
    if not cells:
        return False
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.4), sharex=True, sharey=True)
    for ax, d in zip(axes.flat, cells):
        layers = list(d["layers"])
        colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(layers)))
        for col, lay in zip(colors, layers):
            on = d["layers"][lay]["onsets"]
            vals = ([on[k] for k in sorted(on, key=int)]
                    if isinstance(on, dict) else list(on))
            vals = np.asarray(vals, dtype=float)
            if vals.size == 0 or vals[0] <= 0:
                continue
            js = np.arange(2, 2 + vals.size)
            ax.plot(js, vals / vals[0], "o-", color=col, label=lay,
                    ms=3.5, lw=1.4)
        ax.set_title(_E7_SHORT.get(d["name"], d["name"]), fontsize=9)
        ax.grid(alpha=0.25, lw=0.4)
        ax.legend(fontsize=6, frameon=False, ncol=2)
    for ax in axes[1]:
        ax.set_xlabel(r"floor order $j$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"onset delay $r^*_j / r^*_2$")
    fig.suptitle("E7: higher-order overlaps onset at ever larger scales "
                 "relative to the pairwise onset", fontsize=10)
    _save(fig, "e7_onsets")
    return True


def main():
    made = []
    r1 = _load("e1_mnist.json")
    if r1:
        e1_heatmap(r1); e1_stability(r1); made.append("e1")
    if e2_depth():
        made.append("e2")
    if e3_trajectories():
        made.append("e3")
    if e4_calibration():
        made.append("e4")
    if e5_table():
        made.append("e5")
    if e7_spectrum() and e7_onsets():
        made.append("e7")
    print(f"generated: {made or 'nothing (no results present)'}")


if __name__ == "__main__":
    main()
