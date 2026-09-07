"""Does pairwise dominance hold at initialization, or does training induce it?

The E2 triple scan found the triple quotient at or below the maximal pairwise
quotient in 834/840 dimension-matched deep-layer cells, and the adversarial
search of dominance_search.py showed the geometry alone does not force this
(anisotropic Gaussian triples violate it robustly). So dominance is a property
of *trained* representations -- unless it is already there at initialization,
in which case it is a property of PCA-projected class clouds of natural images
and says nothing about learning. This experiment decides which.

Protocol: rebuild the E3 initialization exactly as train.py does (seeded
build_model, before any optimizer step), push the CIFAR-10 test set through
it, and run the pairwise matrix AND the C(10,3) triple scan at the SAME d0 on
every stage boundary plus the raw normalized input. Report the dominance rate
q_triple <= max q_pair and the ratio distribution, next to the trained cells.

Two entry points, so the ECP side never imports torch (the fork pool after an
OpenMP init deadlocks on macOS):
  python -m experiments.init_dominance extract --seed 0
  python -m experiments.init_dominance measure --seed 0 --d0 4 [--layers ...]
  python -m experiments.init_dominance report
"""

import argparse
import glob
import json
import os
import zlib
from itertools import combinations
from multiprocessing import get_context

import numpy as np

from .common import CONFIGS, POINTS_PER_CLASS, RESULTS_DIR, ROOT, dump_json
from .extract import class_clouds

FEAT_DIR = os.path.join(ROOT, "features", "init")
OUT_DIR = os.path.join(RESULTS_DIR, "init_dominance")
N_JOBS = int(os.environ.get("DISENTANGLE_JOBS", str(os.cpu_count() or 4)))


def e3_config(seed):
    """The E3 ResNet-20 init for `seed`; or any config by index via
    DISENTANGLE_INIT_CONFIG (e.g. 5 = the E2 ResNet-56, 7 = the ViT)."""
    idx = os.environ.get("DISENTANGLE_INIT_CONFIG")
    if idx is not None:
        return CONFIGS[int(idx)]
    return next(c for c in CONFIGS if c.tag == "e3" and c.seed == seed)


def feature_path(cfg):
    return os.path.join(FEAT_DIR, f"{cfg.name}_init.npz")


# --------------------------------------------------------------------- extract
def extract(seed, device=None):
    import torch
    from .data import torch_datasets_for
    from .models import build_model
    cfg = e3_config(seed)
    path = feature_path(cfg)
    if os.path.exists(path):
        print("cached:", path)
        return path
    # identical to train.py up to (not including) the first optimizer step
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    _, ev, te, num_classes = torch_datasets_for(cfg)
    te = ev if cfg.dataset == "cifar100s" else te      # as extract.py does
    model = build_model(cfg, num_classes)
    model.eval()                                  # BN uses running stats = init
    device = device or "cpu"
    model.to(device)
    loader = torch.utils.data.DataLoader(te, batch_size=512, num_workers=0)
    feats, labels = {}, []
    with torch.no_grad():
        for x, y in loader:
            if cfg.arch == "resnet":               # 3072-dim raw input is cheap
                feats.setdefault("input", []).append(x.flatten(1).numpy())
            _, f = model(x.to(device), return_features=True)
            for k, v in f.items():
                feats.setdefault(k, []).append(v.cpu().numpy())
            labels.append(y.numpy())
    out = {k: np.concatenate(v) for k, v in feats.items()}
    out["labels"] = np.concatenate(labels)
    os.makedirs(FEAT_DIR, exist_ok=True)
    np.savez_compressed(path, **out)
    print("wrote:", path, {k: v.shape for k, v in out.items()})
    return path


# --------------------------------------------------------------------- measure
def _job(args):
    from .measure import measure_pair
    key, clouds, B, seed = args
    return key, measure_pair(clouds, B, np.random.default_rng(seed))


def _run(jobs):
    if len(jobs) <= 1 or N_JOBS <= 1:
        return dict(_job(j) for j in jobs)
    with get_context("fork").Pool(min(N_JOBS, len(jobs))) as pool:
        return dict(pool.imap_unordered(_job, jobs, chunksize=1))


def _seed(name, extra):
    return zlib.crc32(f"{name}:init:{extra}".encode())


def measure(seed, d0, layers=None, B=49, m=None):
    cfg = e3_config(seed)
    return measure_npz(feature_path(cfg), f"{cfg.name}_init", d0, layers, B, m)


def measure_npz(path, name, d0, layers=None, B=49, m=None):
    """Pairs + triples at one d0 on any cached feature file (layers + labels).

    Also used for frozen foundation-model features (features/foundation/*.npz),
    where the same dominance score asks whether the pairwise law holds in a
    representation nobody trained here."""
    z = np.load(path)
    labels = z["labels"]
    classes = sorted(np.unique(labels).tolist())
    m = m or min(POINTS_PER_CLASS, min(int(np.sum(labels == c)) for c in classes))
    all_layers = [k for k in z.files if k != "labels"]
    layers = layers or all_layers
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{name}_d0{d0}.json")
    out = json.load(open(out_path)) if os.path.exists(out_path) else {
        "name": name, "epoch": 0, "d0": d0, "B": B, "m": m, "layers": {}}
    for layer in layers:
        if layer in out["layers"]:
            print(f"[{layer}] already measured; skipping")
            continue
        F = z[layer]
        jobs = [(f"{a},{b}", class_clouds(F, labels, [a, b], m=m, d0=d0), B,
                 _seed(name, f"{layer}:{a},{b}"))
                for a, b in combinations(classes, 2)]
        pairs = _run(jobs)
        print(f"[{layer}] pairs done: mean q = "
              f"{np.mean([v['quotient'] for v in pairs.values()]):.3f}", flush=True)
        jobs = [(",".join(map(str, t)), class_clouds(F, labels, list(t), m=m, d0=d0),
                 B, _seed(name, f"tri:{layer}:{t}"))
                for t in combinations(classes, 3)]
        triples = _run(jobs)
        out["layers"][layer] = {"pairs": pairs, "triples": triples}
        dump_json(out, out_path)
        print(f"[{layer}] triples done: {summarize(pairs, triples)}", flush=True)
    return out


def summarize(pairs, triples):
    ratios = []
    for key, t in triples.items():
        a, b, c = key.split(",")
        mp = max(pairs[f"{a},{b}"]["quotient"], pairs[f"{a},{c}"]["quotient"],
                 pairs[f"{b},{c}"]["quotient"])
        ratios.append(t["quotient"] / mp)
    r = np.array(ratios)
    return {"n": int(len(r)), "dominated": int(np.sum(r <= 1)),
            "rate": float(np.mean(r <= 1)), "median_ratio": float(np.median(r)),
            "q90_ratio": float(np.quantile(r, 0.9)), "max_ratio": float(r.max()),
            "pairs_certified": int(sum(v["p_down"] <= 0.05 for v in pairs.values())),
            "mean_pair_q": float(np.mean([v["quotient"] for v in pairs.values()])),
            "mean_triple_q": float(np.mean([v["quotient"] for v in triples.values()]))}


# ---------------------------------------------------------------------- report
def _trained_cells():
    """Dimension-matched trained cells (triple_d0 == the pairwise d0): the
    *_tripd5 rescans (pairs from the base cell, since they ran SKIP_PAIRS) and
    any cell that carries both, e.g. the E10 random-label control."""
    from .common import PCA_DIM
    rows = []
    for f in sorted(glob.glob(os.path.join(RESULTS_DIR, "measure", "*.json"))):
        d = json.load(open(f))
        if not d.get("triples") or d.get("triple_d0") != PCA_DIM:
            continue
        base = d
        if f.endswith("_tripd5.json"):
            base = json.load(open(f.replace("_tripd5", "")))
        for layer, trips in d["triples"].items():
            pairs = base["layers"].get(layer, {}).get("pairs")
            if not pairs:
                continue
            rows.append((d["name"] + (f"@ep{d['epoch']}" if d["name"].startswith("e10") else ""),
                         layer, d["triple_d0"],
                         summarize(pairs, {t["triple"]: t for t in trips})))
    return rows


def report():
    print(f"{'cell':<52} {'layer':<8} {'d0':>2} {'rate':>7} {'med':>6} "
          f"{'q90':>6} {'max':>6} {'q_pair':>7} {'q_tri':>7}")
    def line(name, layer, d0, s):
        print(f"{name:<52} {layer:<8} {d0:>2} {s['dominated']:>3}/{s['n']:<3} "
              f"{s['median_ratio']:>6.3f} {s['q90_ratio']:>6.3f} {s['max_ratio']:>6.3f} "
              f"{s['mean_pair_q']:>7.3f} {s['mean_triple_q']:>7.3f}")
    print("-- initialization (this experiment)")
    summary = {"init": [], "trained": []}
    for f in sorted(glob.glob(os.path.join(OUT_DIR, "*.json"))):
        d = json.load(open(f))
        for layer, L in d["layers"].items():
            s = summarize(L["pairs"], L["triples"])
            line(d["name"], layer, d["d0"], s)
            summary["init"].append({"name": d["name"], "layer": layer, "d0": d["d0"], **s})
    print("-- trained, final epoch, dimension-matched (E2 rescans)")
    for name, layer, d0, s in _trained_cells():
        line(name, layer, d0, s)
        summary["trained"].append({"name": name, "layer": layer, "d0": d0, **s})
    dump_json(summary, os.path.join(RESULTS_DIR, "init_dominance_report.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "measure", "report"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d0", type=int, default=4)
    ap.add_argument("--B", type=int, default=49)
    ap.add_argument("--m", type=int, default=None)
    ap.add_argument("--layers", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--npz", default=None, help="measure this feature file instead")
    ap.add_argument("--name", default=None, help="output name for --npz")
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a.seed, a.device)
    elif a.cmd == "measure" and a.npz:
        measure_npz(a.npz, a.name or os.path.basename(a.npz).split(".")[0], a.d0,
                    a.layers.split(",") if a.layers else None, a.B, a.m)
    elif a.cmd == "measure":
        measure(a.seed, a.d0, a.layers.split(",") if a.layers else None, a.B, a.m)
    else:
        report()
