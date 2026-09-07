"""E8: sensitivity of the paper's conclusions to the PCA dimension d0.

The whole campaign measures on a d0=5 PCA shadow (Limitations).  E1 sweeps
d0 on MNIST only; this experiment re-measures the headline DEPTH conclusions
at d0 in {3, 4, 6} on the E2 models (ResNet-56 x2, ViT) and the pretrained
ViT-B/16 finetune, final epochs, full pairwise matrices with B=199.

Per (config, d0): per layer, all 45 pair quotients + p_down; reported
against the d0=5 reference cell (results/measure/<name>_epoch<k>.json):
  * Spearman correlation of the 45-pair quotient vector vs. reference,
    per layer (does d0 change WHICH pairs are entangled?);
  * mean quotient per layer (does d0 change the depth profile shape?);
  * certified-separated counts (does d0 change what is certifiable?).

One (config, d0) per array task.  Features are loaded ONE LAYER AT A TIME
(np.load is lazy per key), so this runs in ordinary memory even for the
ViT-B/16 cells that OOM'd the all-layers loader in measure.py.

Usage:
  python -m experiments.e8_d0 <task_index>     # task = config x d0 cell
  python -m experiments.e8_d0 --manifest
"""

import os
import sys
import zlib
from itertools import combinations

import numpy as np

from .common import CONFIGS, POINTS_PER_CLASS, RESULTS_DIR, checkpoint_epochs, dump_json
from .extract import class_clouds, extract_checkpoint, feature_path
from .measure import _run_jobs, B_SWEEP

D0_GRID = (3, 4, 6)  # 5 is the campaign reference, already measured
TAGS = ("e2", "vitpre")


def cells():
    out = []
    for i, cfg in enumerate(CONFIGS):
        if cfg.tag in TAGS:
            for d0 in D0_GRID:
                out.append((i, checkpoint_epochs(cfg)[-1], d0))
    return out


def measure_cell(cfg, epoch, d0):
    extract_checkpoint(cfg, epoch)
    z = np.load(feature_path(cfg, epoch))
    labels = z["labels"]
    classes = sorted(np.unique(labels).tolist())
    counts = {c: int(np.sum(labels == c)) for c in classes}
    m = min(POINTS_PER_CLASS, min(counts.values()))

    out = {"name": cfg.name, "epoch": epoch, "d0": d0, "m": m, "layers": {}}
    for layer in [k for k in z.files if k != "labels"]:
        F = z[layer]  # lazy: only this layer resident
        jobs = []
        for c1, c2 in combinations(classes, 2):
            clouds = class_clouds(F, labels, [c1, c2], m=m, d0=d0)
            jobs.append((f"{c1},{c2}", clouds, B_SWEEP,
                         zlib.crc32(f"e8:{cfg.name}:{epoch}:{d0}:{layer}:{c1},{c2}".encode())))
        out["layers"][layer] = {"pairs": _run_jobs(jobs)}
        del F
        print(f"[{cfg.name} ep{epoch} d0={d0}] layer {layer} done", flush=True)

    dump_json(out, os.path.join(RESULTS_DIR, "e8",
                                f"{cfg.name}_epoch{epoch}_d0{d0}.json"))
    return out


def report():
    """Aggregate: compare each e8 cell against its d0=5 reference."""
    import glob
    import json
    from scipy.stats import spearmanr
    rows = []
    for f in sorted(glob.glob(os.path.join(RESULTS_DIR, "e8", "*.json"))):
        d = json.load(open(f))
        ref_path = os.path.join(RESULTS_DIR, "measure",
                                f"{d['name']}_epoch{d['epoch']}.json")
        ref = json.load(open(ref_path)) if os.path.exists(ref_path) else None
        for layer, rec in d["layers"].items():
            keys = sorted(rec["pairs"])
            q = [rec["pairs"][k]["quotient"] for k in keys]
            cert = sum(1 for k in keys if rec["pairs"][k]["p_down"] <= 0.05)
            row = {"name": d["name"], "layer": layer, "d0": d["d0"],
                   "mean_q": float(np.mean(q)), "certified": cert,
                   "n_pairs": len(keys)}
            if ref and layer in ref["layers"]:
                rq = [ref["layers"][layer]["pairs"][k]["quotient"] for k in keys]
                row["spearman_vs_d05"] = float(spearmanr(q, rq)[0])
            rows.append(row)
            print(row)
    dump_json(rows, os.path.join(RESULTS_DIR, "e8_d0_report.json"))


if __name__ == "__main__":
    if sys.argv[1] == "--manifest":
        for i, e, d0 in cells():
            print(i, e, d0)
        sys.exit(0)
    if sys.argv[1] == "--report":
        report()
        sys.exit(0)
    i, epoch, d0 = cells()[int(sys.argv[1])]
    measure_cell(CONFIGS[i], epoch, d0)
