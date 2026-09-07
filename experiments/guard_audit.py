"""Re-evaluate the cancellation guard on every archived pairwise cell, with
the campaign window (skip=0, the archived `guard_fired`) and the refined
window that starts one grid point after first contact (skip=1): sampled
clouds meet in isolated contact components at r*, so the campaign window
could miss an annulus that forms right after contact.

Reads results/measure/*.json for (config, epoch, layer, pair, m_head), rebuilds
the exact clouds from the cached features, recomputes Dchi, and writes
results/guard_audit/<shard>.json; `--report` merges and prints the counts.

Usage: python -m experiments.guard_audit --shard K/N | --report
"""
import argparse
import glob
import json
import os

import numpy as np

from .common import CONFIGS, PCA_DIM, RESULTS_DIR, dump_json
from .ecp import cancellation_guard, default_r_grid, mixup_ecp
from .extract import class_clouds, feature_path, load_features

OUT = os.path.join(RESULTS_DIR, "guard_audit")


def audit_file(path, cfg_by_name):
    rec = json.load(open(path)); cfg = cfg_by_name.get(rec["name"])
    if cfg is None or not os.path.exists(feature_path(cfg, rec["epoch"])):
        return None
    feats, labels = load_features(cfg, rec["epoch"])
    rows = []
    for layer, L in rec["layers"].items():
        if layer not in feats or not L.get("pairs"):
            continue
        F = feats[layer]
        for pair, p in L["pairs"].items():
            c1, c2 = map(int, pair.split(","))
            clouds = class_clouds(F, labels, [c1, c2], m=rec["m_head"], d0=PCA_DIM)
            g = default_r_grid(clouds); d = mixup_ecp(clouds, g)
            f0, rstar = cancellation_guard(d, g, clouds, skip=0, mode="window")
            f1, _ = cancellation_guard(d, g, clouds, skip=1, mode="window")
            fp, _ = cancellation_guard(d, g, clouds, mode="plateau")
            live = np.flatnonzero(g >= rstar)
            rows.append({"name": rec["name"], "epoch": rec["epoch"], "layer": layer, "pair": pair,
                         "archived": bool(p.get("guard_fired", False)), "skip0": f0, "skip1": f1, "plateau": fp,
                         "rstar": float(rstar), "rmax": float(g[-1]), "n_live": int(len(live)),
                         "dchi_live": [int(x) for x in d[live]]})
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--shard", default=None); ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.report:
        rows = [r for f in sorted(glob.glob(os.path.join(OUT, "*.json"))) for r in json.load(open(f))]
        n = len(rows)
        print(f"{n} pairwise cells re-evaluated")
        for key in ("archived", "skip0", "skip1", "plateau"):
            k = sum(r[key] for r in rows); print(f"  {key:9s}: fired {k}")
        mism = [r for r in rows if r["archived"] != r["skip0"]]; print(f"  archived vs recomputed skip0 mismatches: {len(mism)}")
        fired = [r for r in rows if r["plateau"] or r["skip1"]]
        for r in fired[:20]:
            print("   fired:", r["name"], r["epoch"], r["layer"], r["pair"], "skip1" if r["skip1"] else "", "plateau" if r["plateau"] else "", "rstar/rmax %.3f" % (r["rstar"] / r["rmax"]), r["dchi_live"][:12])
        dump_json({"n": n, "fired_archived": sum(r["archived"] for r in rows), "fired_skip0": sum(r["skip0"] for r in rows),
                   "fired_skip1": sum(r["skip1"] for r in rows), "fired_plateau": sum(r["plateau"] for r in rows), "fired_cells": fired}, os.path.join(RESULTS_DIR, "guard_audit.json"))
        return
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "measure", "*.json")))
    files = [f for f in files if not any(t in os.path.basename(f) for t in ("_tripd5", "_tripmid"))]
    k, nsh = map(int, a.shard.split("/")) if a.shard else (0, 1)
    cfg_by_name = {c.name: c for c in CONFIGS}
    out_path = os.path.join(OUT, f"{k}.json")
    rows = json.load(open(out_path)) if os.path.exists(out_path) else []
    done = {(r["name"], r["epoch"]) for r in rows}
    for f in files[k::nsh]:
        rec = json.load(open(f))
        if (rec["name"], rec["epoch"]) in done:
            continue
        r = audit_file(f, cfg_by_name)
        if r is None:
            print("skipped (no features):", os.path.basename(f), flush=True); continue
        rows += r; dump_json(rows, out_path)          # incremental: a timeout loses one file, not the shard
        print("audited", os.path.basename(f), len(r), flush=True)
    print("shard", k, "done", len(rows))


if __name__ == "__main__":
    main()
