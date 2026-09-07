"""Calibration and on-the-fly corruption robustness for the E5 population.

For each config (default: the 96 'pop' models) at its final checkpoint:
test accuracy, negative log-likelihood, expected calibration error (15 equal-
width bins), and accuracy under four corruptions generated on the fly from the
clean test images (no CIFAR-10-C download; compute nodes have no network):
Gaussian noise at sigma 0.04 and 0.08 in [0,1] pixel units, contrast halved
about 0.5, brightness +0.1.  Written to results/robust/<name>.json.

Usage:  python -m experiments.robust_eval [--tags pop] [--shard K/N]
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from .common import CKPT_DIR, CONFIGS, RESULTS_DIR, checkpoint_epochs, dump_json
from .data import torch_datasets_for
from .models import build_model

OUT = os.path.join(RESULTS_DIR, "robust")
MEAN = torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1)
STD = torch.tensor((0.247, 0.243, 0.261)).view(1, 3, 1, 1)


def corrupt(x01, kind, gen):
    if kind == "noise04":
        return (x01 + 0.04 * torch.randn(x01.shape, generator=gen)).clamp(0, 1)
    if kind == "noise08":
        return (x01 + 0.08 * torch.randn(x01.shape, generator=gen)).clamp(0, 1)
    if kind == "contrast":
        return ((x01 - 0.5) * 0.5 + 0.5).clamp(0, 1)
    if kind == "brightness":
        return (x01 + 0.1).clamp(0, 1)
    raise ValueError(kind)


def ece(probs, labels, n_bins=15):
    conf, pred = probs.max(1)
    acc = (pred == labels).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    e = torch.zeros(())
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.float().mean() * (acc[m].mean() - conf[m].mean()).abs()
    return float(e)


def evaluate(cfg, device):
    epoch = checkpoint_epochs(cfg)[-1]
    _, _, te, num_classes = torch_datasets_for(cfg)
    loader = torch.utils.data.DataLoader(te, batch_size=512, num_workers=2)
    model = build_model(cfg, num_classes).to(device)
    ck = torch.load(os.path.join(CKPT_DIR, cfg.name, f"epoch{epoch}.pt"), map_location=device)
    model.load_state_dict(ck["model"]); model.eval()
    kinds = ["clean", "noise04", "noise08", "contrast", "brightness"]
    logits = {k: [] for k in kinds}; labels = []
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for x, y in loader:
            x01 = (x * STD + MEAN)               # back to [0,1]
            labels.append(y)
            for k in kinds:
                xi = x01 if k == "clean" else corrupt(x01, k, gen)
                logits[k].append(model(((xi - MEAN) / STD).to(device)).cpu())
    y = torch.cat(labels)
    out = {"name": cfg.name, "epoch": epoch}
    for k in kinds:
        lg = torch.cat(logits[k]); p = lg.softmax(1)
        out[f"acc_{k}"] = float((p.argmax(1) == y).float().mean())
        if k == "clean":
            out["nll"] = float(F.cross_entropy(lg, y)); out["ece"] = ece(p, y)
    out["acc_corrupt_mean"] = float(np.mean([out[f"acc_{k}"] for k in kinds[1:]]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["pop"])
    ap.add_argument("--shard", default=None)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfgs = [c for c in CONFIGS if c.tag in a.tags]
    if a.shard:
        k, n = map(int, a.shard.split("/")); cfgs = cfgs[k::n]
    os.makedirs(OUT, exist_ok=True)
    for cfg in cfgs:
        p = os.path.join(OUT, f"{cfg.name}.json")
        if os.path.exists(p):
            continue
        r = evaluate(cfg, device); dump_json(r, p)
        print(f"{cfg.name}: acc {r['acc_clean']:.4f} ece {r['ece']:.4f} nll {r['nll']:.3f} corrupt {r['acc_corrupt_mean']:.4f}", flush=True)


if __name__ == "__main__":
    main()
