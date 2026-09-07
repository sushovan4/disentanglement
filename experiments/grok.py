"""Direction B: grokking runs, for certified reorganization analysis.

Grokking (Power et al., 2022): a small transformer on modular arithmetic
memorizes the training set almost immediately, then -- tens of thousands of
steps later, long after the training loss has flattened -- abruptly
generalizes.  The phenomenon is a clean testbed for the question our paired
test is built to answer: *when* does the representation reorganize, and does
that reorganization coincide with, precede, or follow the generalization
jump?  Existing accounts read this off loss curves and eyeballed metrics; we
can attach a p-value to each step.

This module only produces the runs and their checkpoints.  The measurement
and the certified comparison live in `e9_grok.py`.

Task:   (a + b) mod p  for a, b in [0, p), a fixed random train fraction.
Model:  1-2 layer decoder-only transformer, features tapped per block.
Recipe: AdamW, high weight decay -- the standard grokking recipe; weight
        decay is the ingredient the phenomenon is known to depend on.

Checkpoints are log-spaced early and linear later, so the pre-grok plateau
and the transition both get resolution.

Usage:
  python -m experiments.grok --list
  python -m experiments.grok <config_index>
"""

import json
import os
import sys
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import CKPT_DIR, RESULTS_DIR, dump_json, ensure_dirs


# --------------------------------------------------------------------------- #
@dataclass
class GrokConfig:
    p: int = 97                  # modulus (prime)
    op: str = "add"              # 'add' | 'sub' | 'mul'
    train_frac: float = 0.4
    depth: int = 2
    d_model: int = 128
    heads: int = 4
    lr: float = 1e-3
    weight_decay: float = 1.0    # high wd: the standard grokking recipe
    steps: int = 30000
    batch_size: int = 512        # <=0 means full batch
    seed: int = 0
    dense_window: tuple = None   # e.g. (1000, 2000): checkpoint densely here
    dense_every: int = 100

    @property
    def name(self):
        base = (f"grok_{self.op}{self.p}_L{self.depth}d{self.d_model}"
                f"_wd{self.weight_decay:g}_tf{self.train_frac:g}_s{self.seed}")
        if self.dense_window is not None:   # distinct name: never clobbers the
            base += f"_dense{self.dense_window[0]}"   # coarse run's outputs
        return base


def build_configs():
    """Seeds for the main claim, plus a weight-decay arm.

    wd=0 is the control: grokking is known to depend on weight decay, so if
    the certified reorganization tracks generalization it should move (or
    vanish) with the arm that does not grok.
    """
    cfgs = []
    for seed in range(3):
        cfgs.append(GrokConfig(seed=seed))
    for seed in range(2):
        cfgs.append(GrokConfig(weight_decay=0.0, steps=30000, seed=seed))
    # Dense re-runs across the generalization jump (indices 5-8). Training is
    # seeded, so these reproduce the coarse runs' trajectories exactly and only
    # add checkpoints; 5000 steps is ample since the jump is at ~2000.
    for seed in range(3):
        cfgs.append(GrokConfig(seed=seed, steps=5000,
                               dense_window=(1000, 2000), dense_every=100))
    cfgs.append(GrokConfig(weight_decay=0.0, steps=5000, seed=0,
                           dense_window=(1000, 2000), dense_every=100))
    # Indices 9-12: the window above was placed from the coarse grok_step
    # (first checkpoint with val >= 0.5), but the dense run showed val is flat
    # at ~0.40 until step 1900 and only reaches 1.0 by 3000 -- so [1000,2000]
    # covers the plateau BEFORE the transition, not the transition. This window
    # brackets the actual jump with margin.
    for seed in range(3):
        cfgs.append(GrokConfig(seed=seed, steps=5000,
                               dense_window=(1800, 3200), dense_every=100))
    cfgs.append(GrokConfig(weight_decay=0.0, steps=5000, seed=0,
                           dense_window=(1800, 3200), dense_every=100))
    return cfgs


CONFIGS = build_configs()


def checkpoint_steps(cfg):
    """Log-spaced early, linear later, optionally dense across a window.

    The default grid jumps 1000 -> 2000, which is where these runs generalize,
    so an effect located there is resolved only to within a factor of two.
    `dense_window=(a, b)` adds `dense_every` steps inside [a, b] so that lead
    vs lag against the validation jump can actually be read off.
    """
    early = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000]
    late = list(range(2000, cfg.steps + 1, 1000))
    dense = []
    if cfg.dense_window is not None:
        a, b = cfg.dense_window
        dense = list(range(a, b + 1, cfg.dense_every))
    return sorted(set(s for s in early + late + dense if s <= cfg.steps))


# --------------------------------------------------------------------------- #
class GrokTransformer(nn.Module):
    """Decoder-only transformer over the 3-token sequence [a, b, =].

    return_features exposes the residual stream at each block boundary,
    read at the final ('=') position -- the position the readout uses.
    """

    def __init__(self, p, depth=2, d_model=128, heads=4):
        super().__init__()
        self.p = p
        self.embed = nn.Embedding(p + 1, d_model)   # p symbols + '='
        self.pos = nn.Parameter(torch.zeros(1, 3, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=heads, dim_feedforward=4 * d_model,
                dropout=0.0, activation="gelu", batch_first=True,
                norm_first=True)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, p)
        self.depth = depth

    def forward(self, x, return_features=False):
        # x: (B, 2) integer pairs (a, b)
        eq = torch.full((x.size(0), 1), self.p, dtype=torch.long,
                        device=x.device)
        tok = torch.cat([x, eq], dim=1)
        h = self.embed(tok) + self.pos
        feats = {}
        causal = torch.triu(torch.ones(3, 3, device=x.device, dtype=torch.bool),
                            diagonal=1)
        for i, blk in enumerate(self.blocks, start=1):
            h = blk(h, src_mask=causal)
            feats[f"block{i}"] = h[:, -1]           # '=' position
        h = self.norm(h)
        feats["penult"] = h[:, -1]
        logits = self.out(h[:, -1])
        if return_features:
            return logits, feats
        return logits

    @property
    def feature_points(self):
        return tuple(f"block{i}" for i in range(1, self.depth + 1)) + ("penult",)


# --------------------------------------------------------------------------- #
def make_data(cfg):
    p = cfg.p
    a, b = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    a, b = a.ravel(), b.ravel()
    if cfg.op == "add":
        y = (a + b) % p
    elif cfg.op == "sub":
        y = (a - b) % p
    elif cfg.op == "mul":
        y = (a * b) % p
    else:
        raise ValueError(cfg.op)
    X = np.stack([a, b], axis=1)
    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(len(X))
    n_tr = int(round(cfg.train_frac * len(X)))
    return (torch.tensor(X[idx[:n_tr]]), torch.tensor(y[idx[:n_tr]]),
            torch.tensor(X[idx[n_tr:]]), torch.tensor(y[idx[n_tr:]]))


@torch.no_grad()
def accuracy(model, X, y, device, chunk=4096):
    model.eval()
    correct = 0
    for i in range(0, len(X), chunk):
        xb = X[i:i + chunk].to(device)
        correct += (model(xb).argmax(1).cpu() == y[i:i + chunk]).sum().item()
    return correct / len(X)


def train_one(cfg):
    ensure_dirs()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"          # Apple Silicon; these models are small enough
    else:
        device = "cpu"
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))

    Xtr, ytr, Xte, yte = make_data(cfg)
    model = GrokTransformer(cfg.p, cfg.depth, cfg.d_model, cfg.heads).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay, betas=(0.9, 0.98))

    ckpt_dir = os.path.join(CKPT_DIR, cfg.name)
    os.makedirs(ckpt_dir, exist_ok=True)
    want = set(checkpoint_steps(cfg))

    def save(step):
        torch.save({"model": model.state_dict(), "step": step,
                    "config": asdict(cfg)},
                   os.path.join(ckpt_dir, f"step{step}.pt"))

    curve = []
    bs = cfg.batch_size if cfg.batch_size > 0 else len(Xtr)
    g = torch.Generator().manual_seed(cfg.seed)
    t0 = time.time()
    for step in range(cfg.steps + 1):
        if step in want:
            tr = accuracy(model, Xtr, ytr, device)
            te = accuracy(model, Xte, yte, device)
            curve.append({"step": step, "train_acc": tr, "val_acc": te})
            save(step)
            print(f"[{cfg.name}] step {step:>6}  train {tr:.4f}  val {te:.4f}"
                  f"  ({(time.time()-t0)/60:.1f} min)", flush=True)
        if step == cfg.steps:
            break
        model.train()
        sel = torch.randint(0, len(Xtr), (min(bs, len(Xtr)),), generator=g)
        xb, yb = Xtr[sel].to(device), ytr[sel].to(device)
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(model(xb), yb).backward()
        opt.step()

    out = {"name": cfg.name, "config": asdict(cfg), "curve": curve,
           "minutes": (time.time() - t0) / 60, "device": device,
           "grok_step": grok_step(curve)}
    dump_json(out, os.path.join(RESULTS_DIR, "grok", f"{cfg.name}.json"))
    print(f"[{cfg.name}] done; grok step = {out['grok_step']}")
    return out


def grok_step(curve, thresh=0.5):
    """First checkpoint at which val accuracy crosses `thresh`.

    This is the event Direction B asks the topology about; None if the run
    never generalizes (expected for the wd=0 control).
    """
    for r in curve:
        if r["val_acc"] >= thresh:
            return r["step"]
    return None


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        for i, c in enumerate(CONFIGS):
            print(f"{i:3d}  {c.name}")
        print(f"total: {len(CONFIGS)}")
        sys.exit(0)
    train_one(CONFIGS[int(sys.argv[1])])
