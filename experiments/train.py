"""Train one model from CONFIGS (cluster array task) with checkpointing.

Usage:  python -m experiments.train <config_index>
Writes checkpoints/<name>/epoch<k>.pt and results/train/<name>.json.
Supports: supervised CE (cosine/step/const LR), SimCLR (NT-Xent) on the
CIFAR ResNet backbone, and pretrained ViT-B/16 finetuning at 224px.
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from .common import (CONFIGS, CKPT_DIR, RESULTS_DIR, checkpoint_epochs,
                     dump_json, ensure_dirs)
from .data import torch_datasets_for
from .models import build_model


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    return correct / total


def probe_eval(model, loader_ev, loader_te, device, n_fit=5000):
    """Linear-probe accuracy on penult features (SimCLR evaluation)."""
    from sklearn.linear_model import LogisticRegression

    def feats(loader, cap):
        model.eval()
        X, Y = [], []
        with torch.no_grad():
            for x, y in loader:
                _, f = model(x.to(device), return_features=True)
                X.append(f["penult"].cpu().numpy())
                Y.append(y.numpy())
                if sum(len(v) for v in Y) >= cap:
                    break
        return np.concatenate(X)[:cap], np.concatenate(Y)[:cap]

    Xtr, ytr = feats(loader_ev, n_fit)
    Xte, yte = feats(loader_te, 10000)
    clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    return float(clf.score(Xte, yte))


def disentangle_reg(feats, labels, scales=(0.5, 1.0, 2.0), eps=1e-8,
                    detach_scale=False):
    """E6 differentiable disentanglement surrogate on a feature batch.

    A soft, scale-averaged relaxation of the ECP interaction: the mean over
    cross-class point pairs of a multi-scale Gaussian kernel of their feature
    distance --- a smooth stand-in for the mass of the class ball-unions'
    overlap.  Minimizing it pushes different-class clouds apart.

    The bandwidth is the per-batch median *within-class* squared distance ---
    the clouds' own scale, mirroring the interaction quotient's scale-relative
    normalization.  Because that normalizer appears in the denominator,
    K = exp(-D2_cross / (2 s sigma2)) is lowered by making cross-class
    distances large OR the within-class radius small, which is exactly
    "separate the classes relative to their own radius".

    `detach_scale` MUST stay False, and the flag exists only to reproduce the
    original E6 run.  Detaching sigma2 leaves the loss VALUE scale-invariant
    while destroying the scale-invariance of the GRADIENT: backprop then sees
    a fixed bandwidth, so uniformly inflating the features drives the loss to
    zero without separating anything.  The radial component of the gradient is
    negative under the detached form, i.e. descent inflates on purpose, and
    the first E6 campaign duly raised the certified quotient monotonically in
    dis_lambda (0.159 -> 0.344) while costing 5.5 points of accuracy.  Keeping
    sigma2 in the graph makes the radial gradient identically zero, so norm
    inflation buys nothing and only genuine relative separation lowers the
    penalty.
    """
    D2 = torch.cdist(feats, feats).pow(2)               # (B, B)
    B = feats.size(0)
    off = ~torch.eye(B, dtype=torch.bool, device=feats.device)
    same = (labels[:, None] == labels[None, :]) & off
    cross = (labels[:, None] != labels[None, :]) & off
    if cross.sum() == 0 or same.sum() == 0:
        return feats.new_zeros(())
    sigma2 = D2[same].median()                          # clouds' own scale
    if detach_scale:
        sigma2 = sigma2.detach()
    K = sum(torch.exp(-D2 / (2.0 * s * (sigma2 + eps))) for s in scales)
    return (K / len(scales))[cross].mean()


@torch.no_grad()
def eval_corrupted(model, loader, device, sigmas=(0.05, 0.1, 0.2)):
    """Robustness proxy: mean test accuracy under additive Gaussian input
    noise at several severities (self-contained; no CIFAR-10-C download)."""
    model.eval()
    accs = []
    for s in sigmas:
        correct = total = 0
        for x, y in loader:
            x = (x + s * torch.randn_like(x)).to(device)
            correct += (model(x).argmax(1) == y.to(device)).sum().item()
            total += y.numel()
        accs.append(correct / total)
    return float(np.mean(accs)), {f"sigma{s}": a for s, a in zip(sigmas, accs)}


def nt_xent(z1, z2, tau=0.5):
    """NT-Xent loss for a batch of paired views."""
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    n = z1.size(0)
    sim = z @ z.t() / tau
    sim.fill_diagonal_(float("-inf"))
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)


def make_scheduler(opt, cfg):
    if cfg.lr_schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    if cfg.lr_schedule == "step":
        return torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[cfg.epochs // 2, 3 * cfg.epochs // 4], gamma=0.1)
    if cfg.lr_schedule == "const":
        return torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0,
                                                   total_iters=0)
    raise ValueError(cfg.lr_schedule)


def train_one(cfg):
    ensure_dirs()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))

    tr, ev, te, num_classes = torch_datasets_for(cfg)
    loader_tr = torch.utils.data.DataLoader(
        tr, batch_size=cfg.batch_size, shuffle=True, num_workers=2,
        drop_last=True)
    loader_ev = torch.utils.data.DataLoader(ev, batch_size=512, num_workers=2)
    loader_te = torch.utils.data.DataLoader(te, batch_size=512, num_workers=2)

    model = build_model(cfg, num_classes).to(device)
    if cfg.arch in ("vit", "vit_b16"):
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                weight_decay=cfg.weight_decay)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9,
                              weight_decay=cfg.weight_decay)
    sched = make_scheduler(opt, cfg)

    ckpts = set(checkpoint_epochs(cfg))
    ckpt_dir = os.path.join(CKPT_DIR, cfg.name)
    os.makedirs(ckpt_dir, exist_ok=True)

    def save(epoch):
        torch.save({"model": model.state_dict(), "epoch": epoch},
                   os.path.join(ckpt_dir, f"epoch{epoch}.pt"))

    if 0 in ckpts:
        save(0)
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for batch in loader_tr:
            opt.zero_grad(set_to_none=True)
            if cfg.objective == "simclr":
                (v1, v2), _ = batch
                loss = nt_xent(model(v1.to(device)), model(v2.to(device)))
            elif cfg.dis_lambda:
                x, y = batch[0].to(device), batch[1].to(device)
                logits, feats = model(x, return_features=True)
                loss = (F.cross_entropy(logits, y)
                        + cfg.dis_lambda * disentangle_reg(feats["penult"], y))
            else:
                x, y = batch
                loss = F.cross_entropy(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
        sched.step()
        if epoch in ckpts:
            save(epoch)
            print(f"[{cfg.name}] epoch {epoch}  "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    if cfg.objective == "simclr":
        acc_te = probe_eval(model, loader_ev, loader_te, device)
        acc_tr = float("nan")
        gap = float("nan")
    else:
        acc_te = evaluate(model, loader_te, device)
        acc_tr = evaluate(model, loader_ev, device)
        gap = acc_tr - acc_te
    out = {"name": cfg.name, "config": cfg.__dict__,
           "test_acc": acc_te, "train_acc": acc_tr, "gen_gap": gap,
           "minutes": (time.time() - t0) / 60, "device": device}
    if cfg.tag == "e6":  # E6 needs the robustness proxy alongside clean acc
        rob_mean, rob_by = eval_corrupted(model, loader_te, device)
        out["robust_acc"] = rob_mean
        out["robust_by_sigma"] = rob_by
        print(f"[{cfg.name}] robust(mean) {rob_mean:.4f}")
    dump_json(out, os.path.join(RESULTS_DIR, "train", f"{cfg.name}.json"))
    print(f"[{cfg.name}] done: test {acc_te:.4f}")


if __name__ == "__main__":
    train_one(CONFIGS[int(sys.argv[1])])
