"""Direction A: class interaction in foundation-model representations.

The pairwise-sufficiency law is currently a claim about small CIFAR
classifiers. This module takes it to representations people actually use:
frozen features from pretrained image encoders, measured with the same
certified pipeline as the rest of the campaign.

No training is involved -- one forward pass per model -- so this runs
comfortably on a workstation and does not need the cluster.

ORDER OF OPERATIONS MATTERS. Proposition (projection is conservative) says a
separation certificate measured on a d0-dimensional PCA shadow is valid in the
ambient representation, but its quantitative bite depends on the SPECTRAL
DECAY of the features, not on their width: for close-to-isotropic features the
certified scale is a few percent of the ambient one. So for every model we
report, before any measurement:

    spectral_profile      -- retained variance at d0, effective rank
    certified_rstar_bound -- the REALIZED tightness r*(PX,PY)/r*(X,Y),
                             computable exactly because ambient r* is O(n^2 D)

Only then is the measured profile worth reading. `--report` prints both.

Usage
-----
    python -m experiments.foundation --list
    python -m experiments.foundation --prefetch          # login node / online
    python -m experiments.foundation <index> [--device cuda|mps|cpu]
    python -m experiments.foundation --report

Weights come from torchvision and torch.hub; anything unavailable is skipped
with a message rather than failing the run. Compute nodes on Pegasus have no
usable internet, so run --prefetch on the login node first if using the
cluster; locally it is unnecessary.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict

import numpy as np

from .common import DATA_DIR, RESULTS_DIR, ROOT, dump_json
from .hd import (certified_rstar_bound, pca_projection, random_projections,
                 spectral_profile)

FEAT_DIR = os.path.join(ROOT, "features", "foundation")


@dataclass
class FoundationModel:
    name: str
    source: str          # 'torchvision' | 'hub' | 'hf-text'
    ident: str           # constructor name or hub spec
    input_size: int = 224
    batch: int = 64      # ViT-B/16 attention buffers OOM at large batch
    note: str = ""
    modality: str = "image"


MODELS = [
    FoundationModel("vit_b16_in1k", "torchvision", "vit_b_16", 224, 64,
                    "supervised ImageNet-1k ViT"),
    FoundationModel("resnet50_in1k", "torchvision", "resnet50", 224, 128,
                    "supervised conv baseline"),
    FoundationModel("dinov2_vits14", "hub", "facebookresearch/dinov2:dinov2_vits14",
                    224, 64, "self-supervised, no labels"),
    FoundationModel("clip_vitb32", "hub", "openai/clip-vit-base-patch32", 224, 128,
                    "language-supervised (needs transformers)"),
    # Depth coverage: the obstruction is claimed to be graded by depth, so the
    # sweep wants backbones deep enough for the gradient to be visible.
    FoundationModel("vit_l16_in1k", "torchvision", "vit_l_16", 224, 32,
                    "24-block ViT: the longest depth ladder here"),
    FoundationModel("convnext_tiny", "torchvision", "convnext_tiny", 224, 96,
                    "modern conv, different normalization regime"),
    # A SECOND MODALITY is what turns this from a vision quirk into a claim
    # about learned representations.  Text has no spatial pooling and a wholly
    # different tokenization, so if the depth gradient survives here the
    # mechanism is concentration rather than anything architectural.
    FoundationModel("gpt2", "hf-text", "gpt2", batch=32,
                    note="autoregressive LM hidden states", modality="text"),
    FoundationModel("bert_base", "hf-text", "bert-base-uncased", batch=32,
                    note="masked LM, bidirectional", modality="text"),
]


def build(model: FoundationModel):
    """Return (nn.Module, feature_fn) or (None, None) if unavailable."""
    import torch
    import torch.nn as nn

    if model.source == "torchvision":
        import torchvision
        weights = "DEFAULT"
        net = getattr(torchvision.models, model.ident)(weights=weights).eval()

        if model.ident.startswith("vit"):
            def feats(x):
                v = net
                z = v._process_input(x)
                cls = v.class_token.expand(z.shape[0], -1, -1)
                z = torch.cat([cls, z], dim=1) + v.encoder.pos_embedding
                z = v.encoder.dropout(z)
                out = {}
                n = len(v.encoder.layers)
                taps = {max(1, n // 4), max(1, n // 2), max(1, 3 * n // 4), n}
                for i, layer in enumerate(v.encoder.layers, start=1):
                    z = layer(z)
                    if i in taps:
                        out[f"block{i}"] = z[:, 0]              # CLS
                        # POOLING CONTROL: the text models are masked-mean
                        # pooled, so a CLS-vs-mean difference could masquerade
                        # as a modality difference.  Emit both from the same
                        # forward pass; if the depth gradient survives under
                        # mean pooling, pooling is not the explanation.
                        out[f"block{i}_mean"] = z[:, 1:].mean(dim=1)
                zl = v.encoder.ln(z)
                out["penult"] = zl[:, 0]
                out["penult_mean"] = zl[:, 1:].mean(dim=1)
                return out
        elif model.ident.startswith("convnext"):
            # ConvNeXt exposes neither conv1/bn1/maxpool nor .layer1..4, so it
            # cannot ride the ResNet branch.  features is a flat Sequential of
            # stem, stage, downsample, stage, ... -- the four stages sit at the
            # odd indices, and each tap is spatially mean-pooled to match the
            # ResNet convention.
            # Identify the stages structurally -- they are the nn.Sequential
            # children, the stem and downsamples are not -- rather than by
            # hardcoded indices, which would break on another ConvNeXt size.
            stages = [i for i, b in enumerate(net.features)
                      if isinstance(b, nn.Sequential)]

            def feats(x):
                out, z = {}, x
                for i, block in enumerate(net.features):
                    z = block(z)
                    if i in stages:
                        out[f"stage{stages.index(i)+1}"] = z.mean(dim=(2, 3))
                out["penult"] = out[f"stage{len(stages)}"]
                return out
        else:                                   # resnet
            def feats(x):
                out = {}
                z = net.maxpool(net.relu(net.bn1(net.conv1(x))))
                for i, stage in enumerate([net.layer1, net.layer2,
                                           net.layer3, net.layer4], start=1):
                    z = stage(z)
                    out[f"stage{i}"] = z.mean(dim=(2, 3))
                out["penult"] = out["stage4"]
                return out
        return net, feats

    if model.name.startswith("dinov2"):
        repo, entry = model.ident.split(":")
        net = torch.hub.load(repo, entry).eval()

        def feats(x):
            # DINOv2 patches at 14px, so 224 is native but any other input size
            # needs the positional grid resampled -- without this it asserts.
            try:
                z = net.forward_features(x)
            except (AssertionError, RuntimeError):
                z = net.forward_features(x, interpolate_pos_encoding=True)
            return {"penult": z["x_norm_clstoken"]}
        return net, feats

    if model.source == "hf-text":
        try:
            from transformers import AutoModel
        except ImportError:
            print(f"  [{model.name}] needs `pip install transformers`; skipping")
            return None, None
        net = AutoModel.from_pretrained(model.ident).eval()

        def feats(batch):
            o = net(**batch, output_hidden_states=True)
            hs = o.hidden_states                     # tuple: embeddings + blocks
            mask = batch["attention_mask"].unsqueeze(-1).float()
            # Mean-pool over real tokens only.  A CLS vector would not be
            # comparable across an autoregressive and a masked model, and for
            # GPT-2 there is no CLS at all; masked mean is the one pooling both
            # admit without privileging either.
            def pool(h):
                return (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            n = len(hs) - 1
            taps = sorted({max(1, n // 4), max(1, n // 2), max(1, 3 * n // 4)})
            out = {f"block{i}": pool(hs[i]) for i in taps}
            out["penult"] = pool(hs[-1])
            return out
        return net, feats

    if model.name.startswith("clip"):
        try:
            from transformers import CLIPVisionModel
        except ImportError:
            print(f"  [{model.name}] needs `pip install transformers`; skipping")
            return None, None
        net = CLIPVisionModel.from_pretrained(model.ident).eval()

        def feats(x):
            o = net(pixel_values=x, output_hidden_states=True)
            hs = o.hidden_states
            out = {f"block{i}": hs[i][:, 0] for i in (3, 6, 9)}
            out["penult"] = o.last_hidden_state[:, 0]
            return out
        return net, feats

    return None, None


IMAGE_DATASET = os.environ.get("DISENTANGLE_IMAGE_DATASET", "cifar10").lower()
# Second text corpus for E11: DISENTANGLE_TEXT_DATASET=dbpedia (10 of the 14
# DBpedia ontology classes, Wikipedia abstracts; needs `datasets`).  Written
# to <model>__dbpedia.npz so it cannot overwrite the 20 Newsgroups features.
TEXT_DATASET = os.environ.get("DISENTANGLE_TEXT_DATASET", "20ng").lower()


def dataloader(input_size, batch, limit=None):
    """Ten-class image test split, resized to the encoder's input.

    CIFAR-10 keeps the class structure comparable with the rest of the
    campaign: ten classes, the same pairs, the same protocol constants.

    DATASET CONFOUND CONTROL.  The depth gradient is measured on images and
    its absence on text, so the two are confounded with the dataset.  Setting
    DISENTANGLE_IMAGE_DATASET=stl10 re-runs the vision side on a different
    ten-class corpus at much higher native resolution: if the gradient is
    unchanged, it is a property of the representations rather than of
    CIFAR-10, which is the objection that would otherwise sink the claim.
    """
    import torch
    import torchvision as tv
    import torchvision.transforms as T
    tf = T.Compose([
        T.Resize(input_size), T.CenterCrop(input_size), T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    if IMAGE_DATASET == "stl10":
        ds = tv.datasets.STL10(DATA_DIR, split="test", download=True,
                               transform=tf)
    else:
        ds = tv.datasets.CIFAR10(DATA_DIR, train=False, download=True,
                                 transform=tf)
    if limit:
        ds = torch.utils.data.Subset(ds, range(min(limit, len(ds))))
    # num_workers=0: forking after OpenMP has initialized deadlocks on macOS,
    # and the loader is not the bottleneck here (one forward pass).
    return torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=0)


def text_batches(model: FoundationModel, limit=None, max_len=128, n_classes=10,
                 min_words=20):
    """20 Newsgroups, tokenized: the text analogue of the CIFAR-10 protocol.

    Ten classes, so the pair structure and every protocol constant carry over
    from the vision side unchanged and the two are directly comparable.
    """
    import torch
    from sklearn.datasets import fetch_20newsgroups
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model.ident)
    if tok.pad_token is None:                     # GPT-2 ships without one
        tok.pad_token = tok.eos_token
    if TEXT_DATASET == "dbpedia":
        yield from _dbpedia_batches(model, tok, limit, max_len, n_classes, min_words)
        return
    cats = fetch_20newsgroups(subset="test", remove=("headers", "footers",
                                                     "quotes"))
    keep = sorted(set(cats.target.tolist()))[:n_classes]
    # Stripping headers, footers and quotes empties a substantial number of
    # posts outright.  An empty document has no unmasked tokens, so the masked
    # mean divides by the clamp and returns the ZERO VECTOR -- many identical
    # features, a minimum cross distance of exactly 0, and a scale window of
    # nan or 1e7.  Require real content rather than discovering this in the
    # diagnostic.
    idx = [i for i, t in enumerate(cats.target)
           if t in keep and len(cats.data[i].split()) >= min_words]
    if limit:
        idx = idx[:limit]
    texts = [cats.data[i] for i in idx]
    ys = np.array([keep.index(cats.target[i]) for i in idx])
    for s in range(0, len(texts), model.batch):
        chunk = texts[s:s + model.batch]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len)
        yield enc, torch.from_numpy(ys[s:s + model.batch])


def pick_device(requested=None):
    import torch
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dbpedia_batches(model, tok, limit, max_len, n_classes, min_words, per_class=400):
    """DBpedia-14 test split, first n_classes ontology classes, up to
    per_class abstracts each with at least min_words words -- sized like the
    20 Newsgroups sample so every protocol constant carries over."""
    import torch
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("MISSING: pip install datasets (needed for DISENTANGLE_TEXT_DATASET=dbpedia)")
    ds = load_dataset("fancyzhx/dbpedia_14", split="test")
    texts, ys, count = [], [], {}
    for row in ds:
        c = int(row["label"])
        if c >= n_classes or count.get(c, 0) >= per_class:
            continue
        if len(row["content"].split()) < min_words:
            continue
        texts.append(row["content"]); ys.append(c); count[c] = count.get(c, 0) + 1
        if all(count.get(k, 0) >= per_class for k in range(n_classes)):
            break
    if limit:
        texts, ys = texts[:limit], ys[:limit]
    ys = np.array(ys)
    print(f"  [{model.name}] dbpedia: {len(texts)} docs, per class {np.bincount(ys).tolist()}", flush=True)
    for s in range(0, len(texts), model.batch):
        enc = tok(texts[s:s + model.batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len)
        yield enc, torch.from_numpy(ys[s:s + model.batch])


def extract(model: FoundationModel, device=None, limit=None):
    import torch
    device = pick_device(device)
    net, feats = build(model)
    if net is None:
        return None
    net = net.to(device)
    if model.modality == "text":
        loader = text_batches(model, limit)
    else:
        loader = dataloader(model.input_size, model.batch, limit)

    acc, labels = {}, []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if model.modality == "text":
                f = feats({k: v.to(device) for k, v in x.items()})
            else:
                f = feats(x.to(device))
            for k, v in f.items():
                acc.setdefault(k, []).append(v.float().cpu().numpy())
            labels.append(y.numpy())
            if i % 20 == 0:
                print(f"  [{model.name}] batch {i}", flush=True)
    out = {k: np.concatenate(v) for k, v in acc.items()}
    out["labels"] = np.concatenate(labels)
    # Catch degenerate features HERE rather than two steps downstream in the
    # scale-window diagnostic, where duplicates surface as nan or 1e7 and look
    # like concentration.  Any duplicate row means some inputs collapsed to the
    # same vector -- empty documents, all-pad batches, a dead layer tap.
    for k, v in out.items():
        if k == "labels" or v.ndim != 2:
            continue
        uniq = len(np.unique(np.round(v.astype(np.float64), 6), axis=0))
        if uniq < len(v):
            print(f"  [{model.name}] WARNING {k}: only {uniq}/{len(v)} distinct "
                  f"feature vectors -- duplicates will break the scale window")
    os.makedirs(FEAT_DIR, exist_ok=True)
    # Tag non-default corpora so a control run cannot silently overwrite the
    # primary features and leave two incomparable regimes under one name.
    if model.modality == "text":
        suffix = "" if TEXT_DATASET == "20ng" else f"__{TEXT_DATASET}"
    else:
        suffix = "" if IMAGE_DATASET == "cifar10" else f"__{IMAGE_DATASET}"
    path = os.path.join(FEAT_DIR, f"{model.name}{suffix}.npz")
    np.savez_compressed(path, **out)
    print(f"  [{model.name}] wrote {path}  layers={[k for k in out if k!='labels']}")
    return path


# --------------------------------------------------------------------------- #
def regime_report(d0=5, n_pairs=6, seed=0):
    """The gating check: is each representation in the regime where a d0
    projection retains the ambient separation scale?"""
    from itertools import combinations
    rows = []
    for m in MODELS:
        path = os.path.join(FEAT_DIR, f"{m.name}.npz")
        if not os.path.exists(path):
            continue
        z = np.load(path)
        labels = z["labels"]
        classes = sorted(np.unique(labels).tolist())
        rng = np.random.default_rng(seed)
        pairs = [tuple(rng.choice(classes, 2, replace=False))
                 for _ in range(n_pairs)]
        for layer in [k for k in z.files if k != "labels"]:
            F = z[layer]
            sp = spectral_profile(F, d0=d0)
            tight = []
            for c1, c2 in pairs:
                X = F[labels == c1][:200].astype(np.float64)
                Y = F[labels == c2][:200].astype(np.float64)
                P = pca_projection(np.vstack([X, Y]), d0)
                b = certified_rstar_bound(X, Y, [P])
                if np.isfinite(b["tightness"]):
                    tight.append(b["tightness"])
            rows.append({"model": m.name, "layer": layer,
                         "ambient_dim": sp["ambient_dim"],
                         "retained_at_d0": sp["retained_at_d0"],
                         "eff_rank": sp["eff_rank"],
                         "tightness_median": float(np.median(tight)) if tight else None})
            r = rows[-1]
            print(f"{r['model']:>16} {r['layer']:>8}  D={r['ambient_dim']:>5}  "
                  f"retained@{d0}={r['retained_at_d0']:.3f}  "
                  f"eff_rank={r['eff_rank']:.1f}  "
                  f"tightness={r['tightness_median'] if r['tightness_median'] is None else round(r['tightness_median'],3)}")
    if rows:
        dump_json(rows, os.path.join(RESULTS_DIR, "foundation_regime.json"))
        print("\nRead this before trusting any measured profile: low retained "
              "variance and low tightness mean the certificate, while valid, is "
              "weak for that representation.")
    else:
        print("no extracted features yet; run the extraction first")
    return rows


def prefetch_dbpedia():
    """Login node only: download the DBpedia-14 corpus into the HF cache."""
    from datasets import load_dataset
    ds = load_dataset("fancyzhx/dbpedia_14", split="test")
    print(f"dbpedia_14 test split cached: {len(ds)} rows")


def prefetch():
    """Download weights and data while the internet is available."""
    for m in MODELS:
        print(f"prefetching {m.name} ...")
        try:
            build(m)
        except Exception as e:                      # noqa: BLE001
            print(f"  skipped: {type(e).__name__}: {e}")
    dataloader(224, 8, limit=8)
    for m in MODELS:                                # tokenizers + 20NG corpus
        if m.modality == "text":
            try:
                next(text_batches(m, limit=8))
            except Exception as e:                  # noqa: BLE001
                print(f"  [{m.name}] text prefetch skipped: "
                      f"{type(e).__name__}: {e}")
    print("prefetch done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("index", nargs="?")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--prefetch", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N test images (quick pass)")
    a = ap.parse_args()

    if a.list:
        for i, m in enumerate(MODELS):
            print(f"{i:3d}  {m.name:<16} {m.source:<12} {m.note}")
        sys.exit(0)
    if a.prefetch:
        prefetch()
        if TEXT_DATASET == "dbpedia":
            prefetch_dbpedia()
        sys.exit(0)
    if a.report:
        regime_report(); sys.exit(0)
    if a.index is None:
        for m in MODELS:
            extract(m, a.device, a.limit)
    else:
        extract(MODELS[int(a.index)], a.device, a.limit)
