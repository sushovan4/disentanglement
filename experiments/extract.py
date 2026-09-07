"""Feature extraction: class clouds at every stage boundary of a checkpoint.

Features for the full test set are cached to features/<name>/epoch<k>.npz so
the ECP side (torch-free) can be re-run without touching the model.
"""

import os

import numpy as np

from .common import CONFIGS, CKPT_DIR, ROOT, checkpoint_epochs

FEAT_DIR = os.path.join(ROOT, "features")


def feature_path(cfg, epoch):
    return os.path.join(FEAT_DIR, cfg.name, f"epoch{epoch}.npz")


def extract_checkpoint(cfg, epoch, device=None):
    """Run the evaluation split through checkpoint `epoch`; cache features.

    The split is the test set, except cifar100s whose test split has only
    100 images/class -- too few for the fold protocol -- so its (unaugmented)
    train split (500/class) is used instead.
    """
    import torch
    from .data import torch_datasets_for
    from .models import build_model
    path = feature_path(cfg, epoch)
    if os.path.exists(path):
        return path
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _, ev, te, num_classes = torch_datasets_for(cfg)
    # cifar100s: its test split has only 100/class.  random_labels: the
    # fitted labels exist only on the train images (E10 control).
    split = ev if (cfg.dataset == "cifar100s" or cfg.random_labels) else te
    # ViT-B/16 at 224px: attention buffers at batch 512 exceed 24G on CPU
    bs = 32 if cfg.arch == "vit_b16" else 512
    loader = torch.utils.data.DataLoader(split, batch_size=bs, num_workers=2)
    model = build_model(cfg, num_classes).to(device)
    ck = torch.load(os.path.join(CKPT_DIR, cfg.name, f"epoch{epoch}.pt"),
                    map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()

    feats, labels = {}, []
    with torch.no_grad():
        for x, y in loader:
            _, f = model(x.to(device), return_features=True)
            for k, v in f.items():
                feats.setdefault(k, []).append(v.cpu().numpy())
            labels.append(y.numpy())
    out = {k: np.concatenate(v) for k, v in feats.items()}
    out["labels"] = np.concatenate(labels)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **out)
    return path


def load_features(cfg, epoch):
    z = np.load(feature_path(cfg, epoch))
    labels = z["labels"]
    layers = [k for k in z.files if k != "labels"]
    return {k: z[k] for k in layers}, labels


def class_clouds(features, labels, classes, m, d0, fold=None, n_folds=1,
                 seed=0):
    """PCA-project a layer's features and cut per-class clouds.

    PCA is fit once on ALL classes' features (comparability within layer).
    fold=None: the first m points per class.  fold=i: the i-th of n_folds
    disjoint folds with m points per class each.
    """
    from sklearn.decomposition import PCA
    proj = PCA(n_components=d0, random_state=seed).fit_transform(
        features.astype(np.float64))
    clouds = []
    for c in classes:
        idx = np.flatnonzero(labels == c)
        if fold is None:
            take = idx[:m]
        else:
            take = idx[fold * m:(fold + 1) * m]
        if len(take) < m:
            raise ValueError(f"class {c}: need {m}, have {len(take)}")
        clouds.append(proj[take])
    return clouds


if __name__ == "__main__":
    import sys
    cfg = CONFIGS[int(sys.argv[1])]
    for ep in checkpoint_epochs(cfg):
        print(extract_checkpoint(cfg, ep), flush=True)
