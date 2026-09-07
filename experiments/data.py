"""Datasets: CIFAR-10, a fixed 10-class CIFAR-100 subset, MNIST.

Torch loaders for training; plain numpy accessors for the ECP side (which
never needs torch).  All downloads land in common.DATA_DIR; run
`python -m experiments.data` once on a login node to prefetch everything
(compute nodes may lack internet).
"""

import numpy as np

from .common import DATA_DIR

# A fixed, semantically clusterable 10-class subset of CIFAR-100 (E2):
CIFAR100_SUBSET = (3, 42, 43, 88, 97,   # bear, leopard, lion, tiger, wolf
                   15, 19, 21, 31, 38)  # camel, cattle, chimpanzee, elephant, kangaroo


def torch_datasets(name, augment):
    import torch
    import torchvision as tv
    import torchvision.transforms as T

    if name in ("cifar10", "cifar100s"):
        mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
        aug = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()] if augment else []
        tf_train = T.Compose(aug + [T.ToTensor(), T.Normalize(mean, std)])
        tf_test = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        if name == "cifar10":
            tr = tv.datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=tf_train)
            te = tv.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=tf_test)
            return tr, te, 10
        tr = tv.datasets.CIFAR100(DATA_DIR, train=True, download=True, transform=tf_train)
        te = tv.datasets.CIFAR100(DATA_DIR, train=False, download=True, transform=tf_test)
        tr = _subset_cifar100(tr)
        te = _subset_cifar100(te)
        return tr, te, 10
    if name == "mnist":
        tf = T.Compose([T.ToTensor()])
        tr = tv.datasets.MNIST(DATA_DIR, train=True, download=True, transform=tf)
        te = tv.datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)
        return tr, te, 10
    raise ValueError(name)


def _subset_cifar100(ds):
    import torch
    targets = np.asarray(ds.targets)
    keep = np.isin(targets, CIFAR100_SUBSET)
    ds.data = ds.data[keep]
    remap = {c: i for i, c in enumerate(CIFAR100_SUBSET)}
    ds.targets = [remap[t] for t in targets[keep]]
    return ds


class TwoViews:
    """SimCLR two-view transform wrapper."""

    def __init__(self, tf):
        self.tf = tf

    def __call__(self, x):
        return self.tf(x), self.tf(x)


def torch_datasets_for(cfg):
    """Config-aware datasets: SimCLR two-view training, 224px for vit_b16.

    Returns (train_ds, eval_train_ds, test_ds, num_classes); eval_train_ds is
    the train split under test transforms (used for SimCLR probe eval).
    """
    import torchvision as tv
    import torchvision.transforms as T

    if cfg.arch == "vit_b16":
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        tf_train = T.Compose([
            T.Resize(224), T.RandomCrop(224, padding=16),
            T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)])
        tf_test = T.Compose([T.Resize(224), T.ToTensor(),
                             T.Normalize(mean, std)])
        tr = tv.datasets.CIFAR10(DATA_DIR, train=True, download=True,
                                 transform=tf_train)
        ev = tv.datasets.CIFAR10(DATA_DIR, train=True, download=True,
                                 transform=tf_test)
        te = tv.datasets.CIFAR10(DATA_DIR, train=False, download=True,
                                 transform=tf_test)
        return tr, ev, te, 10

    if cfg.objective == "simclr":
        mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
        tf_view = T.Compose([
            T.RandomResizedCrop(32, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(), T.Normalize(mean, std)])
        tf_test = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        tr = tv.datasets.CIFAR10(DATA_DIR, train=True, download=True,
                                 transform=TwoViews(tf_view))
        ev = tv.datasets.CIFAR10(DATA_DIR, train=True, download=True,
                                 transform=tf_test)
        te = tv.datasets.CIFAR10(DATA_DIR, train=False, download=True,
                                 transform=tf_test)
        return tr, ev, te, 10

    tr, te, k = torch_datasets(cfg.dataset, cfg.augment)
    ev, _, _ = torch_datasets(cfg.dataset, augment=False)
    if getattr(cfg, "random_labels", False):
        # E10 control: one fixed random relabeling of the TRAIN split, shared
        # by the training loader and the eval-transform copy so that features
        # are extracted with exactly the labels the network fitted.  A
        # permutation keeps the class sizes equal; the test split keeps its
        # true labels (test accuracy then reads as the chance-level check).
        perm = np.random.default_rng(12345 + cfg.seed).permutation(len(tr.targets))
        shuffled = [tr.targets[i] for i in perm]
        tr.targets = list(shuffled)
        ev.targets = list(shuffled)
    return tr, ev, te, k


def mnist_numpy(train=True):
    """MNIST as (n, 784) float32 in [0,1] + int labels, torch-free at call
    site if the files are already downloaded (uses torchvision when present,
    else sklearn's fetch_openml)."""
    try:
        import torchvision as tv
        ds = tv.datasets.MNIST(DATA_DIR, train=train, download=True)
        X = ds.data.numpy().reshape(len(ds), -1).astype(np.float32) / 255.0
        y = ds.targets.numpy().astype(int)
        return X, y
    except ImportError:
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True,
                            as_frame=False, data_home=DATA_DIR)
        X = X.astype(np.float32) / 255.0
        y = y.astype(int)
        return (X[:60000], y[:60000]) if train else (X[60000:], y[60000:])


if __name__ == "__main__":
    # Prefetch all datasets (login node).
    for name in ("cifar10", "cifar100s", "mnist"):
        torch_datasets(name, augment=False)
        print(f"{name}: ok")
