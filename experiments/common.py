"""Shared configuration for the disentanglement experiment suite.

Every trainable model in the paper (E2, E3, E5) is a row in CONFIGS; cluster
array tasks index into it.  Paths resolve relative to DISENTANGLE_ROOT
(default: the disentangle/ directory containing this package).
"""

import json
import os
from dataclasses import dataclass

ROOT = os.environ.get(
    "DISENTANGLE_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
DATA_DIR = os.environ.get("DISENTANGLE_DATA", os.path.join(ROOT, "data"))
RESULTS_DIR = os.path.join(ROOT, "results")
CKPT_DIR = os.path.join(ROOT, "checkpoints")

# ECP measurement defaults (shared across experiments so numbers compare).
PCA_DIM = 5        # headline pairwise protocol
TRIPLE_PCA_DIM = 4  # triple scans: k=3 IE has 7 terms; d0=5 is ~7x costlier
POINTS_PER_CLASS = 200
N_PERMUTATIONS = 999
R_GRID_SIZE = 200
STAT = "int"  # profile functional Phi; 'int' = \int |Dchi| dr
FOLD_STAT = "intnorm"  # paired-test fold statistic: scale-free (\int|Dchi|dr / r_max); see ecp.profile_stat
N_FOLDS = 8   # paired subsample folds


@dataclass
class TrainConfig:
    tag: str            # role: 'e3', 'e2', 'pop', 'ablr', 'ssl', 'vitpre'
    arch: str           # 'resnet' | 'vit' | 'mlp' | 'vit_b16'
    dataset: str        # 'cifar10' | 'cifar100s' | 'mnist'
    depth: int = 20
    width: int = 1
    weight_decay: float = 5e-4
    augment: bool = True
    lr: float = 0.1
    epochs: int = 80
    batch_size: int = 128
    seed: int = 0
    dense_checkpoints: bool = False  # E3: 0,1,2,4,8,... + every 10
    objective: str = "ce"            # 'ce' | 'simclr'
    lr_schedule: str = "cosine"      # 'cosine' | 'step' | 'const'
    pretrained: bool = False         # vit_b16: ImageNet weights, finetune
    dis_lambda: float = 0.0          # E6: differentiable disentanglement reg.
    random_labels: bool = False      # E10 control: fit a fixed random labeling

    @property
    def name(self):
        aug = "aug" if self.augment else "noaug"
        base = (f"{self.tag}_{self.arch}{self.depth}w{self.width}"
                f"_{self.dataset}_wd{self.weight_decay:g}_{aug}_s{self.seed}")
        if self.lr_schedule != "cosine":
            base += f"_{self.lr_schedule}"
        if self.objective != "ce":
            base += f"_{self.objective}"
        if self.dis_lambda:
            base += f"_dl{self.dis_lambda:g}"
        if self.random_labels:
            base += "_randlab"
        return base


def build_configs():
    cfgs = []
    # --- E3: training dynamics, ResNet-20, 5 seeds, dense checkpoints ------
    for seed in range(5):
        cfgs.append(TrainConfig(tag="e3", arch="resnet", dataset="cifar10",
                                depth=20, seed=seed, dense_checkpoints=True))
    # --- E2: depth-wise, ResNet-56 on CIFAR-10 + CIFAR-100 subset, ViT ----
    cfgs.append(TrainConfig(tag="e2", arch="resnet", dataset="cifar10", depth=56))
    cfgs.append(TrainConfig(tag="e2", arch="resnet", dataset="cifar100s", depth=56))
    cfgs.append(TrainConfig(tag="e2", arch="vit", dataset="cifar10",
                            depth=8, lr=1e-3, weight_decay=5e-2, epochs=100))
    # --- E5: population grid  4 x 2 x 2 x 2 x 3 = 96 models ---------------
    for depth in (20, 32, 44, 56):
        for width in (1, 2):
            for wd in (0.0, 5e-4):
                for augment in (True, False):
                    for seed in range(3):
                        cfgs.append(TrainConfig(
                            tag="pop", arch="resnet", dataset="cifar10",
                            depth=depth, width=width, weight_decay=wd,
                            augment=augment, seed=seed))
    # ----------------------------------------------------------------------
    # Follow-up campaign (appended AFTER the original 104 so that manifest
    # rows 0-170 and all existing results remain index-stable).
    # C: LR-schedule ablation for the E3 late significance wave.
    for sched in ("step", "const"):
        for seed in range(2):
            cfgs.append(TrainConfig(tag="ablr", arch="resnet",
                                    dataset="cifar10", depth=20, seed=seed,
                                    lr_schedule=sched,
                                    dense_checkpoints=True))
    # B: contrastive (SimCLR) encoder — collapse-free geometry, the most
    # plausible generator of higher-order structure.
    for seed in range(2):
        cfgs.append(TrainConfig(tag="ssl", arch="resnet", dataset="cifar10",
                                depth=20, seed=seed, objective="simclr",
                                lr=0.5, weight_decay=1e-4, epochs=200,
                                batch_size=256))
    # C: pretrained ViT-B/16 finetune — is the from-scratch ViT's
    # head re-entanglement an artifact of data starvation?
    cfgs.append(TrainConfig(tag="vitpre", arch="vit_b16", dataset="cifar10",
                            depth=12, pretrained=True, lr=1e-4,
                            weight_decay=1e-4, epochs=8, batch_size=64))
    # E6: differentiable-regularizer proof of concept. ResNet-20/CIFAR-10 at
    # the E3 recipe (wd 5e-4, aug, cosine); dl=0 is the matched control, and
    # the lambda grid traces the accuracy/robustness-vs-disentanglement curve.
    # Appended last so indices 0-110 and all prior manifests stay stable.
    for dl in (0.0, 0.5, 2.0, 8.0):
        for seed in range(2):
            cfgs.append(TrainConfig(tag="e6", arch="resnet", dataset="cifar10",
                                    depth=20, seed=seed, dis_lambda=dl))
    # E10 control: does fitting ANY labeling induce pairwise dominance, or
    # only fitting the true one?  ResNet-20 memorizes a fixed random
    # relabeling of CIFAR-10 (no augmentation, no weight decay -- the
    # Zhang et al. memorization recipe; 150 epochs to reach ~100% train
    # accuracy).  Features are extracted on the TRAIN images with the labels
    # the network fitted, since random labels do not transfer to test images.
    # Appended last: indices 0-118 stay stable.
    cfgs.append(TrainConfig(tag="e10", arch="resnet", dataset="cifar10",
                            depth=20, weight_decay=0.0, augment=False,
                            epochs=150, random_labels=True))
    return cfgs


CONFIGS = build_configs()


def checkpoint_epochs(cfg):
    if cfg.dense_checkpoints:
        dense = [0, 1, 2, 4, 8, 16, 32]
        tail = list(range(40, cfg.epochs + 1, 10))
        return sorted(set(e for e in dense + tail + [cfg.epochs] if e <= cfg.epochs))
    return sorted(set([0, cfg.epochs // 4, cfg.epochs // 2,
                       3 * cfg.epochs // 4, cfg.epochs]))


def ensure_dirs():
    for d in (DATA_DIR, RESULTS_DIR, CKPT_DIR):
        os.makedirs(d, exist_ok=True)


def dump_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)


if __name__ == "__main__":
    # `python -m experiments.common` prints the config table for sbatch sizing
    for i, c in enumerate(CONFIGS):
        print(f"{i:3d}  {c.name}")
    print(f"total: {len(CONFIGS)}")
