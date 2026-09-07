"""Architectures: CIFAR ResNet (6n+2, width multiplier), a small ViT for
32x32 inputs, and the MNIST MLP.  All expose `feature_points(name)` --- the
named stage boundaries at which class clouds are extracted."""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# CIFAR ResNet (He et al. 6n+2: depths 20, 32, 44, 56)
# --------------------------------------------------------------------------- #
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False),
                nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + (x if self.short is None else self.short(x))
        return F.relu(out)


class CifarResNet(nn.Module):
    def __init__(self, depth=20, width=1, num_classes=10):
        super().__init__()
        assert (depth - 2) % 6 == 0, "depth must be 6n+2"
        n = (depth - 2) // 6
        w = [16 * width, 32 * width, 64 * width]
        self.conv1 = nn.Conv2d(3, w[0], 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(w[0])
        self.stage1 = self._stage(w[0], w[0], n, 1)
        self.stage2 = self._stage(w[0], w[1], n, 2)
        self.stage3 = self._stage(w[1], w[2], n, 2)
        self.fc = nn.Linear(w[2], num_classes)

    @staticmethod
    def _stage(cin, cout, n, stride):
        layers = [BasicBlock(cin, cout, stride)]
        layers += [BasicBlock(cout, cout) for _ in range(n - 1)]
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        feats = {}
        x = F.relu(self.bn1(self.conv1(x)))
        feats["stem"] = x
        x = self.stage1(x); feats["stage1"] = x
        x = self.stage2(x); feats["stage2"] = x
        x = self.stage3(x); feats["stage3"] = x
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        feats["penult"] = x
        logits = self.fc(x)
        if return_features:
            # spatial features -> global average pooled vectors
            out = {k: (v.mean(dim=(2, 3)) if v.dim() == 4 else v)
                   for k, v in feats.items()}
            return logits, out
        return logits

    feature_points = ("stem", "stage1", "stage2", "stage3", "penult")


# --------------------------------------------------------------------------- #
# Small ViT for 32x32 (patch 4, dim 192; ~ ViT-Tiny scale)
# --------------------------------------------------------------------------- #
class SmallViT(nn.Module):
    def __init__(self, depth=8, dim=192, heads=3, num_classes=10, patch=4):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, patch, patch)
        n_tok = (32 // patch) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n_tok + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            [nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=4 * dim,
                dropout=0.0, activation="gelu", batch_first=True,
                norm_first=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, num_classes)
        self.depth = depth

    def forward(self, x, return_features=False):
        feats = {}
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(x.size(0), -1, -1), x], dim=1)
        x = x + self.pos
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i + 1) % 2 == 0:
                feats[f"block{i + 1}"] = x[:, 0]  # CLS token
        x = self.norm(x)
        feats["penult"] = x[:, 0]
        logits = self.fc(x[:, 0])
        if return_features:
            return logits, feats
        return logits

    @property
    def feature_points(self):
        return tuple(f"block{i}" for i in range(2, self.depth + 1, 2)) + ("penult",)


# --------------------------------------------------------------------------- #
# MNIST MLP (E1 preliminary architecture: 784 -> 128 -> 64 -> k)
# --------------------------------------------------------------------------- #
class MnistMLP(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x, return_features=False):
        x = x.flatten(1)
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        logits = self.fc3(h2)
        if return_features:
            return logits, {"layer1": h1, "layer2": h2}
        return logits

    feature_points = ("layer1", "layer2")


# --------------------------------------------------------------------------- #
# SimCLR head over the CIFAR ResNet backbone
# --------------------------------------------------------------------------- #
class SimCLRNet(nn.Module):
    """CifarResNet backbone + 2-layer projection head (NT-Xent training).

    return_features exposes the BACKBONE stages (what measure.py analyzes);
    the projection is a training-time artifact and is not measured.
    """

    def __init__(self, depth=20, width=1, proj_dim=128):
        super().__init__()
        self.backbone = CifarResNet(depth, width, num_classes=10)
        feat = 64 * width
        self.proj = nn.Sequential(
            nn.Linear(feat, feat), nn.ReLU(inplace=True),
            nn.Linear(feat, proj_dim))

    def forward(self, x, return_features=False):
        logits, feats = self.backbone(x, return_features=True)
        z = self.proj(feats["penult"])
        if return_features:
            return z, feats
        return z

    feature_points = CifarResNet.feature_points


# --------------------------------------------------------------------------- #
# Pretrained ViT-B/16 (torchvision), finetuned at 224x224
# --------------------------------------------------------------------------- #
class ViTB16(nn.Module):
    """torchvision vit_b_16 with per-block CLS-token feature taps."""

    TAP_BLOCKS = (3, 6, 9, 12)  # 1-indexed encoder layers to tap

    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        import torchvision
        weights = (torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
                   if pretrained else None)
        vit = torchvision.models.vit_b_16(weights=weights)
        vit.heads = nn.Identity()
        self.vit = vit
        self.fc = nn.Linear(vit.hidden_dim, num_classes)

    def forward(self, x, return_features=False):
        v = self.vit
        x = v._process_input(x)
        cls = v.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + v.encoder.pos_embedding
        x = v.encoder.dropout(x)
        feats = {}
        for i, layer in enumerate(v.encoder.layers, start=1):
            x = layer(x)
            if i in self.TAP_BLOCKS:
                feats[f"block{i}"] = x[:, 0]
        x = v.encoder.ln(x)
        feats["penult"] = x[:, 0]
        logits = self.fc(x[:, 0])
        if return_features:
            return logits, feats
        return logits

    @property
    def feature_points(self):
        return tuple(f"block{i}" for i in self.TAP_BLOCKS) + ("penult",)


def build_model(cfg, num_classes):
    if cfg.arch == "resnet":
        if cfg.objective == "simclr":
            return SimCLRNet(cfg.depth, cfg.width)
        return CifarResNet(cfg.depth, cfg.width, num_classes)
    if cfg.arch == "vit":
        return SmallViT(depth=cfg.depth, num_classes=num_classes)
    if cfg.arch == "vit_b16":
        return ViTB16(num_classes, pretrained=cfg.pretrained)
    if cfg.arch == "mlp":
        return MnistMLP(num_classes)
    raise ValueError(cfg.arch)
