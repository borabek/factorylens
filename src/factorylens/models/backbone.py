"""Frozen ImageNet backbones that return intermediate feature maps.

Nothing here is trained. The RGB detector compares patch features of a test
image with patch features of normal training images, so all it needs from the
network is a good local description at a few resolutions.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torchvision
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LAYERS = ("layer1", "layer2", "layer3", "layer4")

_ARCHS = {
    "resnet18": (torchvision.models.resnet18, torchvision.models.ResNet18_Weights.IMAGENET1K_V1),
    "wide_resnet50_2": (
        torchvision.models.wide_resnet50_2,
        torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1,
    ),
}


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Backbone(nn.Module):
    mean: torch.Tensor
    std: torch.Tensor

    def __init__(
        self,
        arch: str = "wide_resnet50_2",
        layers: Sequence[str] = ("layer2", "layer3"),
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if arch not in _ARCHS:
            raise ValueError(f"unknown arch {arch!r}, expected one of {sorted(_ARCHS)}")
        unknown = set(layers) - set(LAYERS)
        if unknown:
            raise ValueError(f"unknown layers {sorted(unknown)}")
        build, weights = _ARCHS[arch]
        net = build(weights=weights if pretrained else None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        # keep only the blocks up to the deepest layer we need
        deepest = max(LAYERS.index(name) for name in layers)
        self.blocks = nn.ModuleDict({name: getattr(net, name) for name in LAYERS[: deepest + 1]})
        self.layers = tuple(layers)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """(B, 3, H, W) float in [0, 1] -> {layer name: (B, C, h, w)}."""
        x = self.stem((images - self.mean) / self.std)
        out = {}
        for name, block in self.blocks.items():
            x = block(x)
            if name in self.layers:
                out[name] = x
        return out


def to_batch(images: Sequence[np.ndarray]) -> torch.Tensor:
    """List of (H, W, 3) uint8 arrays of equal size -> (B, 3, H, W) float in [0, 1]."""
    stacked = np.stack(images).astype(np.float32) / 255.0
    return torch.from_numpy(stacked).permute(0, 3, 1, 2).contiguous()
