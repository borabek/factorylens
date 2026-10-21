"""PatchCore-style anomaly detection on RGB images (Roth et al., CVPR 2022).

Fit: embed every patch of the normal training images and keep them in a
memory bank (thinned with greedy coreset selection when it gets large).
Score: a test patch is as anomalous as its distance to the nearest bank entry.
The image score is the largest patch score.

Only the parts that matter for this project are kept: frozen backbone,
3x3 neighbourhood averaging, layer2 + layer3 at layer2's resolution, channel
reduction by adaptive average pooling, nearest-neighbour scoring and a Gaussian
blur on the upsampled map. The bank and the scoring live in memory_bank.py.

BTF_RGB reproduces the RGB recipe from the Back to the Feature code: no
padding in the 3x3 average (28 -> 26 -> back to 28), all 1536 channels kept,
and the reweighted image score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from factorylens.models.backbone import Backbone, pick_device, to_batch
from factorylens.models.memory_bank import (  # noqa: F401  (re-exported)
    BankConfig,
    MemoryBank,
    greedy_coreset,
    nearest_distance,
)


@dataclass(frozen=True)
class PatchCoreConfig:
    arch: str = "wide_resnet50_2"
    layers: tuple[str, ...] = ("layer2", "layer3")
    neighbourhood: int = 3
    embed_dim: int | None = 1024  # None keeps every backbone channel
    coreset_ratio: float = 0.1
    keep_all_below: int = 10_000  # banks this small (k-shot) are not thinned
    projection_dim: int = 128
    blur_sigma: float = 4.0
    seed: int = 0
    # patch grid side; None means the resolution of the first listed layer.
    # Fixing it (e.g. 28) lets different layer choices be compared fairly.
    grid: int | None = None
    pad: bool = True  # zero-pad the 3x3 average so the grid keeps its size
    reweight: bool = False

    def bank_config(self) -> BankConfig:
        return BankConfig(
            coreset_ratio=self.coreset_ratio,
            keep_all_below=self.keep_all_below,
            projection_dim=self.projection_dim,
            blur_sigma=self.blur_sigma,
            reweight=self.reweight,
            seed=self.seed,
        )

    def as_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["layers"] = list(self.layers)
        return d


def channel_pooling_matrix(c: int, d: int) -> torch.Tensor:
    """(c, d) matrix that does what F.adaptive_avg_pool1d(x, d) does to c channels.

    adaptive_avg_pool1d has no MPS kernel when c is not a multiple of d
    (1536 -> 1024 here), and a matmul runs everywhere.
    """
    m = torch.zeros(c, d)
    for j in range(d):
        start, end = (j * c) // d, -((-(j + 1) * c) // d)  # floor, ceil
        m[start:end, j] = 1.0 / (end - start)
    return m


def _to_grid(m: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(m.shape[-2:]) == size:
        return m
    if m.shape[-1] > size[-1]:
        return F.adaptive_avg_pool2d(m, size)  # finer layer: average down
    return F.interpolate(m, size=size, mode="bilinear", align_corners=False)


class PatchCore:
    def __init__(
        self,
        config: PatchCoreConfig | None = None,
        device: str | torch.device | None = None,
        pretrained: bool = True,
        batch_size: int = 16,
    ) -> None:
        self.config = config or PatchCoreConfig()
        self.device = torch.device(device) if device else pick_device()
        self.backbone = Backbone(self.config.arch, self.config.layers, pretrained).to(self.device)
        self.batch_size = batch_size
        self.memory = MemoryBank(self.config.bank_config())
        self._pool: torch.Tensor | None = None

    @property
    def bank(self) -> torch.Tensor | None:
        return self.memory.bank

    def embed(self, images: np.ndarray) -> torch.Tensor:
        """(N, S, S, 3) uint8 -> (N, h, w, D) patch embeddings."""
        out = []
        k = self.config.neighbourhood
        pad = k // 2 if self.config.pad else 0
        for i in range(0, len(images), self.batch_size):
            feats = self.backbone(to_batch(list(images[i : i + self.batch_size])).to(self.device))
            first = feats[self.config.layers[0]]
            g = self.config.grid
            size = (g, g) if g else (first.shape[-2], first.shape[-1])
            # zero padding at the border, as in PatchCore's own patch extraction
            maps = [
                _to_grid(F.avg_pool2d(feats[name], k, stride=1, padding=pad), size)
                for name in self.config.layers
            ]
            x = torch.cat(maps, dim=1).permute(0, 2, 3, 1)  # (B, h, w, C)
            c = x.shape[-1]
            if self.config.embed_dim is None:
                out.append(x)
                continue
            if self._pool is None or self._pool.shape[0] != c:
                self._pool = channel_pooling_matrix(c, self.config.embed_dim).to(self.device)
            out.append(x @ self._pool)
        return torch.cat(out)

    def fit(self, images: np.ndarray) -> PatchCore:
        self.memory.fit(self.embed(images))
        return self

    def score(self, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Image scores (N,) and pixel maps (N, H, W) at the input resolution."""
        if self.bank is None:
            raise RuntimeError("call fit() before score()")
        size = (images.shape[1], images.shape[2])
        scores, maps = [], []
        for i in range(0, len(images), self.batch_size):
            s, m = self.memory.score(self.embed(images[i : i + self.batch_size]), size)
            scores.append(s)
            maps.append(m)
        return np.concatenate(scores), np.concatenate(maps)


BTF_RGB = PatchCoreConfig(embed_dim=None, pad=False, reweight=True, grid=28)
