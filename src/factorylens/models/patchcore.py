"""PatchCore-style anomaly detection on RGB images (Roth et al., CVPR 2022).

Fit: embed every patch of the normal training images and keep them in a
memory bank (thinned with greedy coreset selection when it gets large).
Score: a test patch is as anomalous as its distance to the nearest bank entry.
The image score is the largest patch score.

Only the parts that matter for this project are kept: frozen backbone,
3x3 neighbourhood averaging, layer2 + layer3 at layer2's resolution, channel
reduction by adaptive average pooling, nearest-neighbour scoring and a Gaussian
blur on the upsampled map.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from factorylens.models.backbone import Backbone, pick_device, to_batch


@dataclass(frozen=True)
class PatchCoreConfig:
    arch: str = "wide_resnet50_2"
    layers: tuple[str, ...] = ("layer2", "layer3")
    neighbourhood: int = 3
    embed_dim: int = 1024
    coreset_ratio: float = 0.1
    keep_all_below: int = 10_000  # banks this small (k-shot) are not thinned
    projection_dim: int = 128
    blur_sigma: float = 4.0
    seed: int = 0
    # patch grid side; None means the resolution of the first listed layer.
    # Fixing it (e.g. 28) lets different layer choices be compared fairly.
    grid: int | None = None

    def as_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["layers"] = list(self.layers)
        return d


def greedy_coreset(
    x: torch.Tensor, m: int, projection_dim: int = 128, seed: int = 0
) -> torch.Tensor:
    """Indices of m rows chosen by greedy k-center selection.

    Each step adds the row farthest from everything chosen so far, so the
    subset covers the whole bank including rare patches. Distances are
    computed after a random projection to `projection_dim`, as in the paper.

    The loop runs on the CPU with squared distances from one matrix-vector
    product per step: on MPS the ~20k tiny steps are dominated by kernel
    launches (3x slower), and the subtract-then-norm form is 2x slower.
    """
    n, d = x.shape
    if m >= n:
        return torch.arange(n, device=x.device)
    g = torch.Generator().manual_seed(seed)
    z = x.detach().float().cpu()
    if projection_dim and d > projection_dim:
        z = z @ (torch.randn(d, projection_dim, generator=g) / math.sqrt(projection_dim))
    sq = (z * z).sum(dim=1)
    start = int(torch.randint(n, (1,), generator=g))
    chosen = [start]
    min_dist = (sq + sq[start] - 2 * (z @ z[start])).clamp_min_(0)
    for _ in range(m - 1):
        i = int(torch.argmax(min_dist))
        chosen.append(i)
        torch.minimum(min_dist, (sq + sq[i] - 2 * (z @ z[i])).clamp_min_(0), out=min_dist)
    return torch.tensor(chosen, device=x.device)


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


def nearest_distance(query: torch.Tensor, bank: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    parts = [
        torch.cdist(query[i : i + chunk], bank).min(dim=1).values
        for i in range(0, len(query), chunk)
    ]
    return torch.cat(parts)


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
        self.bank: torch.Tensor | None = None
        self._pool: torch.Tensor | None = None

    def embed(self, images: np.ndarray) -> torch.Tensor:
        """(N, S, S, 3) uint8 -> (N, h, w, D) patch embeddings."""
        out = []
        k = self.config.neighbourhood
        for i in range(0, len(images), self.batch_size):
            feats = self.backbone(to_batch(list(images[i : i + self.batch_size])).to(self.device))
            maps = [
                F.avg_pool2d(feats[name], k, stride=1, padding=k // 2)
                for name in self.config.layers
            ]
            g = self.config.grid
            size = (g, g) if g else (maps[0].shape[-2], maps[0].shape[-1])
            maps = [_to_grid(m, size) for m in maps]
            x = torch.cat(maps, dim=1).permute(0, 2, 3, 1)  # (B, h, w, C)
            b, h, w, c = x.shape
            if self._pool is None or self._pool.shape[0] != c:
                self._pool = channel_pooling_matrix(c, self.config.embed_dim).to(self.device)
            out.append(x @ self._pool)
        return torch.cat(out)

    def fit(self, images: np.ndarray) -> PatchCore:
        emb = self.embed(images)
        flat = emb.reshape(-1, emb.shape[-1])
        if len(flat) > self.config.keep_all_below:
            m = max(1, int(len(flat) * self.config.coreset_ratio))
            idx = greedy_coreset(flat, m, self.config.projection_dim, self.config.seed)
            flat = flat[idx]
        self.bank = flat.contiguous()
        return self

    def score(self, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Image scores (N,) and pixel maps (N, S, S) at the input resolution."""
        if self.bank is None:
            raise RuntimeError("call fit() before score()")
        size = images.shape[1]
        image_scores, maps = [], []
        for i in range(0, len(images), self.batch_size):
            emb = self.embed(images[i : i + self.batch_size])
            b, h, w, d = emb.shape
            dist = nearest_distance(emb.reshape(-1, d), self.bank).reshape(b, 1, h, w)
            image_scores.append(dist.amax(dim=(1, 2, 3)).cpu())
            up = F.interpolate(dist, size=(size, size), mode="bilinear", align_corners=False)
            maps.append(up[:, 0].cpu())
        pixel = torch.cat(maps).numpy()
        if self.config.blur_sigma > 0:
            pixel = np.stack([ndimage.gaussian_filter(m, self.config.blur_sigma) for m in pixel])
        return torch.cat(image_scores).numpy(), pixel
