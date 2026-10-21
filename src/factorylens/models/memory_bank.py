"""Nearest-neighbour anomaly scoring against a bank of normal patch features.

This is the part of PatchCore that does not care where the features come
from, so the RGB, 3D and fused detectors all share it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage


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


def nearest(
    query: torch.Tensor, bank: torch.Tensor, chunk: int = 4096
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distance to and index of the nearest bank row, for every query row."""
    dists, idx = [], []
    for i in range(0, len(query), chunk):
        d = torch.cdist(query[i : i + chunk], bank).min(dim=1)
        dists.append(d.values)
        idx.append(d.indices)
    return torch.cat(dists), torch.cat(idx)


def nearest_distance(query: torch.Tensor, bank: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    return nearest(query, bank, chunk)[0]


@dataclass(frozen=True)
class BankConfig:
    coreset_ratio: float = 0.1
    keep_all_below: int = 10_000  # banks this small (k-shot) are not thinned
    projection_dim: int = 128
    blur_sigma: float = 4.0
    # PatchCore's eq. 7: scale the max patch score down when its nearest
    # normal patch sits in a dense part of the bank. Off by default; BTF uses it.
    reweight: bool = False
    seed: int = 0


class MemoryBank:
    def __init__(self, config: BankConfig | None = None) -> None:
        self.config = config or BankConfig()
        self.bank: torch.Tensor | None = None

    def fit(self, patches: torch.Tensor) -> MemoryBank:
        """(N, h, w, D) features of normal images."""
        flat = patches.reshape(-1, patches.shape[-1])
        if len(flat) > self.config.keep_all_below:
            m = max(1, int(len(flat) * self.config.coreset_ratio))
            flat = flat[greedy_coreset(flat, m, self.config.projection_dim, self.config.seed)]
        self.bank = flat.contiguous()
        return self

    def score(self, patches: torch.Tensor, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """(N, h, w, D) features -> image scores (N,) and pixel maps (N, *size)."""
        if self.bank is None:
            raise RuntimeError("call fit() before score()")
        n, h, w, d = patches.shape
        image_scores, maps = [], []
        for x in patches:
            dist, idx = nearest(x.reshape(-1, d), self.bank)
            image_scores.append(self._image_score(x.reshape(-1, d), dist, idx))
            up = F.interpolate(
                dist.reshape(1, 1, h, w), size=size, mode="bilinear", align_corners=False
            )
            maps.append(up[0, 0].cpu())
        pixel = torch.stack(maps).numpy()
        if self.config.blur_sigma > 0:
            pixel = np.stack([ndimage.gaussian_filter(m, self.config.blur_sigma) for m in pixel])
        return np.array(image_scores, dtype=np.float32), pixel

    def _image_score(self, x: torch.Tensor, dist: torch.Tensor, idx: torch.Tensor) -> float:
        star = int(torch.argmax(dist))
        s_star = dist[star]
        if not self.config.reweight:
            return float(s_star)
        assert self.bank is not None
        # nearest bank entry of the most anomalous patch, and that entry's own neighbours
        m_star = self.bank[idx[star]]
        neighbours = torch.cdist(m_star[None], self.bank)[0].topk(3, largest=False).indices[1:]
        knn = torch.linalg.vector_norm(x[star] - self.bank[neighbours], dim=1)
        scale = math.sqrt(x.shape[1])  # keeps exp() finite for long feature vectors
        w = 1 - torch.exp(s_star / scale) / torch.exp(knn / scale).sum()
        return float(w * s_star)
