"""Does a set of normal images cover the joint appearance/geometry space?

Early fusion matches a test patch to the nearest training patch in the
concatenated (RGB, FPFH) space, so the training set has to contain that
combination of appearance and geometry. Matching each half on its own only
needs each half to have been seen somewhere. The joint distance can never be
smaller than the separate one:

    d_joint^2 = min_j (|r - r_j|^2 + |g - g_j|^2)
             >= min_j |r - r_j|^2 + min_j |g - g_j|^2 = d_sep^2

and the relative difference, averaged over the patches of an image, is the
coverage gap. It needs no labels: computed on held-out normal images it says
how much of early fusion's score would be "this combination is new" rather
than "this patch is unusual".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from factorylens.models.memory_bank import nearest_distance


@dataclass
class Coverage:
    joint: np.ndarray  # (N,) max over patches of the joint distance (early fusion's score)
    separate: np.ndarray  # (N,) max over patches of sqrt(d_rgb^2 + d_geo^2)
    gap: np.ndarray  # (N,) mean(d_joint^2 - d_sep^2) / mean(d_sep^2) per image


def coverage(
    train_rgb: torch.Tensor,
    train_geo: torch.Tensor,
    rgb: torch.Tensor,
    geo: torch.Tensor,
    geo_weight: float = 1.0,
) -> Coverage:
    """Train features (K, h, w, D), query features (N, h, w, D); geo is scaled by geo_weight."""
    bank_rgb = train_rgb.reshape(-1, train_rgb.shape[-1])
    bank_geo = (train_geo * geo_weight).reshape(-1, train_geo.shape[-1])
    bank_joint = torch.cat([bank_rgb, bank_geo], dim=-1)
    joint, separate, gap = [], [], []
    for r, g in zip(rgb, geo * geo_weight, strict=True):
        qr, qg = r.reshape(-1, r.shape[-1]), g.reshape(-1, g.shape[-1])
        d_sep = nearest_distance(qr, bank_rgb) ** 2 + nearest_distance(qg, bank_geo) ** 2
        d_joint = nearest_distance(torch.cat([qr, qg], dim=-1), bank_joint) ** 2
        joint.append(float(d_joint.max().sqrt()))
        separate.append(float(d_sep.max().sqrt()))
        gap.append(float((d_joint - d_sep).mean() / (d_sep.mean() + 1e-12)))
    return Coverage(np.array(joint), np.array(separate), np.array(gap))
