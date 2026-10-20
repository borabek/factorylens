"""Detection and localisation metrics.

image_auroc  - one score per image, label = defective or not
pixel_auroc  - every pixel of every test image pooled together
aupro        - area under the per-region-overlap curve up to a false positive
               rate of 0.3, normalised to [0, 1] (Bergmann et al., MVTec AD).
               Unlike pixel AUROC it weights every defect region equally, so a
               tiny scratch counts as much as a large stain.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import ndimage
from sklearn.metrics import roc_auc_score


def image_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(bool)
    if labels.all() or not labels.any():
        raise ValueError("image AUROC needs both normal and defective images")
    return float(roc_auc_score(labels, scores))


def pixel_auroc(masks: np.ndarray, maps: np.ndarray) -> float:
    masks = np.asarray(masks).astype(bool)
    if masks.shape != np.shape(maps):
        raise ValueError(f"masks {masks.shape} and maps {np.shape(maps)} differ")
    return float(roc_auc_score(masks.ravel(), np.asarray(maps).ravel()))


def pro_curve(masks: np.ndarray, maps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """FPR and mean per-region overlap at every distinct threshold, sorted by FPR.

    Instead of looping over thresholds, each pixel gets a weight: a normal
    pixel adds 1/n_normal to the FPR once the threshold drops below its score,
    and a pixel of defect region j adds 1/(|region j| * n_regions) to the PRO.
    Sorting by score and taking cumulative sums gives both curves exactly.
    """
    masks = np.asarray(masks).astype(bool)
    maps = np.asarray(maps, dtype=np.float64)
    if masks.shape != maps.shape:
        raise ValueError(f"masks {masks.shape} and maps {maps.shape} differ")

    region_weight = np.zeros(masks.shape)
    n_regions = 0
    for i, m in enumerate(masks):
        labels, n = ndimage.label(m)
        for j in range(1, n + 1):
            region = labels == j
            region_weight[i][region] = 1.0 / region.sum()
        n_regions += n
    n_normal = int((~masks).sum())
    if n_regions == 0 or n_normal == 0:
        raise ValueError("PRO needs at least one defect region and one normal pixel")

    fpr_weight = np.where(masks, 0.0, 1.0 / n_normal).ravel()
    pro_weight = (region_weight / n_regions).ravel()

    order = np.argsort(-maps.ravel(), kind="stable")
    scores = maps.ravel()[order]
    fpr = np.cumsum(fpr_weight[order])
    pro = np.cumsum(pro_weight[order])
    # pixels with equal scores flip together, so keep only the last of each run
    last = np.r_[scores[1:] != scores[:-1], True]
    fpr, pro = fpr[last], pro[last]
    return np.r_[0.0, fpr], np.r_[0.0, pro]


def aupro(masks: np.ndarray, maps: np.ndarray, fpr_limit: float = 0.3) -> float:
    fpr, pro = pro_curve(masks, maps)
    # cut the curve at fpr_limit, interpolating the PRO value there
    pro_at_limit = np.interp(fpr_limit, fpr, pro)
    keep = fpr <= fpr_limit
    x = np.r_[fpr[keep], fpr_limit]
    y = np.r_[pro[keep], pro_at_limit]
    return float(np.trapezoid(y, x) / fpr_limit)


def bootstrap_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float] = image_auroc,
    n_boot: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile CI over images, resampling normal and defective images separately.

    Stratifying keeps both classes in every resample, which an AUROC needs.
    """
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores)
    pos, neg = np.flatnonzero(labels), np.flatnonzero(~labels)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        stats[b] = metric(labels[idx], scores[idx])
    alpha = (1 - level) / 2
    lo, hi = np.quantile(stats, [alpha, 1 - alpha])
    return float(lo), float(hi)
