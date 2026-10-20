import numpy as np
import pytest
from scipy import ndimage

from factorylens.eval import metrics


def brute_force_aupro(masks, maps, fpr_limit=0.3, n_thresholds=4000):
    """Slow textbook version: loop over thresholds, then integrate."""
    masks = masks.astype(bool)
    regions = []
    for i, m in enumerate(masks):
        labels, n = ndimage.label(m)
        regions += [(i, labels == j) for j in range(1, n + 1)]
    normal = ~masks
    fprs, pros = [], []
    for t in np.linspace(maps.max() + 1e-9, maps.min() - 1e-9, n_thresholds):
        pred = maps >= t
        fprs.append((pred & normal).sum() / normal.sum())
        pros.append(np.mean([pred[i][r].mean() for i, r in regions]))
    fprs, pros = np.array(fprs), np.array(pros)
    keep = fprs <= fpr_limit
    x = np.r_[fprs[keep], fpr_limit]
    y = np.r_[pros[keep], np.interp(fpr_limit, fprs, pros)]
    return np.trapezoid(y, x) / fpr_limit


def toy(seed=0, n=4, size=24):
    rng = np.random.default_rng(seed)
    masks = np.zeros((n, size, size), bool)
    masks[0, 2:5, 2:5] = True  # small region
    masks[0, 10:20, 10:20] = True  # large region in the same image
    masks[2, 5:8, 15:22] = True
    maps = rng.normal(0, 1, masks.shape)
    maps[masks] += 1.5  # informative but imperfect
    return masks, maps


def test_aupro_matches_the_brute_force_version():
    for seed in range(3):
        masks, maps = toy(seed)
        assert metrics.aupro(masks, maps) == pytest.approx(brute_force_aupro(masks, maps), abs=2e-3)


def test_aupro_extremes():
    masks, _ = toy()
    assert metrics.aupro(masks, masks.astype(float)) == pytest.approx(1.0)
    assert metrics.aupro(masks, -masks.astype(float)) == pytest.approx(0.0, abs=1e-9)


def test_aupro_of_random_scores_is_about_fpr_limit_over_two():
    rng = np.random.default_rng(1)
    masks = np.zeros((20, 32, 32), bool)
    masks[:, 8:16, 8:16] = True
    # random scores: PRO ~= FPR, so the normalised area up to 0.3 is ~0.15
    assert metrics.aupro(masks, rng.random(masks.shape)) == pytest.approx(0.15, abs=0.02)


def test_aupro_weights_small_regions_like_large_ones():
    masks = np.zeros((1, 40, 40), bool)
    masks[0, 0:2, 0:2] = True  # 4 px
    masks[0, 10:30, 10:30] = True  # 400 px
    only_big = np.where(masks, 0.0, -1.0)
    only_big[0, 0:2, 0:2] = -2.0  # small region found last
    # pixel AUROC barely notices the 4 missed pixels, PRO loses half its regions
    assert metrics.pixel_auroc(masks, only_big) > 0.99
    assert metrics.aupro(masks, only_big) < 0.6


def test_ties_are_handled_as_one_threshold():
    masks = np.zeros((1, 4, 4), bool)
    masks[0, 0, 0] = True
    flat = np.zeros((1, 4, 4))
    fpr, pro = metrics.pro_curve(masks, flat)
    assert len(fpr) == 2  # one step for the whole tied block, not 16
    np.testing.assert_allclose(fpr, [0, 1])
    np.testing.assert_allclose(pro, [0, 1])


def test_aupro_needs_a_defect():
    with pytest.raises(ValueError):
        metrics.aupro(np.zeros((2, 4, 4), bool), np.zeros((2, 4, 4)))


def test_image_and_pixel_auroc():
    assert metrics.image_auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert metrics.image_auroc([0, 1, 0, 1], [0.1, 0.2, 0.3, 0.4]) == 0.75
    with pytest.raises(ValueError):
        metrics.image_auroc([1, 1], [0.3, 0.4])
    masks, maps = toy()
    assert 0.5 < metrics.pixel_auroc(masks, maps) < 1.0
    with pytest.raises(ValueError):
        metrics.pixel_auroc(masks, maps[:, :-1])


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    labels = np.r_[np.zeros(60), np.ones(40)].astype(bool)
    scores = rng.normal(0, 1, 100) + labels * 1.0
    point = metrics.image_auroc(labels, scores)
    lo, hi = metrics.bootstrap_ci(labels, scores, n_boot=300)
    assert lo < point < hi
    assert 0.05 < hi - lo < 0.4
    assert metrics.bootstrap_ci(labels, scores, n_boot=300) == (lo, hi)  # seeded
