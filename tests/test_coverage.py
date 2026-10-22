import numpy as np
import torch

from factorylens.models.coverage import coverage


def patches(rgb_ids, geo_ids, seed=0):
    """One 2x2 image per pair; appearance and geometry come from two prototypes each."""
    gen = torch.Generator().manual_seed(seed)
    protos_rgb = torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    protos_geo = torch.tensor([[0.0, 0.0], [0.0, 5.0]])
    rgb = torch.stack([protos_rgb[i].expand(2, 2, 3) for i in rgb_ids])
    geo = torch.stack([protos_geo[i].expand(2, 2, 2) for i in geo_ids])
    rgb = rgb + 0.01 * torch.randn(rgb.shape, generator=gen)
    geo = geo + 0.01 * torch.randn(geo.shape, generator=gen)
    return rgb, geo


def test_unseen_combination_has_a_large_gap():
    # training only ever pairs appearance 0 with geometry 0 and 1 with 1
    train = patches([0, 1], [0, 1])
    seen = coverage(*train, *patches([0, 1], [0, 1], seed=1))
    new = coverage(*train, *patches([0, 1], [1, 0], seed=2))
    assert seen.gap.max() < 0.5
    assert new.gap.min() > 100
    # each half on its own has been seen, so the separate score stays small
    assert new.separate.max() < 0.1 and new.joint.min() > 5


def test_joint_distance_is_never_below_separate():
    gen = torch.Generator().manual_seed(0)
    train_rgb, train_geo = (
        torch.randn(3, 4, 4, 6, generator=gen),
        torch.randn(3, 4, 4, 5, generator=gen),
    )
    rgb, geo = torch.randn(2, 4, 4, 6, generator=gen), torch.randn(2, 4, 4, 5, generator=gen)
    c = coverage(train_rgb, train_geo, rgb, geo, geo_weight=0.3)
    assert (c.joint >= c.separate - 1e-5).all()
    assert (c.gap >= -1e-5).all()


def test_weight_scales_only_the_geometry_half():
    train = patches([0], [0])
    query = patches([0], [1])
    full = coverage(*train, *query, geo_weight=1.0)
    small = coverage(*train, *query, geo_weight=0.1)
    np.testing.assert_allclose(small.separate, full.separate * 0.1, rtol=0.05)
