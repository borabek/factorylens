import numpy as np
import torch

from factorylens.models.memory_bank import BankConfig, MemoryBank

NO_BLUR = BankConfig(blur_sigma=0)


def grid_features(n, h=4, w=4, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, h, w, d, generator=g)


def test_scores_and_maps_have_the_right_shape():
    bank = MemoryBank(NO_BLUR).fit(grid_features(5))
    scores, maps = bank.score(grid_features(2, seed=1), size=(16, 20))
    assert scores.shape == (2,) and maps.shape == (2, 16, 20)


def test_image_score_is_the_largest_patch_distance():
    train = grid_features(3)
    bank = MemoryBank(NO_BLUR).fit(train)
    test = train[:1].clone()
    test[0, 2, 1] += 10  # one patch moved far away
    scores, maps = bank.score(test, size=(4, 4))
    assert scores[0] == np.float32(maps[0].max())
    assert np.unravel_index(maps[0].argmax(), (4, 4)) == (2, 1)


def test_reweighting_shrinks_the_score_but_keeps_the_ranking():
    train = grid_features(6)
    test = grid_features(4, seed=3)
    test[1, 0, 0] += 6
    plain, _ = MemoryBank(NO_BLUR).fit(train).score(test, size=(4, 4))
    weighted, _ = MemoryBank(BankConfig(blur_sigma=0, reweight=True)).fit(train).score(test, (4, 4))
    assert (weighted <= plain).all() and (weighted > 0).all()
    assert weighted.argmax() == plain.argmax() == 1


def test_coreset_and_keep_all():
    feats = grid_features(10)  # 160 patches
    assert len(MemoryBank(BankConfig(keep_all_below=1000)).fit(feats).bank) == 160
    small = MemoryBank(BankConfig(keep_all_below=100, coreset_ratio=0.25)).fit(feats)
    assert len(small.bank) == 40
