import numpy as np
import pytest
import torch

from factorylens.models.patchcore import PatchCore, PatchCoreConfig, greedy_coreset

# random-init resnet18 on CPU: no weight download, and small enough for CI
SMALL = PatchCoreConfig(arch="resnet18", embed_dim=64, blur_sigma=1.0)


def textures(n, size=64, seed=0):
    """Smooth noise images, all drawn from the same distribution."""
    rng = np.random.default_rng(seed)
    base = rng.normal(128, 20, (n, size // 8, size // 8, 3))
    up = np.kron(base, np.ones((1, 8, 8, 1)))
    return np.clip(up + rng.normal(0, 5, up.shape), 0, 255).astype(np.uint8)


def test_greedy_coreset_picks_the_outlier_and_is_deterministic():
    x = torch.randn(500, 16)
    x[123] += 50  # far from everything else
    idx = greedy_coreset(x, 10, projection_dim=0, seed=1)
    assert len(set(idx.tolist())) == 10
    assert 123 in idx[:2].tolist()  # first or second pick, whatever the start
    assert torch.equal(idx, greedy_coreset(x, 10, projection_dim=0, seed=1))


def test_greedy_coreset_returns_everything_when_m_is_large():
    x = torch.randn(7, 4)
    assert greedy_coreset(x, 10).tolist() == list(range(7))


def test_greedy_coreset_covers_better_than_random():
    torch.manual_seed(0)
    x = torch.cat([torch.randn(900, 8), torch.randn(100, 8) + 6])  # a small second cluster
    idx = greedy_coreset(x, 20, projection_dim=0)
    rand = torch.randperm(1000)[:20]
    cover = lambda sel: torch.cdist(x, x[sel]).min(dim=1).values.max()  # noqa: E731
    assert cover(idx) < cover(rand)


def test_fit_and_score_shapes():
    model = PatchCore(SMALL, device="cpu", pretrained=False).fit(textures(4))
    scores, maps = model.score(textures(3, seed=1))
    assert scores.shape == (3,) and maps.shape == (3, 64, 64)
    assert model.bank is not None and model.bank.shape[1] == 64


def test_training_images_are_their_own_nearest_neighbours():
    train = textures(3)
    model = PatchCore(SMALL, device="cpu", pretrained=False).fit(train)
    seen, _ = model.score(train)
    unseen, _ = model.score(textures(3, seed=5))
    # not exactly 0: cdist's matmul path leaves ~1e-3 of rounding error
    assert seen.max() < 0.02 * unseen.min()


def test_a_pasted_square_is_found():
    model = PatchCore(SMALL, device="cpu", pretrained=False).fit(textures(6))
    test = textures(2, seed=7)
    test[1, 40:56, 8:24] = (250, 20, 20)  # red square, bottom left
    scores, maps = model.score(test)
    assert scores[1] > scores[0]
    r, c = np.unravel_index(maps[1].argmax(), maps[1].shape)
    assert 32 <= r < 64 and c < 32


def test_coreset_kicks_in_above_the_threshold():
    cfg = PatchCoreConfig(arch="resnet18", embed_dim=32, keep_all_below=100, coreset_ratio=0.25)
    model = PatchCore(cfg, device="cpu", pretrained=False).fit(textures(4))  # 4 * 8*8 = 256 patches
    assert model.bank is not None and len(model.bank) == 64


def test_score_before_fit():
    with pytest.raises(RuntimeError):
        PatchCore(SMALL, device="cpu", pretrained=False).score(textures(1))


@pytest.mark.parametrize(("c", "d"), [(1536, 1024), (384, 64), (10, 10), (7, 3)])
def test_channel_pooling_matrix_matches_adaptive_avg_pool(c, d):
    from factorylens.models.patchcore import channel_pooling_matrix

    x = torch.randn(5, c)
    expected = torch.nn.functional.adaptive_avg_pool1d(x[:, None, :], d)[:, 0]
    torch.testing.assert_close(x @ channel_pooling_matrix(c, d), expected)


def test_fixed_grid_puts_every_layer_choice_on_the_same_patches():
    for layers in [("layer1",), ("layer3",), ("layer1", "layer2", "layer3")]:
        cfg = PatchCoreConfig(arch="resnet18", embed_dim=32, layers=layers, grid=4)
        emb = PatchCore(cfg, device="cpu", pretrained=False).embed(textures(2))
        assert emb.shape == (2, 4, 4, 32)


def test_default_grid_follows_the_first_layer():
    cfg = PatchCoreConfig(arch="resnet18", embed_dim=32, layers=("layer1", "layer3"))
    emb = PatchCore(cfg, device="cpu", pretrained=False).embed(textures(1))
    assert emb.shape[1:3] == (16, 16)  # layer1 of a 64 px image


def test_maps_keep_the_shape_of_non_square_inputs():
    model = PatchCore(SMALL, device="cpu", pretrained=False).fit(textures(2))
    wide = np.concatenate([textures(1, seed=3), textures(1, seed=4)], axis=2)  # 64 x 128
    _, maps = model.score(wide)
    assert maps.shape == (1, 64, 128)


def test_btf_style_options():
    from dataclasses import replace

    from factorylens.models.patchcore import BTF_RGB

    model = PatchCore(replace(BTF_RGB, arch="resnet18"), device="cpu", pretrained=False)
    emb = model.embed(textures(2))
    # no padding shrinks 8x8 to 6x6, the fixed grid stretches it to 28x28; all 128+256 channels kept
    assert emb.shape == (2, 28, 28, 384)
    scores, maps = model.fit(textures(3)).score(textures(2, seed=9))
    assert scores.shape == (2,) and maps.shape == (2, 64, 64)
