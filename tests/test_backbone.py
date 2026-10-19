import numpy as np
import pytest
import torch

from factorylens.models.backbone import Backbone, to_batch

# pretrained=False everywhere: same shapes, no weight download in CI


def test_feature_shapes_and_strides():
    net = Backbone("resnet18", layers=("layer1", "layer2", "layer3"), pretrained=False)
    feats = net(torch.rand(2, 3, 64, 64))
    assert {k: tuple(v.shape) for k, v in feats.items()} == {
        "layer1": (2, 64, 16, 16),
        "layer2": (2, 128, 8, 8),
        "layer3": (2, 256, 4, 4),
    }


def test_unused_deeper_blocks_are_not_kept():
    net = Backbone("resnet18", layers=("layer2",), pretrained=False)
    assert list(net.blocks) == ["layer1", "layer2"]


def test_backbone_is_frozen_and_in_eval_mode():
    net = Backbone("resnet18", pretrained=False)
    assert not net.training
    assert not any(p.requires_grad for p in net.parameters())
    x = torch.rand(1, 3, 32, 32)
    assert torch.equal(net(x)["layer3"], net(x)["layer3"])  # no dropout / batchnorm drift


def test_rejects_unknown_arch_and_layer():
    with pytest.raises(ValueError, match="unknown arch"):
        Backbone("vgg16", pretrained=False)
    with pytest.raises(ValueError, match="unknown layers"):
        Backbone("resnet18", layers=("layer9",), pretrained=False)


def test_to_batch():
    imgs = [np.full((8, 10, 3), 255, np.uint8), np.zeros((8, 10, 3), np.uint8)]
    batch = to_batch(imgs)
    assert batch.shape == (2, 3, 8, 10) and batch.dtype == torch.float32
    assert batch[0].min() == 1.0 and batch[1].max() == 0.0
