"""What each backbone layer picks up: nearest-normal-patch distance maps per layer.

For a few defective test images, every patch feature is compared with the
patch features of 10 normal training images (the k=10, seed 0 subset) and the
distance to the closest one is shown. This is the core of the PatchCore-style
RGB detector, done one layer at a time so the layers can be compared.

What it shows (first run, 2026-10-19): with raw single-layer features and only
10 normals, the whole object lights up more than the defect, because 10 images
do not cover the variation in pose and texture. Only the deeper layers start to
separate some defects (bagel crack, cable_gland bend). That is the motivation
for averaging over a 3x3 patch neighbourhood, combining layer2 and layer3, and
using the full training set, which is what the detector does. The numbers for
each layer choice come from the layer ablation, not from this figure.

usage: uv run python scripts/layer_maps.py data/raw/mvtec3d
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from factorylens.data import mvtec3d, subsets  # noqa: E402
from factorylens.data.preprocess import resize_mask, resize_rgb  # noqa: E402
from factorylens.models.backbone import Backbone, pick_device, to_batch  # noqa: E402

SIZE = 224
LAYERS = ("layer1", "layer2", "layer3")
EXAMPLES = [("bagel", "crack"), ("cable_gland", "bent"), ("foam", "color"), ("potato", "hole")]
HEAT = LinearSegmentedColormap.from_list("heat", ["#fcfcfb", "#9ec5f4", "#2a78d6", "#0d366b"])
MASK_EDGE = "#eb6834"


def patch_features(net: Backbone, images: list[np.ndarray], device: torch.device) -> dict:
    feats = net(to_batch(images).to(device))
    return {k: v.permute(0, 2, 3, 1) for k, v in feats.items()}  # (B, h, w, C)


def nearest_distance(query: torch.Tensor, bank: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
    """(h, w, C) query, (N, C) bank -> (h, w) distance to the nearest bank row."""
    h, w, c = query.shape
    flat = query.reshape(-1, c)
    out = torch.cat(
        [
            torch.cdist(flat[i : i + chunk], bank).min(dim=1).values
            for i in range(0, len(flat), chunk)
        ]
    )
    return out.reshape(h, w)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--out", default="docs/figures/layer_maps.png")
    args = parser.parse_args()

    device = pick_device()
    net = Backbone("wide_resnet50_2", layers=LAYERS).to(device)
    kshot = subsets.load("configs/kshot.json")

    fig, axes = plt.subplots(len(EXAMPLES), 1 + len(LAYERS), figsize=(7.2, 7.4))
    for row, (category, defect) in zip(axes, EXAMPLES, strict=True):
        train = subsets.select(mvtec3d.list_samples(args.root, category, "train"), kshot, k=10)
        normals = [resize_rgb(mvtec3d.load_rgb(s.rgb_path), SIZE) for s in train]
        test = next(
            s for s in mvtec3d.list_samples(args.root, category, "test") if s.defect == defect
        )
        image = resize_rgb(mvtec3d.load_rgb(test.rgb_path), SIZE)
        mask = resize_mask(mvtec3d.load_mask(test.gt_path), SIZE)

        bank = patch_features(net, normals, device)
        query = patch_features(net, [image], device)

        row[0].imshow(image)
        row[0].contour(mask, levels=[0.5], colors=MASK_EDGE, linewidths=1.0)
        row[0].set_ylabel(f"{category}\n{defect}", rotation=0, ha="right", va="center", fontsize=8)
        for ax, layer in zip(row[1:], LAYERS, strict=True):
            c = bank[layer].shape[-1]
            dist = nearest_distance(query[layer][0], bank[layer].reshape(-1, c))
            dist = F.interpolate(dist[None, None], size=SIZE, mode="bilinear")[0, 0].cpu().numpy()
            ax.imshow(dist, cmap=HEAT, vmin=np.percentile(dist, 1), vmax=dist.max())
            ax.contour(mask, levels=[0.5], colors=MASK_EDGE, linewidths=0.8)
            h, w = query[layer].shape[1:3]
            print(f"{category:12s} {layer}: feature map {h}x{w}, {c} channels")
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    titles = ["image + defect", "layer1 (56×56)", "layer2 (28×28)", "layer3 (14×14)"]
    for ax, title in zip(axes[0], titles, strict=True):
        ax.set_title(title, fontsize=9, color="#52514e")
    fig.text(
        0.5,
        0.01,
        "distance to the nearest normal patch (10 training images), each map scaled on its own",
        ha="center",
        fontsize=8,
        color="#52514e",
    )
    fig.patch.set_facecolor("#fcfcfb")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(args.out, dpi=130)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
