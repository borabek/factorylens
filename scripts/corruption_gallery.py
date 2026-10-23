"""Figure: every FactoryLens-C corruption at severities 1, 3 and 5 on one test sample.

RGB corruptions are shown on the image. 3D corruptions are shown as a shaded
depth map (light from the top left), which makes noise, steps and holes
visible; misregistration is drawn as the object outline from the RGB
foreground over the shifted depth.

usage: uv run python scripts/corruption_gallery.py data/raw/mvtec3d --cache data/processed
Writes docs/figures/corruptions.png.
"""

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from factorylens import corruptions  # noqa: E402
from factorylens.data import cache  # noqa: E402
from factorylens.data.mvtec3d import valid_points  # noqa: E402

SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
LEVELS = (1, 3, 5)


def shaded(xyz: np.ndarray) -> np.ndarray:
    """Grey hillshade of the depth map, white where there is no point."""
    z = xyz[..., 2].astype(float)
    valid = valid_points(xyz)
    z = np.where(valid, z, np.nan)
    gy, gx = np.gradient(z)
    light = np.array([-1.0, -1.0, 1.0]) / np.sqrt(3)
    scale = 2e-4  # metres per pixel, roughly
    n = np.stack([-gx / scale, -gy / scale, np.ones_like(z)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    shade = np.clip(n @ light, 0, 1)
    return np.where(valid, 0.15 + 0.8 * np.nan_to_num(shade), 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root")
    parser.add_argument("--cache", default="data/processed")
    parser.add_argument("--category", default="cookie")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("docs/figures/corruptions.png"))
    args = parser.parse_args()

    test = cache.get(args.root, args.cache, args.category, "test")
    i = slice(args.index, args.index + 1)
    one = replace(
        test,
        rgb=test.rgb[i],
        xyz=test.xyz[i],
        foreground=test.foreground[i],
        mask=test.mask[i],
        defect=test.defect[i],
        stem=test.stem[i],
    )
    names = [n for n, c in corruptions.CORRUPTIONS.items() if c.severities > 1]
    fig, axes = plt.subplots(len(names), 1 + len(LEVELS), figsize=(5.6, 1.5 * len(names)))
    fig.patch.set_facecolor(SURFACE)
    for row, name in zip(axes, names, strict=True):
        is_rgb = corruptions.CORRUPTIONS[name].sensor == "rgb"
        clean = one.rgb[0] if is_rgb else shaded(one.xyz[0])
        panels = [clean]
        for level in LEVELS:
            c = corruptions.corrupt(one, name, level)
            panels.append(c.rgb[0] if is_rgb else shaded(c.xyz[0]))
        for j, (ax, img) in enumerate(zip(row, panels, strict=True)):
            ax.imshow(img, cmap=None if is_rgb else "gray", vmin=0, vmax=1)
            if name == "misregister" and j > 0:
                ax.contour(one.foreground[0], levels=[0.5], colors="#eb6834", linewidths=0.6)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        row[0].set_ylabel(
            name.replace("_", " "), rotation=0, ha="right", va="center", fontsize=8, color=INK
        )
    for ax, title in zip(axes[0], ["clean", *[f"severity {s}" for s in LEVELS]], strict=True):
        ax.set_title(title, fontsize=8, color=INK_2)
    fig.tight_layout(h_pad=0.2, w_pad=0.2)
    fig.savefig(args.out, dpi=110, facecolor=SURFACE)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
