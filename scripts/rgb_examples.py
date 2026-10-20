"""Hits, misses and false alarms of a detector, from its saved test scores.

For each category: the defective image it scores highest (a clear hit), the
defective image it scores lowest (its worst miss), and the good image it
scores highest (its worst false alarm). Heatmaps share one colour scale per
category, so a missed defect really does look faint.

usage: uv run python scripts/rgb_examples.py --scores outputs/scores/rgb
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from factorylens.data import cache  # noqa: E402

HEAT = LinearSegmentedColormap.from_list("heat", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
MASK_EDGE = "#eb6834"
CATEGORIES = ("bagel", "foam", "potato", "tire")
ROWS = ("best hit", "worst miss", "worst false alarm")


def pick(scores: np.ndarray, labels: np.ndarray) -> list[int]:
    bad, good = np.flatnonzero(labels), np.flatnonzero(~labels)
    return [
        int(bad[scores[bad].argmax()]),
        int(bad[scores[bad].argmin()]),
        int(good[scores[good].argmax()]),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", default="outputs/scores/rgb")
    parser.add_argument("--root", default="data/raw/mvtec3d")
    parser.add_argument("--cache", default="data/processed")
    parser.add_argument("--out", default="docs/figures/rgb_examples.png")
    args = parser.parse_args()

    fig, axes = plt.subplots(len(ROWS), len(CATEGORIES), figsize=(7.6, 6.2))
    fig.patch.set_facecolor("#fcfcfb")
    for col, category in enumerate(CATEGORIES):
        [path] = sorted(Path(args.scores).glob(f"{category}_kall_s0_*.npz"))
        saved = np.load(path)
        test = cache.get(args.root, args.cache, category, "test")
        maps = saved["maps"].astype(np.float32)
        lo, hi = np.percentile(maps, 1), np.percentile(maps, 99.9)
        for row, idx in enumerate(pick(saved["scores"], saved["labels"])):
            ax = axes[row, col]
            gray = test.rgb[idx].mean(axis=-1)
            heat = np.clip((maps[idx] - lo) / (hi - lo), 0, 1)
            ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
            # only the upper part of the scale is drawn, so the photo stays readable
            alpha = np.clip((heat - 0.35) / 0.65, 0, 1) * 0.75
            ax.imshow(heat, cmap=HEAT, alpha=alpha, vmin=0, vmax=1)
            if test.mask[idx].any():
                ax.contour(test.mask[idx], levels=[0.5], colors=MASK_EDGE, linewidths=1.0)
            ax.set_xlabel(
                f"{test.defect[idx]}, score {saved['scores'][idx]:.2f}",
                fontsize=7.5,
                color="#52514e",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(category, fontsize=9, color="#0b0b0b")
            if col == 0:
                ax.set_ylabel(ROWS[row], fontsize=8.5, color="#52514e")
    fig.text(
        0.5,
        0.005,
        "orange outline = annotated defect; blue = anomaly score, one scale per category",
        ha="center",
        fontsize=7.5,
        color="#52514e",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(args.out, dpi=130)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
