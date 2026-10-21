"""Which sensor finds which defect: RGB vs 3D per category and defect type.

Reads the per-defect-type tables written by defect_breakdown.py for the RGB
and FPFH detectors and draws the difference (3D minus RGB image AUROC) on a
diverging scale: orange where 3D is better, blue where RGB is better.

usage: uv run python scripts/compare_sensors.py
Writes results/analysis/rgb_vs_3d_defect_types.csv and
docs/figures/rgb_vs_3d_defect_types.png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# blue = RGB better, neutral grey midpoint, orange = 3D better
DIVERGING = LinearSegmentedColormap.from_list(
    "diverging", ["#1c5cab", "#86b6ef", "#f0efec", "#f2a27e", "#c2471a"]
)


def main() -> None:
    rgb = pd.read_csv("results/analysis/rgb_defect_types.csv")
    geo = pd.read_csv("results/analysis/fpfh_defect_types.csv")
    both = rgb.merge(geo, on=["category", "defect", "n"], suffixes=("_rgb", "_3d"))
    both["diff"] = both["image_auroc_3d"] - both["image_auroc_rgb"]
    both["better"] = np.where(both["diff"] > 0, "3d", "rgb")
    out = Path("results/analysis/rgb_vs_3d_defect_types.csv")
    both.to_csv(out, index=False, float_format="%.4f")

    table = both.pivot(index="category", columns="defect", values="diff")
    table = table[table.notna().sum().sort_values(ascending=False).index]
    values = table.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    fig.patch.set_facecolor("#fcfcfb")
    im = ax.imshow(values, cmap=DIVERGING, vmin=-0.5, vmax=0.5, aspect="auto")
    for (i, j), v in np.ndenumerate(values):
        if np.isfinite(v):
            ax.text(
                j,
                i,
                f"{v:+.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="#fcfcfb" if abs(v) > 0.3 else "#0b0b0b",
            )
    ax.set_xticks(range(len(table.columns)), table.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(table.index)), table.index)
    ax.tick_params(length=0, labelsize=8.5, colors="#52514e")
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=7.5, colors="#898781")
    cbar.set_label("3D minus RGB image AUROC", fontsize=8, color="#52514e")
    ax.set_title(
        "orange: 3D finds it better, blue: RGB finds it better", fontsize=9, color="#52514e"
    )
    fig.tight_layout()
    fig.savefig("docs/figures/rgb_vs_3d_defect_types.png", dpi=150)
    plt.close(fig)

    # how much of the variation in the difference each factor explains on its own
    total = ((both["diff"] - both["diff"].mean()) ** 2).sum()
    explained = {
        factor: 1
        - ((both["diff"] - both.groupby(factor)["diff"].transform("mean")) ** 2).sum() / total
        for factor in ("category", "defect")
    }
    pd.DataFrame([explained]).to_csv(
        "results/analysis/rgb_vs_3d_variance_explained.csv", index=False, float_format="%.4f"
    )
    print(f"3D better on {(both['diff'] > 0).sum()} of {len(both)} category/defect pairs")
    print("share of variance explained by category alone / defect type alone:", explained)
    print(both.groupby("defect")["diff"].agg(["mean", "count"]).round(3).sort_values("mean"))


if __name__ == "__main__":
    main()
