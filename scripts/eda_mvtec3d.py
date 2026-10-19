"""Per-sample statistics and overview figures for MVTec 3D-AD.

usage: uv run python scripts/eda_mvtec3d.py data/raw/mvtec3d

Writes results/eda/mvtec3d_samples.csv and three figures in docs/figures/.
Takes about two minutes, mostly the plane fit on every scan.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from factorylens.data import mvtec3d, preprocess  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
MASK_EDGE = "#eb6834"
DEPTH = LinearSegmentedColormap.from_list("depth", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": INK_2,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": INK_2,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def sample_stats(s: mvtec3d.Sample) -> dict:
    xyz = mvtec3d.load_xyz(s.xyz_path)
    valid = mvtec3d.valid_points(xyz)
    _, fg, plane = preprocess.remove_background(xyz)
    height = np.abs(plane.distance(xyz[fg])) if fg.any() else np.array([0.0])
    row = {
        "category": s.category,
        "split": s.split,
        "defect": s.defect,
        "stem": s.rgb_path.stem,
        "height_px": xyz.shape[0],
        "width_px": xyz.shape[1],
        "valid_frac": valid.mean(),
        "foreground_frac": fg.mean(),
        "object_height_mm": 1000 * np.percentile(height, 95),
    }
    if s.gt_path is not None and s.is_anomalous:
        mask = mvtec3d.load_mask(s.gt_path)
        n = mask.sum()
        row["defect_frac"] = n / mask.size
        row["defect_no_point_frac"] = (mask & ~valid).sum() / n
        row["defect_on_table_frac"] = (mask & valid & ~fg).sum() / n
    return row


def collect(root: str) -> pd.DataFrame:
    rows = []
    for category in mvtec3d.CATEGORIES:
        for split in mvtec3d.SPLITS:
            rows += [sample_stats(s) for s in mvtec3d.list_samples(root, category, split)]
        print(f"  {category}")
    return pd.DataFrame(rows)


def sample_grid(root: str, out: Path) -> None:
    """One defective test sample per category: RGB, depth of the object, mask outline."""
    fig, axes = plt.subplots(len(mvtec3d.CATEGORIES), 3, figsize=(6.0, 19.0))
    for row, category in zip(axes, mvtec3d.CATEGORIES, strict=True):
        s = next(x for x in mvtec3d.list_samples(root, category, "test") if x.is_anomalous)
        rgb = mvtec3d.load_rgb(s.rgb_path)
        xyz = mvtec3d.load_xyz(s.xyz_path)
        _, fg, plane = preprocess.remove_background(xyz)
        depth = np.where(fg, np.abs(plane.distance(xyz)) * 1000, np.nan)
        mask = mvtec3d.load_mask(s.gt_path)

        row[0].imshow(rgb)
        row[1].imshow(np.ones(fg.shape), cmap="gray", vmin=0, vmax=1.15)
        im = row[1].imshow(depth, cmap=DEPTH)
        row[2].imshow(rgb)
        row[2].contour(mask, levels=[0.5], colors=MASK_EDGE, linewidths=1.2)
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        row[0].set_ylabel(f"{category}\n({s.defect})", rotation=0, ha="right", va="center")
        cbar = fig.colorbar(im, ax=row[1], fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=7, colors=MUTED)
        cbar.outline.set_visible(False)
    for ax, title in zip(axes[0], ["RGB", "height above table (mm)", "defect mask"], strict=True):
        ax.set_title(title, color=INK_2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def defect_sizes(df: pd.DataFrame, out: Path) -> None:
    test = df[df["defect_frac"].notna()]
    order = test.groupby("category")["defect_frac"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    rng = np.random.default_rng(0)
    for i, category in enumerate(order):
        vals = 100 * test.loc[test["category"] == category, "defect_frac"].to_numpy()
        ax.scatter(vals, i + rng.uniform(-0.18, 0.18, len(vals)), s=9, color=BLUE, alpha=0.55, lw=0)
        median = np.median(vals)
        ax.plot([median, median], [i - 0.32, i + 0.32], color=INK, lw=2, solid_capstyle="round")
    ax.set_xscale("log")
    ax.set_yticks(range(len(order)), order)
    ax.set_xlabel("defect area, % of image (log scale, bar = median)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def defects_off_object(df: pd.DataFrame, out: Path) -> None:
    """How much of each category's defect area has no object geometry under it."""
    test = df[df["defect_frac"].notna()]
    share = test.groupby("category")[["defect_on_table_frac", "defect_no_point_frac"]].mean() * 100
    share["total"] = share.sum(axis=1)
    share = share.sort_values("total")
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    y = np.arange(len(share))
    ax.barh(y, share["total"], color=BLUE, height=0.6)
    for yi, v in zip(y, share["total"], strict=True):
        ax.text(v + 0.3, yi, f"{v:.1f}%", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(y, share.index)
    ax.set_xlabel("defect pixels with no 3D point or on the table (%, mean per sample)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--csv", default="results/eda/mvtec3d_samples.csv")
    parser.add_argument("--figures", default="docs/figures")
    args = parser.parse_args()

    figures = Path(args.figures)
    figures.mkdir(parents=True, exist_ok=True)
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)

    print("collecting per-sample stats")
    df = collect(args.root)
    df.to_csv(args.csv, index=False, float_format="%.5g")
    defect_sizes(df, figures / "defect_sizes.png")
    defects_off_object(df, figures / "defects_off_object.png")
    sample_grid(args.root, figures / "samples.png")
    print(f"wrote {args.csv} ({len(df)} rows) and figures in {figures}")


if __name__ == "__main__":
    main()
