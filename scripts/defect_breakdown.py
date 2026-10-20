"""Which defect types a detector catches, from its saved test scores.

For each category and defect type, the detector is scored on the good test
images plus the images of that one type: image AUROC (does it flag the image)
and AUPRO (does it find the defective region).

usage: uv run python scripts/defect_breakdown.py --scores outputs/scores/rgb --name rgb
Writes results/analysis/<name>_defect_types.csv and docs/figures/<name>_defect_types.png.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from factorylens.data import cache, mvtec3d  # noqa: E402
from factorylens.eval import metrics  # noqa: E402

SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])


def breakdown(scores_dir: Path, root: str, cache_dir: str) -> pd.DataFrame:
    rows = []
    for category in mvtec3d.CATEGORIES:
        [path] = sorted(scores_dir.glob(f"{category}_kall_s0_*.npz"))
        saved = np.load(path)
        test = cache.get(root, cache_dir, category, "test")
        if saved["stem"].tolist() != test.stem or saved["defect"].tolist() != test.defect:
            raise ValueError(f"{path.name} does not line up with the cached test split")
        maps = saved["maps"].astype(np.float32)
        defect = np.array(test.defect)
        good = defect == "good"
        for d in sorted(set(test.defect) - {"good"}):
            keep = good | (defect == d)
            rows.append(
                {
                    "category": category,
                    "defect": d,
                    "n": int((defect == d).sum()),
                    "image_auroc": metrics.image_auroc(~good[keep], saved["scores"][keep]),
                    "aupro": metrics.aupro(test.mask[keep], maps[keep]),
                }
            )
    return pd.DataFrame(rows)


def heatmap(df: pd.DataFrame, column: str, title: str, out: Path) -> None:
    table = df.pivot(index="category", columns="defect", values=column)
    order = table.notna().sum().sort_values(ascending=False).index
    table = table[order]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    fig.patch.set_facecolor("#fcfcfb")
    im = ax.imshow(table.to_numpy(dtype=float), cmap=SEQ, vmin=0.5, vmax=1.0, aspect="auto")
    for (i, j), v in np.ndenumerate(table.to_numpy(dtype=float)):
        if np.isfinite(v):
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="#fcfcfb" if v > 0.8 else "#0b0b0b",
            )
    ax.set_xticks(range(len(table.columns)), table.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(table.index)), table.index)
    ax.tick_params(length=0, labelsize=8.5, colors="#52514e")
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=7.5, colors="#898781")
    ax.set_title(title, fontsize=9, color="#52514e")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", default="outputs/scores/rgb")
    parser.add_argument("--name", default="rgb")
    parser.add_argument("--root", default="data/raw/mvtec3d")
    parser.add_argument("--cache", default="data/processed")
    args = parser.parse_args()

    df = breakdown(Path(args.scores), args.root, args.cache)
    out = Path("results/analysis")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{args.name}_defect_types.csv", index=False, float_format="%.4f")
    heatmap(
        df,
        "image_auroc",
        f"{args.name}: image AUROC, good vs one defect type (blank = type not in category)",
        Path("docs/figures") / f"{args.name}_defect_types.png",
    )
    print(df.groupby("defect")[["image_auroc", "aupro"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
