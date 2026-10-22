"""Summarise the data-efficiency sweep (`factorylens sweep`).

1. results/analysis/sweep_paired.csv: on the full training set, each fusion
   model against each single sensor, per category, with a paired bootstrap
   over the test images (both models scored on the same resample).
2. results/analysis/sweep_kshot.csv: mean over categories for every model and
   k, with the spread over the five k-shot seeds.
3. docs/figures/data_efficiency.png: the same as a figure.

usage: uv run python scripts/sweep_report.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from factorylens import experiments  # noqa: E402
from factorylens.eval.metrics import paired_bootstrap  # noqa: E402

MODELS = ("rgb", "fpfh", "early", "late", "rgb_aug")
PAIRS = [("early", "rgb"), ("early", "fpfh"), ("late", "rgb"), ("late", "fpfh"), ("early", "late")]
NAMES = {
    "rgb": "RGB",
    "fpfh": "3D (FPFH)",
    "early": "early fusion",
    "late": "late fusion",
    "rgb_aug": "RGB + flips",
}
COLORS = {
    "rgb": "#2a78d6",
    "fpfh": "#eb6834",
    "early": "#1baf7a",
    "late": "#7a5bd6",
    "rgb_aug": "#2a78d6",
}
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"


def load(results: Path) -> pd.DataFrame:
    frames = []
    for model in MODELS:
        if (results / model).is_dir():
            frames.append(experiments.collect(results / model).assign(sweep=model))
    runs = pd.concat(frames, ignore_index=True)
    runs["k"] = runs["k"].astype("Int64")
    return runs


def paired(runs: pd.DataFrame, scores: Path) -> pd.DataFrame:
    full = runs[runs["k"].isna()]
    rows = []
    for category in sorted(full["category"].unique()):
        files = full[full["category"] == category].set_index("sweep")["file"]
        loaded = {
            m: np.load(scores / m / Path(f).with_suffix(".npz").name) for m, f in files.items()
        }
        labels = loaded["rgb"]["labels"]
        for a, b in PAIRS:
            if a not in loaded or b not in loaded:
                continue
            assert (loaded[a]["stem"] == loaded[b]["stem"]).all()
            test = paired_bootstrap(labels, loaded[a]["scores"], loaded[b]["scores"])
            rows.append({"category": category, "model": a, "versus": b, **test})
    return pd.DataFrame(rows)


def kshot(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean over categories per seed first, then mean and sd over seeds."""
    runs = runs.assign(k=runs["k"].astype("float").fillna(np.inf))
    per_seed = runs.groupby(["sweep", "k", "seed"])[["image_auroc", "aupro"]].agg(["mean", "size"])
    rows = []
    for (model, k), part in per_seed.groupby(level=[0, 1]):
        rows.append(
            {
                "model": model,
                "k": k,
                "seeds": len(part),
                "categories": int(part[("image_auroc", "size")].min()),
                "image_auroc": part[("image_auroc", "mean")].mean(),
                "image_auroc_sd": part[("image_auroc", "mean")].std(ddof=1),
                "aupro": part[("aupro", "mean")].mean(),
                "aupro_sd": part[("aupro", "mean")].std(ddof=1),
            }
        )
    return pd.DataFrame(rows)


def label_ends(ax: plt.Axes, table: pd.DataFrame, metric: str, n_train: int) -> None:
    """Name each line at its right end, pushing labels apart where they would overlap."""
    ends = []
    for model in MODELS:
        part = table[table["model"] == model].sort_values("k")
        if not part.empty:
            ends.append([part[metric].iloc[-1], model, part["k"].replace(np.inf, n_train).iloc[-1]])
    ends.sort()
    lo, hi = ax.get_ylim()
    gap = 0.045 * (hi - lo)
    for prev, cur in zip(ends, ends[1:], strict=False):
        cur[0] = max(cur[0], prev[0] + gap)
    for y, model, x in ends:
        ax.annotate(
            NAMES[model],
            (x, y),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=7.5,
            color=COLORS[model],
        )


def figure(table: pd.DataFrame, n_train: int, out: Path) -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    fig.patch.set_facecolor(SURFACE)
    for ax, metric, title in zip(
        axes, ["image_auroc", "aupro"], ["image AUROC", "AUPRO (FPR ≤ 0.3)"], strict=True
    ):
        ax.set_facecolor(SURFACE)
        for model in MODELS:
            part = table[table["model"] == model].sort_values("k")
            if part.empty:
                continue
            x = part["k"].replace(np.inf, n_train).to_numpy()
            y, sd = part[metric].to_numpy(), part[f"{metric}_sd"].fillna(0).to_numpy()
            dashed = model == "rgb_aug"
            ax.plot(
                x,
                y,
                marker="o",
                ms=3.5,
                lw=1.4,
                color=COLORS[model],
                ls=(0, (3, 2)) if dashed else "-",
                label=NAMES[model],
            )
            ax.fill_between(x, y - sd, y + sd, color=COLORS[model], alpha=0.12, lw=0)
        label_ends(ax, table, metric, n_train)
        ax.set_xscale("log")
        ticks = [1, 5, 10, 50, n_train]
        ax.set_xticks(ticks, [*map(str, ticks[:-1]), f"all\n(~{n_train})"])
        ax.minorticks_off()
        ax.set_xlim(0.8, n_train * 2.6)
        ax.set_xlabel("normal training images per category")
        ax.set_title(f"{title}, mean of 10 categories", loc="left", fontsize=9, color=INK)
        ax.grid(axis="y", color="#e1e0d9", lw=0.6)
        ax.tick_params(colors=INK_2)
    axes[0].axhline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 3)))
    fig.text(
        0.01,
        0.01,
        "band = ± 1 sd over five k-shot draws; 'all' is one run",
        fontsize=7.5,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=Path("results/sweep"))
    parser.add_argument("--scores", type=Path, default=Path("outputs/scores/sweep"))
    parser.add_argument("--out", type=Path, default=Path("results/analysis"))
    parser.add_argument("--figure", type=Path, default=Path("docs/figures/data_efficiency.png"))
    args = parser.parse_args()

    runs = load(args.results)
    args.out.mkdir(parents=True, exist_ok=True)
    tests = paired(runs, args.scores)
    tests.round(4).to_csv(args.out / "sweep_paired.csv", index=False)
    table = kshot(runs)
    table.round(4).to_csv(args.out / "sweep_kshot.csv", index=False)
    n_train = int(runs.loc[runs["k"].isna(), "n_train"].median())
    figure(table, n_train, args.figure)

    wide = table.pivot(index="k", columns="model", values="image_auroc")
    print(wide.round(3).to_string())
    wins = tests.assign(win=tests["ci_low"] > 0, loss=tests["ci_high"] < 0)
    print(
        wins.groupby(["model", "versus"])[["diff", "win", "loss"]].agg(
            {"diff": "mean", "win": "sum", "loss": "sum"}
        )
    )


if __name__ == "__main__":
    main()
