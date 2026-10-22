"""Figures for the Future Factories analog data.

1. docs/figures/ff_timeline.png: every cycle on a time axis, coloured by label,
   with the period the camera images cover. Shows the confound in one picture.
2. docs/figures/ff_gripper.png: gripper load through one cycle for each robot,
   normal vs NoNose (day 1 only), to see where a missing part shows up.

usage: uv run python scripts/ff/figures.py "data/raw/ff_analog/Analog Dataset"
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from factorylens.data import ff  # noqa: E402
from factorylens.models import sensor  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from confound_check import IMAGE_WINDOW  # noqa: E402

SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
LABEL_COLORS = {
    "normal": "#2a78d6",
    "NoNose": "#eb6834",
    "NoNose,NoBody2": "#1baf7a",
    "NoNose,NoBody2,NoBody1": "#eda100",
}
BLOCK_TEXT = {
    "normal": "normal",
    "NoNose": "no nose",
    "NoNose,NoBody2": "no nose,\nbody 2",
    "NoNose,NoBody2,NoBody1": "no nose,\nbodies 1+2",
}
SHORT = {
    "normal": "normal",
    "NoNose": "no nose",
    "NoNose,NoBody2": "no nose, body 2",
    "NoNose,NoBody2,NoBody1": "no nose, bodies 1+2",
}
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


def timeline(cycles: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 3.0))
    for _, c in cycles.iterrows():
        ax.fill_between([c["start"], c["end"]], 0.35, 0.85, color=LABEL_COLORS[c["label"]], lw=0)

    # direct labels on each block, so identity never depends on colour alone
    for _, b in ff.label_blocks(cycles).iterrows():
        mid = b["start"] + (b["end"] - b["start"]) / 2
        text = f"{BLOCK_TEXT[b['label']]}\n{b['cycles']} cycles"
        ax.text(mid, 0.9, text, ha="center", va="bottom", fontsize=7.5, color=INK_2)

    lo, hi = IMAGE_WINDOW
    arrow = {"arrowstyle": "<->", "color": INK, "lw": 1.2}
    ax.annotate("", xy=(lo, 0.22), xytext=(hi, 0.22), arrowprops=arrow)
    ax.text(
        lo + (hi - lo) / 2,
        0.06,
        "period covered by camera images",
        ha="center",
        va="bottom",
        fontsize=8,
        color=INK,
    )
    ax.plot([ff.DAY_BOUNDARY] * 2, [0.3, 0.9], color=MUTED, lw=0.8, ls=(0, (3, 3)))

    ax.set_ylim(0, 1.35)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M", tz="UTC"))
    ax.set_xlabel("time (UTC)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in LABEL_COLORS.values()]
    ax.legend(
        handles,
        SHORT.values(),
        ncol=4,
        frameon=False,
        fontsize=7.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.32),
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def gripper(df: pd.DataFrame, cycles: pd.DataFrame, out: Path) -> None:
    day1 = cycles[cycles["day"] == 1]
    groups = {"normal": day1[day1["label"] == "normal"], "NoNose": day1[day1["label"] == "NoNose"]}
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 6.4), sharex=True)
    for ax, robot in zip(axes, ("R01", "R02", "R03", "R04"), strict=True):
        column = f"I_{robot}_Gripper_Load"
        for label, part in groups.items():
            prof = sensor.cycle_profiles(df, part, (column,))[..., 0]
            med = np.nanmedian(prof, axis=0)
            lo, hi = np.nanpercentile(prof, [10, 90], axis=0)
            x = np.arange(len(med))
            ax.fill_between(x, lo, hi, color=LABEL_COLORS[label], alpha=0.18, lw=0)
            ax.plot(
                x,
                med,
                color=LABEL_COLORS[label],
                lw=1.6,
                label=f"{SHORT[label]} ({len(part)} cycles)",
            )
        ax.set_ylabel(robot, rotation=0, ha="right", va="center")
        if robot == "R03":
            ax.annotate(
                "pulse missing\nwithout the nose cone",
                xy=(94, 5300),
                xytext=(140, 5200),
                fontsize=7.5,
                color=INK_2,
                va="center",
                arrowprops={"arrowstyle": "->", "color": INK_2, "lw": 0.8},
            )
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.tick_params(axis="y", labelsize=7)
    axes[0].legend(frameon=False, fontsize=8, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.02))
    axes[-1].set_xlabel("seconds since cycle start (line = median, band = 10-90th percentile)")
    fig.text(
        0.01, 0.5, "gripper load (raw units)", rotation=90, va="center", fontsize=8, color=INK_2
    )
    fig.tight_layout(rect=(0.03, 0, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--figures", default="docs/figures")
    args = parser.parse_args()
    figures = Path(args.figures)
    figures.mkdir(parents=True, exist_ok=True)

    df = ff.load_analog(args.root)
    cycles = ff.segment_cycles(df)
    timeline(cycles, figures / "ff_timeline.png")
    gripper(df, cycles, figures / "ff_gripper.png")
    print("wrote", figures / "ff_timeline.png", "and", figures / "ff_gripper.png")


if __name__ == "__main__":
    main()
