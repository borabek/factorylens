"""Alarm thresholds from the validation normals, and what they are worth.

Everything comes from the scores saved by `factorylens robustness` in its
last run (outputs/scores/robustness-extent, which holds every model), so
nothing is recomputed. For each category the validation normals (22-33
parts) are the calibration set.

1. clean: false alarm rate on the normal test parts and detection rate on
   the defective ones, for alpha in {0.05, 0.1, 0.2}, per detector;
2. calibration size: the same with the calibration set cut to n in {5, 10,
   15, 20} parts (200 random subsets each), against the Beta band theory
   gives for one calibration set;
3. broken sensor: false alarms and detections at alpha = 0.1 on every
   FactoryLens-C condition, thresholds still from the clean validation parts.

Fused scores are computed on the calibration parts the same way as on the
test parts. The z-scores of late fusion use the validation statistics, so
calibration and test are not strictly exchangeable there; the plain
detectors (RGB, FPFH) are the clean check.

usage: uv run python scripts/conformal_alarms.py
Writes results/analysis/conformal_*.csv and docs/figures/conformal.png.
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from factorylens import conformal, corruptions  # noqa: E402
from factorylens.data.mvtec3d import CATEGORIES  # noqa: E402

ALPHAS = (0.05, 0.1, 0.2)
SIZES = (5, 10, 15, 20)
DETECTORS = ("rgb", "fpfh", "late", "reliable_window", "reliable_extent")
NAMES = {
    "rgb": "RGB",
    "fpfh": "FPFH",
    "late": "late fusion",
    "reliable_window": "reliability fusion",
    "reliable_extent": "reliability + extent",
}
COLORS = {
    "rgb": "#2a78d6",
    "fpfh": "#eb6834",
    "late": "#7a5bd6",
    "reliable_window": "#0b0b0b",
    "reliable_extent": "#d6336c",
}
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
FILE = re.compile(
    r"(?P<category>.+?)_(?P<corruption>"
    + "|".join(["clean", *corruptions.CORRUPTIONS])
    + r")(?P<severity>\d)_s0\.npz"
)


def scores(d: dict) -> tuple[dict, dict]:
    """Calibration (validation) and test scores per detector, on the same scale."""
    rv, gv = d["rgb_val"], d["fpfh_val"]
    zr = lambda s: (s - rv.mean()) / rv.std()  # noqa: E731
    zg = lambda s: (s - gv.mean()) / gv.std()  # noqa: E731
    cal = {
        "rgb": rv,
        "fpfh": gv,
        "late": zr(rv) + zg(gv),
        # every validation part sits inside its own reference range, so both weights are 1
        "reliable_window": (zr(rv) + zg(gv)) / np.sqrt(2),
        "reliable_extent": (zr(rv) + zg(gv)) / np.sqrt(2),
    }
    test = {m: d[m] for m in DETECTORS}
    return cal, test


def rates(cal: np.ndarray, test: np.ndarray, labels: np.ndarray, alpha: float) -> tuple:
    alarm = test > conformal.threshold(cal, alpha)
    return alarm[~labels].mean(), alarm[labels].mean()


def clean_and_sizes(folder: Path, repeats: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    clean_rows, size_rows = [], []
    for category in CATEGORIES:
        d = dict(np.load(folder / f"{category}_clean0_s0.npz"))
        y = d["labels"].astype(bool)
        cal, test = scores(d)
        for m in DETECTORS:
            for alpha in ALPHAS:
                far, tpr = rates(cal[m], test[m], y, alpha)
                clean_rows.append(
                    {
                        "category": category,
                        "detector": m,
                        "alpha": alpha,
                        "n": len(cal[m]),
                        "false_alarm": far,
                        "detection": tpr,
                    }
                )
        # calibration size: the plain RGB detector, so calibration is untouched by fusion
        for n in SIZES:
            for _ in range(repeats):
                pick = rng.choice(len(cal["rgb"]), n, replace=False)
                far, tpr = rates(cal["rgb"][pick], test["rgb"], y, 0.2)
                size_rows.append(
                    {"category": category, "n": n, "false_alarm": far, "detection": tpr}
                )
    return pd.DataFrame(clean_rows), pd.DataFrame(size_rows)


def broken(folder: Path, alpha: float) -> pd.DataFrame:
    rows = []
    for path in sorted(folder.glob("*.npz")):
        m = FILE.fullmatch(path.name)
        if m is None or m["corruption"] == "clean":
            continue
        name = m["corruption"]
        d = dict(np.load(path))
        y = d["labels"].astype(bool)
        cal, test = scores(d)
        healthy = "fpfh" if corruptions.CORRUPTIONS[name].sensor == "rgb" else "rgb"
        for det in (*DETECTORS, "healthy sensor"):
            src = healthy if det == "healthy sensor" else det
            far, tpr = rates(cal[src], test[src], y, alpha)
            rows.append(
                {
                    "category": m["category"],
                    "corruption": name,
                    "severity": int(m["severity"]),
                    "sensor": corruptions.CORRUPTIONS[name].sensor,
                    "detector": det,
                    "false_alarm": far,
                    "detection": tpr,
                }
            )
    return pd.DataFrame(rows)


def figure(clean: pd.DataFrame, sizes: pd.DataFrame, bad: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
    fig.patch.set_facecolor(SURFACE)

    ax = axes[0]
    for m in DETECTORS:
        part = clean[clean["detector"] == m].groupby("alpha")["false_alarm"].mean()
        ax.plot(part.index, part.values, "o-", color=COLORS[m], label=NAMES[m], ms=4)
    ax.plot([0, 0.25], [0, 0.25], color=MUTED, ls=(0, (3, 3)), lw=0.8)
    ax.set_xlabel("target false alarm rate (alpha)")
    ax.set_ylabel("false alarms on normal test parts", color=INK_2)
    ax.set_title("clean data, mean of 10 categories", loc="left", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=7.5)

    ax = axes[1]
    q = sizes.groupby("n")["false_alarm"].quantile([0.05, 0.5, 0.95]).unstack()
    ax.fill_between(
        q.index, q[0.05], q[0.95], color="#2a78d6", alpha=0.2, lw=0, label="observed, 5-95%"
    )
    ax.plot(q.index, q[0.5], "o-", color="#2a78d6", ms=4, label="observed, median")
    ns = np.arange(4, 41)
    band = np.array([conformal.false_alarm_quantiles(int(n), 0.2, [0.05, 0.95]) for n in ns])
    ax.plot(ns, band[:, 0], color=INK, lw=0.9, ls=(0, (2, 2)), label="theory, 5-95%")
    ax.plot(ns, band[:, 1], color=INK, lw=0.9, ls=(0, (2, 2)))
    ax.axhline(0.2, color=MUTED, lw=0.8)
    ax.set_xlabel("normal parts used for calibration")
    ax.set_ylabel("false alarm rate, alpha = 0.2", color=INK_2)
    ax.set_title("how many normal parts the promise needs (RGB)", loc="left", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=7.5)

    ax = axes[2]
    order = [n for n in corruptions.CORRUPTIONS]
    for i, m in enumerate(["late", "reliable_window", "reliable_extent", "healthy sensor"]):
        part = bad[bad["detector"] == m].groupby("corruption")["false_alarm"].mean().reindex(order)
        color = COLORS.get(m, MUTED)
        ax.barh(
            np.arange(len(order)) + (i - 1.5) * 0.2,
            part.values,
            0.19,
            color=color,
            label=NAMES.get(m, m),
        )
    ax.axvline(0.1, color=MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.set_yticks(np.arange(len(order)), [o.replace("_", " ") for o in order])
    ax.invert_yaxis()
    ax.set_xlabel("false alarms on normal parts, alpha = 0.1")
    ax.set_title(
        "thresholds from clean parts, one sensor broken", loc="left", fontsize=9, color=INK
    )
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")

    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=INK_2, labelsize=7.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scores", type=Path, default=Path("outputs/scores/robustness-extent"))
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("results/analysis"))
    parser.add_argument("--figure", type=Path, default=Path("docs/figures/conformal.png"))
    args = parser.parse_args()

    clean, sizes = clean_and_sizes(args.scores, args.repeats, seed=0)
    bad = broken(args.scores, alpha=0.1)
    args.out.mkdir(parents=True, exist_ok=True)
    clean.round(4).to_csv(args.out / "conformal_clean.csv", index=False)
    sizes.round(4).to_csv(args.out / "conformal_sizes.csv", index=False)
    bad.round(4).to_csv(args.out / "conformal_broken.csv", index=False)
    figure(clean, sizes, bad, args.figure)

    print(clean.groupby(["detector", "alpha"])[["false_alarm", "detection"]].mean().round(3))
    print(sizes.groupby("n")["false_alarm"].describe(percentiles=[0.05, 0.5, 0.95]).round(3))
    print(bad.groupby(["sensor", "detector"])[["false_alarm", "detection"]].mean().round(3))


if __name__ == "__main__":
    main()
