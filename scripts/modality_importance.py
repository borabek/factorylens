"""Which sensor each defect type needs, measured inside the fused model.

For late fusion on the full training set, every defect type (within its
category) is scored against all normal test images three ways: the fused
score, the fused score with the RGB term dropped (FPFH alone) and with the 3D
term dropped (RGB alone). The AUROC lost by dropping a term is that sensor's
importance for the defect type, the "drop one feature" idea from permutation
importance applied to whole modalities.

Day 5 (`compare_sensors.py`) compared the two sensors directly. This asks the
question the fused model raises: for which defects does it need both?

usage: uv run python scripts/modality_importance.py
Writes results/analysis/modality_importance.csv and docs/figures/modality_importance.png.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from factorylens import experiments  # noqa: E402
from factorylens.eval.metrics import image_auroc  # noqa: E402

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
BLUE, ORANGE = "#2a78d6", "#eb6834"
BOTH = 0.02  # each sensor adds at least this much AUROC to the fusion


def load(scores: Path, results: Path, model: str) -> dict[str, dict]:
    runs = experiments.collect(results / model)
    runs = runs[runs["k"].isna()]
    out = {}
    for _, r in runs.iterrows():
        out[r["category"]] = dict(
            np.load(scores / model / Path(r["file"]).with_suffix(".npz").name)
        )
    return out


def table(scores: Path, results: Path) -> pd.DataFrame:
    late, rgb, geo = (load(scores, results, m) for m in ("late", "rgb", "fpfh"))
    rows = []
    for category, fused in late.items():
        defect = fused["defect"]
        assert (rgb[category]["stem"] == fused["stem"]).all()
        normal = defect == "good"
        for kind in sorted(set(defect) - {"good"}):
            keep = normal | (defect == kind)
            y = defect[keep] != "good"
            a_late = image_auroc(y, fused["scores"][keep])
            a_rgb = image_auroc(y, rgb[category]["scores"][keep])
            a_geo = image_auroc(y, geo[category]["scores"][keep])
            rows.append(
                {
                    "category": category,
                    "defect": kind,
                    "n": int((defect == kind).sum()),
                    "late": a_late,
                    "rgb_only": a_rgb,
                    "fpfh_only": a_geo,
                    "rgb_importance": a_late - a_geo,  # what dropping RGB costs
                    "fpfh_importance": a_late - a_rgb,
                }
            )
    t = pd.DataFrame(rows)
    t["needs"] = np.select(
        [
            (t["rgb_importance"] >= BOTH) & (t["fpfh_importance"] >= BOTH),
            t["rgb_importance"] >= BOTH,
            t["fpfh_importance"] >= BOTH,
        ],
        ["both", "RGB", "3D"],
        default="either",
    )
    return t


def figure(t: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    colors = {"both": INK, "RGB": BLUE, "3D": ORANGE, "either": MUTED}
    for need, part in t.groupby("needs"):
        ax.scatter(
            part["rgb_importance"],
            part["fpfh_importance"],
            s=12 + part["n"],
            color=colors[need],
            alpha=0.8,
            label=f"{need} ({len(part)})",
        )
    ax.axhline(0, color=MUTED, lw=0.7)
    ax.axvline(0, color=MUTED, lw=0.7)
    ax.set_xlabel("AUROC lost when RGB is dropped from the fusion")
    ax.set_ylabel("AUROC lost when 3D is dropped from the fusion", color=INK_2)
    ax.set_title("each point: one defect type of one category", loc="left", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=8, title="needs", title_fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scores", type=Path, default=Path("outputs/scores/sweep"))
    parser.add_argument("--results", type=Path, default=Path("results/sweep"))
    parser.add_argument(
        "--out", type=Path, default=Path("results/analysis/modality_importance.csv")
    )
    parser.add_argument("--figure", type=Path, default=Path("docs/figures/modality_importance.png"))
    args = parser.parse_args()

    t = table(args.scores, args.results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t.round(4).to_csv(args.out, index=False)
    figure(t, args.figure)
    print(t["needs"].value_counts())
    by_defect = t.groupby("defect")[["rgb_importance", "fpfh_importance", "n"]].agg(
        {"rgb_importance": "mean", "fpfh_importance": "mean", "n": "sum"}
    )
    print(by_defect.sort_values("n", ascending=False).round(3).to_string())
    print(
        t.sort_values("late")
        .head(8)[["category", "defect", "late", "rgb_only", "fpfh_only"]]
        .round(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
