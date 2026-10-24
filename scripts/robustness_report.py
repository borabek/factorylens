"""Summarise the FactoryLens-C runs (`factorylens robustness`).

1. results/analysis/robustness.csv: image AUROC and AUPRO per model, corruption
   and severity, averaged over the categories that have finished.
2. results/analysis/robustness_summary.csv: one row per model and corruption:
   clean score, mean over the five severities, worst severity, and the share
   of the clean score that is kept.
3. results/analysis/robustness_tests.csv: per corruption, late vs early fusion
   and each fusion vs the healthy sensor (the uncorrupted sensor alone, at its
   clean score in that category). Scores are averaged over severities per
   category and compared over the 10 categories (Wilcoxon signed-rank).
4. docs/figures/robustness.png: image AUROC against severity, one panel per
   corruption, the clean score as a dotted line.

usage: uv run python scripts/robustness_report.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from factorylens import corruptions, experiments  # noqa: E402

MODELS = ("rgb", "fpfh", "early", "late", "reliable", "reliable_window")
NAMES = {
    "rgb": "RGB",
    "fpfh": "3D (FPFH)",
    "early": "early fusion",
    "late": "late fusion",
    "reliable": "reliability, per image",
    "reliable_window": "reliability, 16-part window",
}
COLORS = {
    "rgb": "#2a78d6",
    "fpfh": "#eb6834",
    "early": "#1baf7a",
    "late": "#7a5bd6",
    "reliable": "#e3a008",
    "reliable_window": "#0b0b0b",
}
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"


def load(results: Path) -> pd.DataFrame:
    runs = pd.concat(
        [experiments.collect(results / m).assign(detector=m) for m in MODELS], ignore_index=True
    )
    # only categories with every condition and model, so averages compare like with like
    expected = (len(corruptions.CORRUPTIONS) - 2) * 5 + 2 + 1
    complete = runs.groupby("category").size() == expected * len(MODELS)
    return runs[runs["category"].isin(complete[complete].index)]


def by_condition(runs: pd.DataFrame) -> pd.DataFrame:
    return (
        runs.groupby(["detector", "corruption", "severity"])[["image_auroc", "aupro"]]
        .mean()
        .reset_index()
    )


def summary(table: pd.DataFrame) -> pd.DataFrame:
    clean = table[table["corruption"] == "clean"].set_index("detector")
    rows = []
    for (model, name), part in table[table["corruption"] != "clean"].groupby(
        ["detector", "corruption"]
    ):
        c = clean.loc[model, "image_auroc"]
        rows.append(
            {
                "model": model,
                "corruption": name,
                "sensor": corruptions.CORRUPTIONS[name].sensor,
                "clean": c,
                "mean": part["image_auroc"].mean(),
                "worst": part["image_auroc"].min(),
                "kept": part["image_auroc"].mean() / c,
                "aupro_mean": part["aupro"].mean(),
            }
        )
    return pd.DataFrame(rows)


def tests(runs: pd.DataFrame) -> pd.DataFrame:
    auroc = runs.pivot_table(
        index=["category", "corruption", "severity"], columns="detector", values="image_auroc"
    )
    clean = auroc.xs("clean", level="corruption").droplevel("severity")
    rows = []
    for name, c in corruptions.CORRUPTIONS.items():
        per_cat = auroc.xs(name, level="corruption").groupby(level="category").mean()
        healthy = clean["fpfh" if c.sensor == "rgb" else "rgb"]
        pairs = {
            ("late", "early"): per_cat["late"] - per_cat["early"],
            ("late", "healthy sensor"): per_cat["late"] - healthy,
            ("early", "healthy sensor"): per_cat["early"] - healthy,
            ("reliable", "late"): per_cat["reliable"] - per_cat["late"],
            ("reliable_window", "late"): per_cat["reliable_window"] - per_cat["late"],
            ("reliable_window", "reliable"): per_cat["reliable_window"] - per_cat["reliable"],
            ("reliable_window", "healthy sensor"): per_cat["reliable_window"] - healthy,
        }
        for (a, b), diff in pairs.items():
            # a dead sensor makes late fusion equal the healthy one exactly
            p = float(wilcoxon(diff).pvalue) if (diff != 0).any() else 1.0
            rows.append(
                {
                    "corruption": name,
                    "sensor": c.sensor,
                    "model": a,
                    "versus": b,
                    "mean_diff": diff.mean(),
                    "categories_better": int((diff > 0).sum()),
                    "categories_worse": int((diff < 0).sum()),
                    "wilcoxon_p": p,
                }
            )
    return pd.DataFrame(rows)


def figure(table: pd.DataFrame, n_categories: int, out: Path) -> None:
    names = [n for n, c in corruptions.CORRUPTIONS.items() if c.severities > 1]
    missing = [n for n, c in corruptions.CORRUPTIONS.items() if c.severities == 1]
    fig, axes = plt.subplots(2, 6, figsize=(13.0, 4.8), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    clean = table[table["corruption"] == "clean"].set_index("detector")["image_auroc"]
    for ax, name in zip(axes.flat, [*names, "missing sensor"], strict=False):
        ax.set_facecolor(SURFACE)
        for model in MODELS:
            if name == "missing sensor":
                part = table[(table["detector"] == model) & table["corruption"].isin(missing)]
                part = part.set_index("corruption").loc[missing]
                # side by side: late fusion often lands exactly on the healthy sensor
                dx = (MODELS.index(model) - (len(MODELS) - 1) / 2) * 0.08
                ax.plot([0 + dx, 1 + dx], part["image_auroc"], "o", color=COLORS[model], ms=5)
                ax.set_xticks(range(2), ["no RGB", "no 3D"])
                ax.set_xlim(-0.5, 1.5)
            else:
                part = table[(table["detector"] == model) & (table["corruption"] == name)]
                part = part.sort_values("severity")
                ax.plot(
                    part["severity"],
                    part["image_auroc"],
                    marker="o",
                    ms=2.5,
                    lw=1.3,
                    color=COLORS[model],
                )
                ax.axhline(clean[model], color=COLORS[model], lw=0.6, ls=(0, (1, 2)))
                ax.set_xticks(range(1, 6))
        sensor = "RGB" if name in names and corruptions.CORRUPTIONS[name].sensor == "rgb" else "3D"
        label = (
            name.replace("_", " ")
            if name == "missing sensor"
            else f"{name.replace('_', ' ')} ({sensor})"
        )
        ax.set_title(label, fontsize=8.5, color=INK, loc="left")
        ax.axhline(0.5, color=MUTED, lw=0.6, ls=(0, (3, 3)))
        ax.grid(axis="y", color="#e1e0d9", lw=0.5)
        ax.tick_params(colors=INK_2, labelsize=7.5)
    for ax in axes.flat[len(names) + 1 :]:
        ax.set_visible(False)
    handles = [plt.Line2D([], [], color=COLORS[m], lw=1.6) for m in MODELS]
    fig.legend(
        handles,
        [NAMES[m] for m in MODELS],
        ncol=len(MODELS),
        frameon=False,
        fontsize=8,
        loc="lower center",
    )
    fig.supylabel(f"image AUROC, mean of {n_categories} categories", fontsize=8.5, color=INK_2)
    fig.supxlabel("severity (dotted: clean score)", fontsize=8.5, color=INK_2, y=0.065)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=Path("results/robustness"))
    parser.add_argument("--out", type=Path, default=Path("results/analysis"))
    parser.add_argument("--figure", type=Path, default=Path("docs/figures/robustness.png"))
    args = parser.parse_args()

    runs = load(args.results)
    n = runs["category"].nunique()
    table = by_condition(runs)
    summ = summary(table)
    args.out.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(args.out / "robustness.csv", index=False)
    summ.round(4).to_csv(args.out / "robustness_summary.csv", index=False)
    stats = tests(runs)
    stats.round(4).to_csv(args.out / "robustness_tests.csv", index=False)
    print(stats.pivot(index="corruption", columns=["model", "versus"], values="mean_diff").round(3))
    figure(table, n, args.figure)

    print(f"{n} categories")
    print(summ.pivot(index="corruption", columns="model", values="mean").round(3).to_string())
    by_sensor = summ.groupby(["sensor", "model"])["mean"].mean().unstack().round(3)
    print(by_sensor.to_string())


if __name__ == "__main__":
    main()
