"""Tables and figures for the paper, written from the result files.

Like docs/results.md, no number in paper/ is typed by hand: every table in
paper/generated/ comes from results/ through this script, and the figures are
copied from docs/figures/. Run it before compiling the paper.

usage: uv run python scripts/paper_assets.py
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from factorylens import corruptions, experiments

OUT = Path("paper/generated")
FIGURES = Path("paper/figures")
ANALYSIS = Path("results/analysis")
FIGURE_FILES = [
    "robustness.png",
    "fault_stream.png",
    "data_efficiency.png",
    "corruptions.png",
    "demo_stream.png",
]
NAMES = {
    "rgb": "RGB",
    "fpfh": "3D",
    "early": "Early fusion",
    "late": "Late fusion",
    "reliable_window": "Reliability fusion",
    "reliable_extent": "\\quad + extent signal",
}


def f3(x: float) -> str:
    return f"{x:.3f}"


def per_category(results_dir: str, metric: str = "image_auroc") -> pd.Series:
    runs = experiments.collect(results_dir)
    if "k" in runs:
        runs = runs[runs["k"].isna()]
    return runs.groupby("category")[metric].mean()


def write(name: str, lines: list[str]) -> None:
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n")


def btf_table() -> None:
    ref = json.loads(Path("configs/reference/btf.json").read_text())
    paper = {m: np.mean(v["image_auroc"]) for m, v in ref["methods"].items()}
    rows = [
        ("RGB", per_category("results/rgb-btf-btfprep").mean(), paper["RGB iNet"]),
        ("FPFH", per_category("results/fpfh-btfprep").mean(), paper["FPFH"]),
        (
            "RGB + FPFH, concatenated",
            per_category("results/btf-btfprep").mean(),
            paper["BTF (RGB + FPFH)"],
        ),
        ("RGB + FPFH, halves balanced", per_category("results/btf-balanced-btfprep").mean(), None),
    ]
    write(
        "btf",
        [
            "\\begin{tabular}{lcc}",
            "\\toprule",
            "Back to the Feature setup & ours & published \\\\",
            "\\midrule",
            *[f"{n} & {f3(o)} & {f3(p) if p is not None else '--'} \\\\" for n, o, p in rows],
            "\\bottomrule",
            "\\end{tabular}",
        ],
    )


def sweep_tables() -> None:
    k = pd.read_csv(ANALYSIS / "sweep_kshot.csv")
    k["k"] = k["k"].map(lambda v: "all" if v == float("inf") else str(int(v)))
    models = ["rgb", "fpfh", "early", "late"]
    cols = ["1", "5", "10", "50", "all"]
    lines = [
        "\\begin{tabular}{l" + "c" * len(cols) + "}",
        "\\toprule",
        "Training images & " + " & ".join(cols) + " \\\\",
        "\\midrule",
    ]
    for m in models:
        part = k[k["model"] == m].set_index("k")
        cells = []
        for c in cols:
            r = part.loc[c]
            sd = (
                ""
                if pd.isna(r["image_auroc_sd"])
                else f"\\,{{\\scriptsize$\\pm${r['image_auroc_sd']:.3f}}}"
            )
            cells.append(f"{r['image_auroc']:.3f}{sd}")
        lines.append(f"{NAMES[m]} & " + " & ".join(cells) + " \\\\")
    write("kshot", [*lines, "\\bottomrule", "\\end{tabular}"])


def robustness_tables() -> None:
    summ = pd.read_csv(ANALYSIS / "robustness_summary.csv")
    clean = summ.drop_duplicates("model").set_index("model")["clean"]
    by = summ.groupby(["sensor", "model"])["mean"].mean()
    aupro = summ.groupby(["sensor", "model"])["aupro_mean"].mean()
    rob = pd.read_csv(ANALYSIS / "robustness.csv")
    clean_aupro = rob[rob["corruption"] == "clean"].set_index("detector")["aupro"]
    models = ["rgb", "fpfh", "early", "late", "reliable_window", "reliable_extent"]
    lines = [
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        " & \\multicolumn{3}{c}{image AUROC} & \\multicolumn{3}{c}{AUPRO} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
        " & clean & RGB corr. & 3D corr. & clean & RGB corr. & 3D corr. \\\\",
        "\\midrule",
    ]
    for m in models:
        lines.append(
            f"{NAMES[m]} & {f3(clean[m])} & {f3(by[('rgb', m)])} & {f3(by[('xyz', m)])} & "
            f"{f3(clean_aupro[m])} & {f3(aupro[('rgb', m)])} & {f3(aupro[('xyz', m)])} \\\\"
        )
    lines += [
        "\\midrule",
        f"Healthy sensor alone & -- & {f3(clean['fpfh'])} & {f3(clean['rgb'])} & -- & "
        f"{f3(clean_aupro['fpfh'])} & {f3(clean_aupro['rgb'])} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    write("robustness", lines)

    table = summ.pivot(index="corruption", columns="model", values="mean")
    sensor = summ.drop_duplicates("corruption").set_index("corruption")["sensor"]
    order = [c for c in corruptions.CORRUPTIONS if c in table.index]
    lines = [
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Corruption & sensor & healthy & early & late & reliability & + extent \\\\",
        "\\midrule",
    ]
    for c in order:
        healthy = clean["fpfh"] if sensor[c] == "rgb" else clean["rgb"]
        name = c.replace("_", " ")
        lines.append(
            f"{name} & {'RGB' if sensor[c] == 'rgb' else '3D'} & {f3(healthy)} & "
            f"{f3(table.loc[c, 'early'])} & {f3(table.loc[c, 'late'])} & "
            f"{f3(table.loc[c, 'reliable_window'])} & {f3(table.loc[c, 'reliable_extent'])} \\\\"
        )
    write("corruptions", [*lines, "\\bottomrule", "\\end{tabular}"])


def alarm_table() -> None:
    cc = pd.read_csv(ANALYSIS / "conformal_clean.csv")
    cc = cc[cc["alpha"] == 0.1].groupby("detector")[["false_alarm", "detection"]].mean()
    cb = pd.read_csv(ANALYSIS / "conformal_broken.csv")
    cb = cb.groupby(["sensor", "detector"])[["false_alarm", "detection"]].mean()

    def cell(v: pd.Series) -> str:
        return f"{100 * v['false_alarm']:.0f}\\% / {100 * v['detection']:.0f}\\%"

    lines = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        " & clean & RGB corrupted & 3D corrupted \\\\",
        "\\midrule",
    ]
    for m in ("late", "reliable_window", "reliable_extent"):
        clean = cell(cc.loc[m]) if m in cc.index else "--"
        lines.append(
            f"{NAMES[m]} & {clean} & {cell(cb.loc[('rgb', m)])} & {cell(cb.loc[('xyz', m)])} \\\\"
        )
    lines.append(
        f"Healthy sensor alone & -- & {cell(cb.loc[('rgb', 'healthy sensor')])} & "
        f"{cell(cb.loc[('xyz', 'healthy sensor')])} \\\\"
    )
    write("alarms", [*lines, "\\bottomrule", "\\end{tabular}"])


def numbers() -> None:
    """A few single values used in the text, as LaTeX macros."""
    stream = pd.read_csv(ANALYSIS / "fault_stream.csv")
    g = stream[
        stream["corruption"].isin(
            ["gaussian_noise", "defocus_blur", "jpeg", "depth_noise", "quantise"]
        )
    ]
    g = g.groupby("fusion").agg(
        after=("auroc_after", "mean"), delay=("delay", "median"), false=("false_switch", "mean")
    )
    conf = pd.read_csv(ANALYSIS / "acquisition_confound.csv").set_index("category")
    sens = pd.read_csv(ANALYSIS / "reliability_sensitivity.csv")
    rgb = (
        sens[(sens["sensor"] == "rgb") & (sens["window"] >= 8)]
        .groupby(["ramp", "window"])["auroc"]
        .mean()
    )
    inspectors = pd.read_csv(ANALYSIS / "inspectors.csv")
    cable = conf.loc["cable_gland", "background_mean"]
    values = {
        "delaySixteen": f"{g.loc['window 16', 'delay']:.0f}",
        "delayFour": f"{g.loc['window 4', 'delay']:.0f}",
        "afterSixteen": f3(g.loc["window 16", "after"]),
        "afterFour": f3(g.loc["window 4", "after"]),
        "afterLate": f3(g.loc["late", "after"]),
        "falseFour": f"{g.loc['window 4', 'false']:.1f}",
        "cableBackground": f"{max(cable, 1 - cable):.2f}",
        "sensMin": f3(rgb.min()),
        "sensMax": f3(rgb.max()),
        "msPerPart": f"{inspectors['ms_per_part'].mean():.0f}",
    }
    write("numbers", [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in values.items()])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    btf_table()
    sweep_tables()
    robustness_tables()
    alarm_table()
    numbers()
    for f in FIGURE_FILES:
        shutil.copy(Path("docs/figures") / f, FIGURES / f)
    n = len(list(OUT.glob("*.tex")))
    print(f"wrote {n} tables to {OUT}, {len(FIGURE_FILES)} figures to {FIGURES}")


if __name__ == "__main__":
    main()
