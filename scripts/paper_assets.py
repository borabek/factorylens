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

from factorylens import conformal, corruptions, experiments

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
    values |= text_numbers()
    write("numbers", [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in values.items()])


def text_numbers() -> dict[str, str]:
    """The other values the text quotes, so none of them is typed by hand either."""
    ref = json.loads(Path("configs/reference/btf.json").read_text())
    runs = experiments.collect("results/btf-balanced-btfprep")
    weights = [
        json.loads((Path("results/btf-balanced-btfprep") / f).read_text())["config"][
            "fpfh_weight_fitted"
        ]
        for f in runs["file"]
    ]
    rob = pd.read_csv(ANALYSIS / "robustness_summary.csv")
    clean = rob.drop_duplicates("model").set_index("model")["clean"]
    occl = rob[rob["corruption"] == "occlusion"].set_index("model")["mean"]
    broken = pd.read_csv(ANALYSIS / "conformal_broken.csv")
    rgb = broken[broken["sensor"] == "rgb"]
    far = rgb.groupby("detector")["false_alarm"].mean()
    no_occl = rgb[rgb["corruption"] != "occlusion"].groupby("detector")["false_alarm"].mean()
    checks = pd.read_csv(ANALYSIS / "conformal_checks.csv")
    tire = checks[(checks["check"] == "exchangeability") & (checks["sensor"] == "fpfh")]
    tire = tire.set_index("category").loc["tire"]
    n_val = int(pd.read_csv(ANALYSIS / "conformal_clean.csv")["n"].median())
    lo, hi = conformal.false_alarm_quantiles(n_val, 0.1, [0.05, 0.95])
    kt = pd.read_csv(ANALYSIS / "sweep_kshot_tests.csv")
    fifty = kt[(kt["k"] == 50) & (kt["model"] == "early") & (kt["versus"] == "best single")]

    def pct(x: float) -> str:
        return f"{100 * x:.0f}\\%"

    return {
        "btfConcat": f3(per_category("results/btf-btfprep").mean()),
        "btfPaper": f3(np.mean(ref["methods"]["BTF (RGB + FPFH)"]["image_auroc"])),
        "btfBalanced": f3(per_category("results/btf-balanced-btfprep").mean()),
        "spreadMin": f"{1 / max(weights):.0f}",
        "spreadMax": f"{1 / min(weights):.0f}",
        "earlyOurs": f3(clean["early"]),
        "occlusionPerImage": f3(occl["reliable"]),
        "occlusionLate": f3(occl["late"]),
        "extentCost": f3(clean["reliable_window"] - clean["reliable_extent"]),
        "lateRgbFalse": pct(far["late"]),
        "reliableRgbFalse": pct(far["reliable_window"]),
        "extentRgbFalse": pct(far["reliable_extent"]),
        "healthyRgbFalse": pct(far["healthy sensor"]),
        "reliableRgbFalseNoOcclusion": pct(no_occl["reliable_window"]),
        "tireFalse": pct(tire["false_alarm"]),
        "tireFive": pct(tire["false_alarm_plus5"]),
        "minCalibration": str(conformal.min_calibration(0.05, 0.02)),
        "betaLow": pct(lo),
        "betaHigh": pct(hi),
        "earlyFiftyP": f"{fifty['wilcoxon_p'].iloc[0]:.2f}",
    }


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
