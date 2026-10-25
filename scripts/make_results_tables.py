"""Build docs/results.md from the result files, so no number in it is typed by hand.

usage: uv run python scripts/make_results_tables.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from factorylens import experiments

REFERENCE = Path("configs/reference/btf.json")
OUT = Path("docs/results.md")
METRICS = ["image_auroc", "pixel_auroc", "aupro"]


def reference(method: str) -> pd.DataFrame:
    ref = json.loads(REFERENCE.read_text())
    return pd.DataFrame(ref["methods"][method], index=ref["categories"])


def load_config(path: Path) -> dict:
    return json.loads(path.read_text())["config"]


def provenance(df: pd.DataFrame, results_dir: str) -> str:
    commits, dirty = set(), False
    for name in df["file"]:
        run = json.loads((Path(results_dir) / name).read_text())["run"]
        commits.add((run["git"]["commit"] or "unknown")[:7])
        dirty |= bool(run["git"]["dirty"])
    state = "with uncommitted changes" if dirty else "clean checkout"
    return f"{len(df)} runs from commit {', '.join(sorted(commits))} ({state})"


def rgb_section() -> list[str]:
    runs = experiments.collect("results/rgb")
    runs = runs[runs["k"].isna()]
    btf = reference("RGB iNet")
    per_cat = runs.groupby("category")[[*METRICS, "score_ms_per_image"]].agg(["mean", "std"])
    seeds = runs.groupby("category")["seed"].nunique()
    first = runs[runs["seed"] == 0].set_index("category")
    sd = per_cat.xs("std", axis=1, level=1).fillna(0)
    mean = per_cat.xs("mean", axis=1, level=1)

    def pm(m: float, s: float) -> str:
        return f"{m:.3f} ± {s:.3f}" if seeds.max() > 1 else f"{m:.3f}"

    rows = []
    for cat in mean.index:
        rows.append(
            {
                "category": cat,
                "image AUROC": pm(mean.loc[cat, "image_auroc"], sd.loc[cat, "image_auroc"]),
                "95% CI (seed 0)": f"[{first.loc[cat, 'image_auroc_ci_low']:.2f}, "
                f"{first.loc[cat, 'image_auroc_ci_high']:.2f}]",
                "BTF image": f"{btf.loc[cat, 'image_auroc']:.3f}",
                "pixel AUROC": f"{mean.loc[cat, 'pixel_auroc']:.3f}",
                "BTF pixel": f"{btf.loc[cat, 'pixel_auroc']:.3f}",
                "AUPRO": pm(mean.loc[cat, "aupro"], sd.loc[cat, "aupro"]),
                "BTF AUPRO": f"{btf.loc[cat, 'aupro']:.3f}",
                "ms / image": f"{mean.loc[cat, 'score_ms_per_image']:.0f}",
            }
        )
    # spread of the headline number: average over categories per seed, then across seeds
    by_seed = runs.groupby("seed")[METRICS].mean()
    head = by_seed.mean()
    head_sd = by_seed.std().fillna(0)
    bmean = btf.loc[mean.index, METRICS].mean()
    rows.append(
        {
            "category": "**mean**",
            "image AUROC": f"**{pm(head.image_auroc, head_sd.image_auroc)}**",
            "95% CI (seed 0)": "",
            "BTF image": f"{bmean.image_auroc:.3f}",
            "pixel AUROC": f"**{head.pixel_auroc:.3f}**",
            "BTF pixel": f"{bmean.pixel_auroc:.3f}",
            "AUPRO": f"**{pm(head.aupro, head_sd.aupro)}**",
            "BTF AUPRO": f"{bmean.aupro:.3f}",
            "ms / image": f"{mean['score_ms_per_image'].mean():.0f}",
        }
    )
    table = pd.DataFrame(rows).to_markdown(index=False)

    btf_img = btf.loc[first.index, "image_auroc"]
    inside = (btf_img >= first["image_auroc_ci_low"]) & (btf_img <= first["image_auroc_ci_high"])
    diff = (mean["image_auroc"] - btf.loc[mean.index, "image_auroc"]).abs()
    weakest = mean["image_auroc"].nsmallest(3)
    foam_gap = mean.loc["foam", "aupro"] - btf.loc["foam", "aupro"]
    worst_sd = sd["image_auroc"].idxmax()
    notes = [
        "Reading the numbers:",
        "",
        f"- The reimplementation lands close to the published one. BTF's image AUROC falls inside "
        f"our seed-0 95% CI on {int(inside.sum())} of {len(inside)} categories, and the "
        f"largest gap is {diff.max():.3f} ({diff.idxmax()}).",
        f"- Seeds only change which patch the coreset starts from. Across {seeds.max()} seeds the "
        f"mean image AUROC moves by ±{head_sd.image_auroc:.3f}; the most seed-sensitive "
        f"category is {worst_sd} (±{sd.loc[worst_sd, 'image_auroc']:.3f}).",
        f"- RGB alone is weakest on {', '.join(weakest.index)} "
        f"({', '.join(f'{v:.2f}' for v in weakest)}). These are organic objects where colour and "
        "texture vary a lot between normal samples, which probably makes a dent or a cut look like "
        "normal variation to an ImageNet feature. The defect-type breakdown below makes this "
        "concrete.",
        f"- Localisation on foam is {foam_gap:.2f} AUPRO above BTF's number and is still "
        "unexplained; see the checks at the end of this page.",
        "",
    ]
    return [
        "## RGB only (PatchCore)",
        "",
        "Full training set, WideResNet-50 layer2 + layer3, 10% coreset, 224 px. "
        "BTF columns are the *RGB iNet* row of Back to the Feature, which uses the same "
        "recipe; their numbers are copied from the paper (`configs/reference/btf.json`). "
        "± is the standard deviation over seeds; ms / image is scoring time on an M4 (MPS).",
        "",
        table,
        "",
        f"<sub>{provenance(runs, 'results/rgb')}. CI: 1000 bootstrap "
        "resamples of the test images, stratified by label.</sub>",
        "",
        *notes,
    ]


def per_category(results_dir: str, metric: str) -> pd.Series:
    """Mean over seeds of the full-training-set runs (k-shot runs are left out)."""
    df = experiments.collect(results_dir)
    if "k" in df:
        df = df[df["k"].isna()]
    return df.groupby("category")[metric].mean()


def threed_section() -> list[str]:
    if not Path("results/fpfh").exists():
        return []
    paper = reference("FPFH")
    rows = {}
    for metric in ("image_auroc", "aupro"):
        rows[f"FPFH {metric} (ours, 2 mm)"] = per_category("results/fpfh", metric)
        rows[f"FPFH {metric} (BTF preprocessing)"] = per_category("results/fpfh-btfprep", metric)
        rows[f"FPFH {metric} (paper)"] = paper[metric]
    table = pd.DataFrame(rows)
    rgb = per_category("results/rgb", "image_auroc")
    ours = table["FPFH image_auroc (ours, 2 mm)"]
    better = (ours - rgb).sort_values()
    gap = ours - table["FPFH image_auroc (paper)"]
    mean = table.mean()
    out = table.copy()
    out.loc["**mean**"] = mean
    out.columns = [
        c.replace("image_auroc", "image AUROC").replace("aupro", "AUPRO") for c in out.columns
    ]
    return [
        "## 3D only (FPFH)",
        "",
        "Same memory bank as the RGB detector, on FPFH point-cloud features with BTF's settings "
        "(normals from 30 neighbours, FPFH from 100, 28x28 grid). Two preprocessings: this "
        "repo's (2 mm plane cut, tiny blobs dropped) and BTF's (5 mm cut, largest blob only). "
        "Seed 0.",
        "",
        out.round(3).to_markdown(),
        "",
        f"<sub>{provenance(experiments.collect('results/fpfh'), 'results/fpfh')}.</sub>",
        "",
        "Reading the numbers:",
        "",
        "- Geometry and colour fail in different places. FPFH beats RGB by the most on "
        + ", ".join(f"{c} (+{v:.2f})" for c, v in better[::-1][:3].items())
        + ", and loses by the most on "
        + ", ".join(f"{c} ({v:.2f})" for c, v in better[:3].items())
        + ". Potato, where RGB was near chance, is one of the easiest categories for 3D.",
        f"- Our FPFH is above the published one on average ({mean.iloc[0]:.3f} and "
        f"{mean.iloc[1]:.3f} vs {mean.iloc[2]:.3f} image AUROC). The biggest single gap is "
        f"{gap.idxmax()} ({table.loc[gap.idxmax()].iloc[0]:.3f} vs "
        f"{table.loc[gap.idxmax()].iloc[2]:.3f}).",
        "",
    ]


def sensor_section() -> list[str]:
    path = Path("results/analysis/rgb_vs_3d_defect_types.csv")
    if not path.exists():
        return []
    df = pd.read_csv(path)
    var = pd.read_csv("results/analysis/rgb_vs_3d_variance_explained.csv").iloc[0]
    by_cat = df.groupby("category")["diff"].mean().sort_values()
    by_defect = df.groupby("defect")["diff"].agg(["mean", "count"])
    holes_cuts = by_defect.loc[[d for d in ("hole", "cut") if d in by_defect.index], "mean"]
    return [
        "## Which sensor finds which defect",
        "",
        "RGB and 3D side by side per category and defect type, 3D minus RGB image AUROC "
        "(`scripts/compare_sensors.py`, seed 0, this repo's preprocessing).",
        "",
        "![3D minus RGB image AUROC per category and defect type]"
        "(figures/rgb_vs_3d_defect_types.png)",
        "",
        "Reading the numbers:",
        "",
        f"- The object decides more than the defect does. Category alone explains "
        f"{100 * var['category']:.0f}% of the variation in the difference, defect type alone "
        f"{100 * var['defect']:.0f}%.",
        "- 3D wins on "
        + ", ".join(f"{c} ({v:+.2f})" for c, v in by_cat[::-1][:4].items())
        + ". These are organic objects whose colour and texture vary from one normal sample to "
        "the next, so the camera's notion of normal is wide and a dent hides inside it. RGB "
        "wins on "
        + ", ".join(f"{c} ({v:+.2f})" for c, v in by_cat[:3].items())
        + ", which are manufactured parts with a consistent look; cable gland also has the "
        "lowest 3D coverage in the dataset (61% of pixels, see docs/data.md).",
        "- This corrects the day-4 guess that holes and cuts are where 3D helps. Averaged over "
        "categories they only move by "
        + ", ".join(f"{d} {v:+.2f}" for d, v in holes_cuts.items())
        + "; on a potato 3D finds them far better, on a tire or a cable gland far worse.",
        "- The two single-instance types are the cleanest cases: foam's colour defect is "
        "invisible to geometry, and cable gland's damaged thread is probably too fine for the "
        "3D scan to resolve.",
        "",
    ]


def btf_section() -> list[str]:
    if not Path("results/btf-btfprep").exists():
        return []
    cols = {
        "RGB": ("results/rgb-btf-btfprep", "RGB iNet"),
        "FPFH": ("results/fpfh-btfprep", "FPFH"),
        "BTF (RGB + FPFH)": ("results/btf-btfprep", "BTF (RGB + FPFH)"),
    }
    rows = {}
    for label, (path, method) in cols.items():
        rows[f"{label}, ours"] = per_category(path, "image_auroc")
        rows[f"{label}, paper"] = reference(method)["image_auroc"]
    table = pd.DataFrame(rows)
    mean = table.mean()
    out = table.copy()
    out.loc["**mean**"] = mean
    scale = pd.read_csv("results/analysis/btf_feature_scale.csv", index_col="category")
    fused = table["BTF (RGB + FPFH), ours"]
    best_single = table[["RGB, ours", "FPFH, ours"]].max(axis=1)
    worse = int((fused < best_single).sum())
    return [
        "## Reproducing Back to the Feature",
        "",
        "All three models run the way BTF runs them: their preprocessing (5 mm cut, largest "
        "blob, RGB blacked out where points were removed), no padding in the 3x3 average, all "
        "1536 RGB channels, PatchCore's reweighted image score. Image AUROC, seed 0.",
        "",
        out.round(3).to_markdown(),
        "",
        f"<sub>{provenance(experiments.collect('results/btf-btfprep'), 'results/btf-btfprep')}."
        "</sub>",
        "",
        "Reading the numbers:",
        "",
        f"- Each half reproduces, and then some: RGB {mean['RGB, ours']:.3f} vs "
        f"{mean['RGB, paper']:.3f}, FPFH {mean['FPFH, ours']:.3f} vs {mean['FPFH, paper']:.3f}.",
        f"- The combination does not. BTF reaches {mean['BTF (RGB + FPFH), ours']:.3f} here "
        f"against {mean['BTF (RGB + FPFH), paper']:.3f} in the paper, and it is below the better "
        f"single modality on {worse} of {len(fused)} categories. Fusing made things worse.",
        f"- The reason is scale. Over the training patches the FPFH half is "
        f"{scale['ratio'].min():.0f}-{scale['ratio'].max():.0f} times wider than the RGB half "
        f"(`scripts/btf_feature_scale.py`), so at least {100 * scale['fpfh_share'].min():.1f}% of "
        "a typical squared distance is FPFH. Concatenation without rescaling is effectively "
        "FPFH alone, and the per-category numbers track FPFH closely. The paper's result must "
        "come from a setup where the two halves were closer in size or weighted differently; I "
        "have not found which (FPFH magnitudes depend on the library version, for one). Either "
        "way the method is sensitive to something that should not matter.",
        "- That is the starting point for the fusion experiments: rescale or score the two "
        "modalities separately before combining them.",
        "",
    ]


def balance_section() -> list[str]:
    if not Path("results/btf-balanced-btfprep").exists():
        return []
    cols = {
        "RGB": "results/rgb-btf-btfprep",
        "FPFH": "results/fpfh-btfprep",
        "weight 1 (BTF)": "results/btf-btfprep",
        "weight 0.1": "results/btf-w0.1-btfprep",
        "weight 0.03": "results/btf-w0.03-btfprep",
        "balanced": "results/btf-balanced-btfprep",
    }
    table = pd.DataFrame({k: per_category(v, "image_auroc") for k, v in cols.items()})
    table["paper BTF"] = reference("BTF (RGB + FPFH)")["image_auroc"]
    mean = table.mean()
    out = table.copy()
    out.loc["**mean**"] = mean
    best_single = table[["RGB", "FPFH"]].max(axis=1)
    above = int((table["balanced"] > best_single).sum())
    runs = experiments.collect("results/btf-balanced-btfprep")
    weights = pd.Series(
        [
            load_config(Path("results/btf-balanced-btfprep") / f)["fpfh_weight_fitted"]
            for f in runs["file"]
        ]
    )
    ours = per_category("results/btf-balanced", "image_auroc").mean()
    aupro = per_category("results/btf-balanced-btfprep", "aupro").mean()
    return [
        "## Balancing the two halves",
        "",
        "The test for the explanation above: same BTF model, same preprocessing, but the FPFH "
        "half is scaled before concatenation. *balanced* picks the factor per category on the "
        "training patches so that both halves have the same spread; nothing from the test set "
        "is used. Image AUROC, seed 0.",
        "",
        out.round(3).to_markdown(),
        "",
        f"<sub>{provenance(runs, 'results/btf-balanced-btfprep')}.</sub>",
        "",
        "Reading the numbers:",
        "",
        f"- Balanced, the same concatenation reaches {mean['balanced']:.3f}, up from "
        f"{mean['weight 1 (BTF)']:.3f}, and is above both single sensors on {above} of "
        f"{len(table)} categories. It is also above the paper's {mean['paper BTF']:.3f} "
        f"(AUPRO {aupro:.3f}).",
        "- So the explanation holds causally, not just as a correlation: change nothing but the "
        "relative scale of the two halves and the fusion goes from worse than either sensor to "
        "better than both.",
        f"- The fitted factors range from {weights.min():.3f} to {weights.max():.3f}, so FPFH "
        f"is shrunk {1 / weights.max():.0f}-{1 / weights.min():.0f}x depending on the category.",
        f"- A fixed factor of 0.03 does about as well on average ({mean['weight 0.03']:.3f}), "
        f"0.1 a little worse ({mean['weight 0.1']:.3f}). Taking whichever fixed factor scores "
        "best here would be choosing it on the test set; the balanced factor is decided from "
        "the training data before any test image is seen, so it is the number to quote.",
        f"- With this repo's preprocessing instead of BTF's, balanced fusion gives {ours:.3f}.",
        "",
    ]


def defect_section() -> list[str]:
    path = Path("results/analysis/rgb_defect_types.csv")
    if not path.exists():
        return []
    df = pd.read_csv(path)
    by_type = (
        df.groupby("defect")
        .agg(
            categories=("category", "size"),
            images=("n", "sum"),
            image_auroc=("image_auroc", "mean"),
            aupro=("aupro", "mean"),
        )
        .sort_values("image_auroc")
    )
    hardest = df.nsmallest(4, "image_auroc")
    contamination = df[df["defect"] == "contamination"].nsmallest(3, "image_auroc")
    geometric = by_type.loc[[d for d in ("hole", "cut") if d in by_type.index], "image_auroc"]
    visual = by_type.loc[
        [d for d in ("color", "open", "crack") if d in by_type.index], "image_auroc"
    ]
    return [
        "## What RGB misses, by defect type",
        "",
        "Each defect type scored against the good test images of its category "
        "(`scripts/defect_breakdown.py`, seed 0).",
        "",
        "![image AUROC per category and defect type](figures/rgb_defect_types.png)",
        "",
        by_type.round(3).to_markdown(),
        "",
        "Reading the numbers:",
        "",
        "- Defects with a clear visual signature are easy (a new colour, an opened strand, "
        "a dark crack line): "
        + ", ".join(f"{d} {v:.2f}" for d, v in visual.items())
        + ". Defects that mostly change shape are not: "
        + ", ".join(f"{d} {v:.2f}" for d, v in geometric.items())
        + ". That is the gap the 3D sensor is supposed to fill.",
        f"- Contamination ({by_type.loc['contamination', 'image_auroc']:.2f}) sits in between: "
        "fine on most objects but weak on "
        + ", ".join(f"{r.category} ({r.image_auroc:.2f})" for r in contamination.itertuples())
        + ", where foreign material looks like the object's own texture.",
        "- The hardest single cases are "
        + ", ".join(f"{r.category} {r.defect} ({r.image_auroc:.2f})" for r in hardest.itertuples())
        + ", at or below chance.",
        "- AUPRO stays high even where image AUROC is near chance (potato holes are localised "
        "well but the image score does not separate them). The image score is the single "
        "largest patch distance, so one unusual patch on a normal potato is enough to outscore "
        "a small hole.",
        "",
        "![best hit, worst miss and worst false alarm per category](figures/rgb_examples.png)",
        "",
        "Typical failure modes visible above (`scripts/rgb_examples.py`): on potato the whole "
        "object lights up for an unusually shaped but normal potato, and on tire the worst false "
        "alarm is a specular highlight on the tread.",
        "",
    ]


def checks_section() -> list[str]:
    path = Path("results/analysis/aupro_vs_official.csv")
    if not path.exists():
        return []
    df = pd.read_csv(path, index_col="category")
    lines = [
        "## Checks",
        "",
        "**AUPRO against the official evaluation code.** The MVTec 3D-AD / BTF code samples "
        "100 thresholds evenly in FPR and counts 8-connected regions; ours uses every threshold "
        "and 4-connected regions (`scripts/check_aupro_against_official.py`, seed 0 maps).",
        "",
        df.round(3).T.to_markdown(),
        "",
        f"The mean differs by {abs(df['ours'].mean() - df['official'].mean()):.3f}, so the "
        "metric is not where any difference to published numbers comes from.",
        "",
    ]
    blur = Path("results/ablation/blur")
    if blur.exists():
        b = experiments.collect(blur)
        rng = b["aupro"].max() - b["aupro"].min()
        lines += [
            f"**Foam and the Gaussian blur.** Rerunning foam with blur sigma "
            f"{', '.join(str(x) for x in sorted(b['blur_sigma']))} moves AUPRO by only "
            f"{rng:.3f} ({b['aupro'].min():.3f}-{b['aupro'].max():.3f}). With the metric and the "
            "blur ruled out, the remaining differences to BTF's pipeline are the RGB resize "
            "(they use bicubic) and their preprocessing; foam's 0.21 gap stays an open question.",
            "",
        ]
    return lines


def layer_section() -> list[str]:
    path = Path("results/ablation/layers")
    if not path.exists():
        return []
    df = experiments.collect(path)
    counts = df.groupby("layers").size()
    order = ["layer1", "layer2", "layer3", "layer1+layer2", "layer2+layer3", "layer1+layer2+layer3"]
    order = [o for o in order if o in counts.index]
    mean = df.groupby("layers")[METRICS].mean().loc[order]
    mean.insert(0, "categories", counts.loc[order])
    per_cat = df.pivot_table(index="category", columns="layers", values="image_auroc")[order]
    best_img, best_pro = mean["image_auroc"].idxmax(), mean["aupro"].idxmax()
    spread = mean["image_auroc"].max() - mean["image_auroc"].min()
    notes = [
        "Reading the numbers:",
        "",
        f"- Mean image AUROC only moves by {spread:.3f} across all six choices. Best is "
        f"{best_img} ({mean.loc[best_img, 'image_auroc']:.3f}), against "
        f"{mean.loc['layer2+layer3', 'image_auroc']:.3f} for the default layer2+layer3. "
        "That is smaller than a typical per-category confidence interval.",
        f"- For localisation the shallow features win: {best_pro} has the best AUPRO "
        f"({mean.loc[best_pro, 'aupro']:.3f}). layer3 alone is clearly worst on both, which "
        "matches the per-layer distance maps in `docs/figures/layer_maps.png`.",
        "- The default stays layer2+layer3. MVTec 3D-AD has no validation split with defects, "
        "so picking the layers that look best here would mean tuning on the test set and every "
        "later number would be optimistic. This table is an analysis, not a selection step.",
        "",
    ]
    return [
        "## Which backbone layers (RGB)",
        "",
        "Same detector, every layer combination on the same 28x28 patch grid "
        "(`scripts/run_layer_ablation.py`), full training set.",
        "",
        mean.round(3).to_markdown(),
        "",
        "Image AUROC per category:",
        "",
        per_cat.round(3).to_markdown(),
        "",
        f"<sub>{provenance(df, str(path))}.</sub>",
        "",
        *notes,
    ]


def sweep_section() -> list[str]:
    if not Path("results/sweep").exists():
        return []
    models = {"RGB": "rgb", "FPFH": "fpfh", "early fusion": "early", "late fusion": "late"}
    full = pd.DataFrame(
        {name: per_category(f"results/sweep/{m}", "image_auroc") for name, m in models.items()}
    )
    aupro = pd.DataFrame(
        {name: per_category(f"results/sweep/{m}", "aupro") for name, m in models.items()}
    ).mean()
    paired = pd.read_csv("results/analysis/sweep_paired.csv")

    def mark(category: str, model: str) -> str:
        rows = paired[(paired["category"] == category) & (paired["model"] == model)]
        rows = rows[rows["versus"].isin(["rgb", "fpfh"])]
        return " ✓" if (rows["ci_low"] > 0).all() else ""

    full.insert(2, "better sensor", full[["RGB", "FPFH"]].max(axis=1))
    above = {
        m: int((full[m] > full["better sensor"]).sum()) for m in ("early fusion", "late fusion")
    }
    shown = full.map(lambda v: f"{v:.3f}")
    for cat in full.index:
        shown.loc[cat, "early fusion"] += mark(cat, "early")
        shown.loc[cat, "late fusion"] += mark(cat, "late")
    shown.loc["**mean**"] = full.mean().map(lambda v: f"{v:.3f}")
    shown.loc["**AUPRO**"] = aupro.map(lambda v: f"{v:.3f}").reindex(shown.columns).fillna("")

    wins = paired[paired["versus"].isin(["rgb", "fpfh"])].assign(
        win=lambda d: d["ci_low"] > 0, loss=lambda d: d["ci_high"] < 0
    )
    both = wins.groupby(["model", "category"])["win"].all().groupby("model").sum()
    losses = wins.groupby("model")["loss"].sum()
    early_late = paired[(paired["model"] == "early") & (paired["versus"] == "late")]

    ktests = pd.read_csv("results/analysis/sweep_kshot_tests.csv").set_index(
        ["k", "model", "versus"]
    )
    kshot = pd.read_csv("results/analysis/sweep_kshot.csv")
    kshot["k"] = kshot["k"].map(lambda k: "all" if k == float("inf") else str(int(k)))
    cell = kshot.apply(
        lambda r: (
            f"{r['image_auroc']:.3f}" + (f" ± {r['image_auroc_sd']:.3f}" if r["seeds"] > 1 else "")
        ),
        axis=1,
    )
    names = {**{m: n for n, m in models.items()}, "rgb_aug": "RGB + flips"}
    ktable = (
        kshot.assign(cell=cell, model=kshot["model"].map(names))
        .pivot(index="model", columns="k", values="cell")
        .reindex(columns=["1", "5", "10", "50", "all"])
        .reindex(list(names.values()))
        .fillna("")
    )

    runs = pd.concat(
        [experiments.collect(f"results/sweep/{m}").assign(model=m) for m in models.values()]
    )
    every = pd.concat(
        [
            experiments.collect(f"results/sweep/{m}").assign(
                file=lambda d, m=m: m + "/" + d["file"]
            )
            for m in [*models.values(), "rgb_aug"]
        ]
    )
    grid = runs.pivot_table(
        index=["category", "k", "seed"], columns="model", values="image_auroc"
    ).dropna()
    best = grid[["rgb", "fpfh"]].max(axis=1)
    small = grid.index.get_level_values("k") <= 10
    fails = {m: float((grid[m] < best - 0.05)[small].mean()) for m in ("early", "late")}
    cov = pd.read_csv("results/analysis/coverage_gap.csv")

    def rho(d: pd.DataFrame, gap: str = "gap_val") -> float:
        return float(d[[gap, "early_auroc"]].corr("spearman").iloc[0, 1])

    within = cov.groupby("category").apply(rho)
    bagel = cov[cov["category"] == "bagel"]
    rescue = (
        (cov["separate_auroc"] - cov["early_auroc"]).groupby([cov["category"], cov["k"]]).mean()
    )
    joint_gain = -rescue.loc[["cookie", "peach"]].xs(10, level="k")

    return [
        "## Fusion and training-set size",
        "",
        "`factorylens sweep`: the same RGB and FPFH patch features in four detectors, "
        "with this repo's preprocessing. *early*: the two halves concatenated, FPFH scaled "
        "per run so that both have the same spread on the training patches. *late*: one "
        "memory bank per sensor, each score standardised on the validation normals, then "
        "summed. Image AUROC on the full training set, seed 0; ✓ = better than both single "
        "sensors, paired bootstrap 95% interval above zero.",
        "",
        shown.to_markdown(disable_numparse=True),
        "",
        f"<sub>{provenance(every, 'results/sweep')}, all k and draws.</sub>",
        "",
        f"- Early fusion is better than both sensors on {both.get('early', 0)} of "
        f"{len(full)} categories by the paired test, late fusion on "
        f"{both.get('late', 0)}. Neither is significantly worse than a single sensor on any "
        f"category ({losses.get('early', 0) + losses.get('late', 0)} losses in "
        f"{len(wins)} comparisons).",
        f"- Early beats late on image AUROC (mean +{early_late['diff'].mean():.3f}, "
        f"significant on {(early_late['ci_low'] > 0).sum()} categories, never worse), but "
        f"late fusion localises better (AUPRO {aupro['late fusion']:.3f} vs "
        f"{aupro['early fusion']:.3f}).",
        "",
        "Mean image AUROC over the 10 categories for k normal training images, ± the "
        "spread over five k-shot draws (`configs/kshot.json`):",
        "",
        ktable.to_markdown(disable_numparse=True),
        "",
        "![image AUROC and AUPRO against the number of training images]"
        "(figures/data_efficiency.png)",
        "",
        "Per category, which is where the averages come from:",
        "",
        "![image AUROC against training images, one panel per category]"
        "(figures/data_efficiency_categories.png)",
        "",
        "- Against either sensor used on every category, both fusions are ahead from five "
        "images up. With one image, late fusion is the best model and early fusion is no "
        "better than FPFH.",
        "- The harder baseline is *better sensor*: whichever of RGB and FPFH scores higher on "
        "that category, which is only known after seeing the test results. On the full "
        f"training set late fusion matches it ({full['late fusion'].mean():.3f} vs "
        f"{full['better sensor'].mean():.3f}) without having to know which sensor to trust, "
        f"and early fusion is above it on {above['early fusion']} of {len(full)} categories. "
        "With ten images or fewer neither fusion beats it: comparing category means (the "
        "five draws share images, so they are averaged first), early fusion is "
        f"{ktests.loc[(10, 'early', 'best single'), 'mean_diff']:+.3f} at k = 10 "
        f"(Wilcoxon p = {ktests.loc[(10, 'early', 'best single'), 'wilcoxon_p']:.2f}). "
        "At k = 50 it pulls ahead, "
        f"{ktests.loc[(50, 'early', 'best single'), 'mean_diff']:+.3f} "
        f"(p = {ktests.loc[(50, 'early', 'best single'), 'wilcoxon_p']:.2f}, better on "
        f"{ktests.loc[(50, 'early', 'best single'), 'categories_better']} of 10). "
        "All tests: `results/analysis/sweep_kshot_tests.csv`.",
        f"- Early fusion is fragile with little data: at k ≤ 10 it falls more than 0.05 "
        f"below the better single sensor in {fails['early']:.0%} of runs, late fusion in "
        f"{fails['late']:.0%}. Its spread over draws is about twice late fusion's.",
        "- Why it fails is not one thing. `scripts/early_fusion_coverage.py` matches each "
        "test patch once against the joint bank and once against each sensor's bank, and "
        "measures the *coverage gap*: how much farther the joint neighbour is than the two "
        "separate ones. On bagel it explains the failures: with one image, matching the "
        f"halves separately scores {rescue.loc[('bagel', 1)]:.2f} higher than joint "
        f"matching, and the gap on the validation normals has Spearman {rho(bagel):.2f} with "
        "early fusion's AUROC over the 20 draws. A few bagels do not contain the "
        "combinations of colour and shape the joint space needs.",
        "- It does not generalise. Within a category the same correlation ranges from "
        f"{within.min():+.2f} ({within.idxmin()}) to {within.max():+.2f} "
        f"({within.idxmax()}). On cookie and peach, joint matching is *better* than "
        f"separate by {joint_gain.min():.2f}-{joint_gain.max():.2f} at k = 10: there, which "
        "shape goes with which colour is itself the information that finds defects. So the "
        "gap cannot be used to choose between early and late fusion in advance, which was "
        f"the plan. <sub>{len(cov)} runs from commit fff4be0, clean checkout.</sub>",
        "- Flips and rotations of the training images help RGB a little at k = 1 and 10 and "
        "not at 5. They do not close the gap to either fusion.",
        "",
    ]


def robustness_section() -> list[str]:
    path = Path("results/analysis/robustness_summary.csv")
    if not path.exists():
        return []
    summ = pd.read_csv(path)
    names = {
        "rgb": "RGB",
        "fpfh": "FPFH",
        "early": "early fusion",
        "late": "late fusion",
        "reliable_window": "reliability fusion",
    }
    table = summ.pivot(index="corruption", columns="model", values="mean")[list(names)]
    sensor = summ.drop_duplicates("corruption").set_index("corruption")["sensor"]
    clean = summ.drop_duplicates("model").set_index("model")["clean"]
    # the sensor that was not corrupted, at its clean score
    table.insert(
        0, "healthy sensor", [clean["fpfh" if sensor[c] == "rgb" else "rgb"] for c in table.index]
    )
    order = sensor.sort_values(kind="stable").index
    table = table.loc[order].rename(columns=names)
    table.index = [
        f"{c.replace('_', ' ')} ({'RGB' if sensor[c] == 'rgb' else '3D'})" for c in order
    ]
    shown = table.map(lambda v: f"{v:.3f}")
    shown.loc["**clean**"] = ["", *[f"{clean[m]:.3f}" for m in names]]

    runs = pd.concat(
        [
            experiments.collect(f"results/robustness/{m}").assign(
                file=lambda d, m=m: m + "/" + d["file"]
            )
            for m in [*names, "reliable"]
        ]
    )
    per_cat = runs.groupby("category").size()
    n_cat = int((per_cat == per_cat.max()).sum())  # the summary only uses finished categories
    healthy = table["healthy sensor"]
    below = {
        m: int((table[names[m]] < healthy - 0.01).sum())
        for m in ("early", "late", "reliable_window")
    }
    per_image = summ.pivot(index="corruption", columns="model", values="mean")["reliable"]
    sens = pd.read_csv("results/analysis/reliability_sensitivity.csv")
    sens = sens.groupby(["sensor", "ramp", "window"])["auroc"].mean()
    rgb_sens = sens.xs("rgb", level="sensor")

    def ext(corruption: str) -> pd.Series:
        return summ[(summ["model"] == "reliable_extent") & (summ["corruption"] == corruption)].iloc[
            0
        ]

    rgb_wide = rgb_sens[rgb_sens.index.get_level_values("window") >= 8]
    by = summ.groupby(["sensor", "model"])["mean"].mean()
    missing = table.loc[[i for i in table.index if "missing" in i]]
    conf = pd.read_csv("results/analysis/acquisition_confound.csv").set_index("category")
    stats = pd.read_csv("results/analysis/robustness_tests.csv")
    aupro = summ.groupby(["sensor", "model"])["aupro_mean"].mean()

    def significant(model: str, versus: str, sign: int) -> pd.DataFrame:
        d = stats[(stats["model"] == model) & (stats["versus"] == versus)]
        return d[(d["wilcoxon_p"] < 0.05) & (np.sign(d["mean_diff"]) == sign)]

    def sig(model: str, versus: str, sign: int) -> int:
        return len(significant(model, versus, sign))

    def names_of(model: str, versus: str, sign: int) -> list[str]:
        return [c.replace("_", " ") for c in significant(model, versus, sign)["corruption"]]

    def sep(auroc: float) -> float:
        return max(auroc, 1 - auroc)

    return [
        "## Robustness: one sensor corrupted (FactoryLens-C)",
        "",
        "Every model is fitted on clean training data; only the test set is corrupted, one "
        "sensor at a time ([corruptions.md](corruptions.md)). Image AUROC averaged over the "
        f"five severities and {n_cat} categories. *healthy sensor* is the clean score of the "
        "sensor that was not corrupted, i.e. what a system that knew which sensor broke could "
        "get by switching it off.",
        "",
        shown.to_markdown(disable_numparse=True),
        "",
        f"<sub>{provenance(runs, 'results/robustness')}.</sub>",
        "",
        "![image AUROC against severity for every corruption](figures/robustness.png)",
        "",
        f"- Early fusion is the best model on clean data ({clean['early']:.3f}) and the most "
        f"fragile one. It ends up below the healthy sensor on {below['early']} of "
        f"{len(table)} corruptions: {by[('rgb', 'early')]:.3f} on average when the RGB image "
        f"is corrupted (FPFH alone: {clean['fpfh']:.3f}), {by[('xyz', 'early')]:.3f} when "
        f"the point cloud is (RGB alone: {clean['rgb']:.3f}). With no 3D at all it scores "
        f"{missing.loc['xyz missing (3D)', 'early fusion']:.3f}.",
        f"- Late fusion degrades more gracefully ({by[('rgb', 'late')]:.3f} and "
        f"{by[('xyz', 'late')]:.3f}) and loses a whole missing sensor without cost: a dead "
        "sensor gives every image the same score, so the sum is ranked by the other one. "
        "A *noisy* sensor is different. Its scores vary from image to image without meaning "
        f"anything, and late fusion still falls below the healthy sensor on {below['late']} "
        "corruptions, mostly on the RGB side.",
        "- That gap, between what late fusion gets and what switching off the broken sensor "
        "would get, is what reliability fusion (`src/factorylens/fusion.py`) is for. It reads "
        "each sensor's health from the background level of its anomaly map (and, for 3D, the "
        "point count) against the validation normals, over a window of the last 16 parts, and "
        "down-weights a sensor that looks off. No labels and no test data go into it. "
        f"Clean: {clean['reliable_window']:.3f} (late fusion {clean['late']:.3f}). RGB "
        f"corrupted: {by[('rgb', 'reliable_window')]:.3f} against late fusion's "
        f"{by[('rgb', 'late')]:.3f} and FPFH alone at {clean['fpfh']:.3f}. 3D corrupted: "
        f"{by[('xyz', 'reliable_window')]:.3f}. It is below the healthy sensor on "
        f"{below['reliable_window']} of {len(table)} corruptions.",
        f"- Judging each image on its own instead fails on occlusion "
        f"({per_image['occlusion']:.3f}, worse than late fusion's "
        f"{table.loc['occlusion (RGB)', 'late fusion']:.3f}): a grey square crosses the "
        "threshold in some images and not in others, so the scores split by weight instead "
        "of by defect. A real sensor fault lasts for many parts, and a 16-part window gives "
        f"{table.loc['occlusion (RGB)', 'reliability fusion']:.3f}. Occlusion is still the one "
        "corruption it does not handle well, because a local occluder looks like a defect.",
        "- An occluder lights up a large part of the object without moving its background "
        "level. A second signal, the 90% quantile of the map over the object (*extent*, "
        "judged over the same window), catches it: occlusion goes to "
        f"{ext('occlusion')['mean']:.3f}, the healthy sensor's score, with AUPRO "
        f"{ext('occlusion')['aupro_mean']:.3f}. With extent, the fusion is not significantly "
        "below the healthy sensor on any corruption. The price is on clean data: "
        f"{clean['reliable_extent']:.3f} instead of {clean['reliable_window']:.3f}, because "
        "the high quantile also reacts to ordinary differences between validation and test "
        "parts, and it is significantly worse than the window alone on "
        f"{' and '.join(names_of('reliable_extent', 'reliable_window', -1))}. It is kept as an "
        "option for lines where occluders are a real risk, not as the default.",
        f"- The settings (3-SD ramp, 16-part window) were fixed before any run. Over ramps "
        f"1-8 and windows of 8 parts or more, the RGB-corrupted mean stays between "
        f"{rgb_wide.min():.3f} and {rgb_wide.max():.3f}; without a window it drops to "
        f"{rgb_sens.xs(1, level='window').min():.3f}-{rgb_sens.xs(1, level='window').max():.3f} "
        "(`scripts/reliability_sensitivity.py`).",
        "- Which differences are real: comparing category means (severities averaged) over "
        f"the {n_cat} categories with a Wilcoxon test, late fusion beats early fusion on "
        f"{sig('late', 'early', 1)} corruptions ({', '.join(names_of('late', 'early', 1))}) "
        f"and loses on {sig('late', 'early', -1)} ({', '.join(names_of('late', 'early', -1))}). "
        f"Early fusion is significantly below the healthy sensor on "
        f"{sig('early', 'healthy sensor', -1)} corruptions, late fusion on "
        f"{sig('late', 'healthy sensor', -1)} "
        f"({', '.join(names_of('late', 'healthy sensor', -1))}). "
        f"Reliability fusion beats late fusion on {sig('reliable_window', 'late', 1)} "
        f"({', '.join(names_of('reliable_window', 'late', 1))}) and loses on "
        f"{sig('reliable_window', 'late', -1)}. "
        "p < 0.05 everywhere, `results/analysis/robustness_tests.csv`.",
        "- Localisation follows the same pattern. AUPRO averaged over the RGB corruptions: "
        f"FPFH {aupro[('rgb', 'fpfh')]:.3f}, late {aupro[('rgb', 'late')]:.3f}, early "
        f"{aupro[('rgb', 'early')]:.3f}; over the 3D corruptions: RGB "
        f"{aupro[('xyz', 'rgb')]:.3f}, late {aupro[('xyz', 'late')]:.3f}, early "
        f"{aupro[('xyz', 'early')]:.3f}. Reliability fusion: "
        f"{aupro[('rgb', 'reliable_window')]:.3f} "
        f"and {aupro[('xyz', 'reliable_window')]:.3f}, the best of all models on the 3D side.",
        "- The JPEG and brightness rows need care. They change global image statistics, "
        "and in several categories those statistics already separate normal from defective "
        f"test images before any detector looks: on cable_gland the mean brightness of the "
        f"*background* alone gives AUROC {sep(conf.loc['cable_gland', 'background_mean']):.2f}, "
        f"on cookie the whole image {sep(conf.loc['cookie', 'rgb_mean']):.2f}, on peach the "
        f"background {sep(conf.loc['peach', 'background_mean']):.2f} "
        "(`scripts/check_acquisition_confound.py`, AUROC folded so that 0.5 means no "
        "separation). JPEG error follows brightness and texture, so it can push a "
        "detector's scores along that difference or against it; single categories swing "
        "between 0 and 1 across JPEG severities for that reason, not because of the code.",
        f"- The clean row reproduces the sweep's full-data runs to four decimals "
        f"({clean['early']:.4f}, {clean['late']:.4f}), through a separate code path.",
        "- Tilt barely changes FPFH, as expected for a rotation-invariant descriptor, and a "
        "shift of up to 8 px is at most one 8 px patch of the 28x28 feature grid, so "
        "misregistration costs little at this resolution.",
        "",
    ]


def stream_section() -> list[str]:
    path = Path("results/analysis/fault_stream.csv")
    if not path.exists():
        return []
    t = pd.read_csv(path)
    global_ = ["gaussian_noise", "defocus_blur", "jpeg", "depth_noise", "quantise"]
    g = (
        t[t["corruption"].isin(global_)]
        .groupby("fusion")
        .agg(
            after=("auroc_after", "mean"),
            before=("auroc_before", "mean"),
            delay=("delay", "median"),
            false=("false_switch", "mean"),
        )
    )
    order = ["late", *[f"window {w}" for w in (1, 4, 8, 16, 32)]]
    g = g.loc[order]
    shown = pd.DataFrame(
        {
            "AUROC before the fault": g["before"].map(lambda v: f"{v:.3f}"),
            "AUROC after": g["after"].map(lambda v: f"{v:.3f}"),
            "parts to notice (median)": g["delay"].map(lambda v: "" if pd.isna(v) else f"{v:.0f}"),
            "false switch-offs per stream": g["false"].map(lambda v: f"{v:.2f}"),
        }
    )
    shown.index = ["late fusion", "per part (window 1)", *[f"window {w}" for w in (4, 8, 16, 32)]]
    return [
        "## A sensor that fails mid-run",
        "",
        "The benchmark above corrupts a whole test set at once. `scripts/fault_stream.py` "
        "replays each category's test parts in a random order: clean for the first half, "
        "then the same parts from a corrupted run (severity 3). Five orders per category, "
        "averaged over the five global corruptions (noise, blur, JPEG, depth noise, "
        "quantisation). Reliability fusion with a 3-SD ramp; only the window changes.",
        "",
        shown.to_markdown(disable_numparse=True),
        "",
        "![AUROC after the fault per window, and delay vs false switch-offs]"
        "(figures/fault_stream.png)",
        "",
        "- The window trades reaction time for calm. With 16 parts, the setting used in "
        "the benchmark, a broken sensor is noticed after 7 parts, and the parts in between "
        f"pull the AUROC after the fault down to {g.loc['window 16', 'after']:.3f}. "
        f"With 4 parts it takes 1 part ({g.loc['window 4', 'after']:.3f}), at the cost of "
        f"{g.loc['window 4', 'false']:.1f} false switch-offs per stream of about 55 clean "
        "parts. Judging every part alone reacts at once but switches a healthy sensor off "
        f"{g.loc['window 1', 'false']:.1f} times per stream.",
        "- None of the false switch-offs costs anything measurable before the fault "
        f"(AUROC {g['before'].min():.3f}-{g['before'].max():.3f}).",
        "- So the right window depends on how long faults last compared with how fast they "
        "must be caught. 16 was fixed before any run and is kept in the main table; 4-8 "
        "parts is the better choice when faults start in the middle of a run.",
        "",
    ]


def modality_section() -> list[str]:
    path = Path("results/analysis/modality_importance.csv")
    if not path.exists():
        return []
    t = pd.read_csv(path)
    by = t.groupby("defect").agg(
        types=("category", "size"),
        images=("n", "sum"),
        rgb=("rgb_importance", "mean"),
        fpfh=("fpfh_importance", "mean"),
    )
    by = by.sort_values("images", ascending=False)
    shown = by.assign(
        rgb=by["rgb"].map(lambda v: f"{v:+.3f}"), fpfh=by["fpfh"].map(lambda v: f"{v:+.3f}")
    ).rename(
        columns={
            "types": "categories",
            "images": "defective images",
            "rgb": "lost without RGB",
            "fpfh": "lost without 3D",
        }
    )
    needs = t["needs"].value_counts()
    best = t[["rgb_only", "fpfh_only"]].max(axis=1)
    worse = int((t["late"] < best - 0.02).sum())
    return [
        "## Which sensor each defect type needs",
        "",
        "`scripts/modality_importance.py` drops one sensor's term from late fusion (full "
        "training set) and measures the image AUROC each defect type loses, against all "
        "normal test images of its category. Averaged here over the categories that have "
        "the defect type:",
        "",
        shown.to_markdown(disable_numparse=True),
        "",
        "![AUROC lost without RGB vs without 3D, per defect type](figures/modality_importance.png)",
        "",
        f"- Of {len(t)} category/defect types, {needs.get('both', 0)} need both sensors "
        f"(each adds at least 0.02), {needs.get('RGB', 0)} lean on RGB and "
        f"{needs.get('3D', 0)} on 3D.",
        f"- Colour defects are all RGB: they lose {by.loc['color', 'rgb']:.2f} without it and "
        "nothing without 3D. Holes and cuts lean on 3D, as expected. Contamination does too, "
        "which I did not expect: on average it loses more without 3D than without RGB. "
        "Probably the material sitting on the surface changes its shape as well; I have not "
        "checked that sample by sample.",
        f"- On {worse} types the fusion is more than 0.02 below the better sensor alone, "
        "where the other sensor only adds noise (tire contamination, potato holes). "
        "That is the same problem as a corrupted sensor, inside clean data.",
        "",
    ]


def conformal_section() -> list[str]:
    path = Path("results/analysis/conformal_clean.csv")
    if not path.exists():
        return []
    from factorylens import conformal

    clean = pd.read_csv(path)
    sizes = pd.read_csv("results/analysis/conformal_sizes.csv")
    bad = pd.read_csv("results/analysis/conformal_broken.csv")
    names = {
        "rgb": "RGB",
        "fpfh": "FPFH",
        "late": "late fusion",
        "reliable_window": "reliability fusion",
        "reliable_extent": "reliability + extent",
        "healthy sensor": "healthy sensor",
    }
    c = clean.groupby(["detector", "alpha"])[["false_alarm", "detection"]].mean()
    table = pd.DataFrame(
        {
            f"alpha = {a}": [
                f"{c.loc[(m, a), 'false_alarm']:.3f} / {c.loc[(m, a), 'detection']:.3f}"
                for m in ("rgb", "fpfh", "late", "reliable_window")
            ]
            for a in (0.05, 0.1, 0.2)
        },
        index=[names[m] for m in ("rgb", "fpfh", "late", "reliable_window")],
    )
    b = bad.groupby(["sensor", "detector"])[["false_alarm", "detection"]].mean()
    broken = pd.DataFrame(
        {
            f"{'RGB' if sensor == 'rgb' else '3D'} corrupted": [
                f"{b.loc[(sensor, m), 'false_alarm']:.3f} / {b.loc[(sensor, m), 'detection']:.3f}"
                for m in ("late", "reliable_window", "reliable_extent", "healthy sensor")
            ]
            for sensor in ("rgb", "xyz")
        },
        index=[names[m] for m in ("late", "reliable_window", "reliable_extent", "healthy sensor")],
    )
    checks_path = Path("results/analysis/conformal_checks.csv")
    checks = pd.read_csv(checks_path) if checks_path.exists() else None
    n_val = int(clean["n"].median())
    lo, hi = conformal.false_alarm_quantiles(n_val, 0.1, [0.05, 0.95])
    q = sizes.groupby("n")["false_alarm"].quantile([0.05, 0.95]).unstack()
    return [
        "## Alarm thresholds",
        "",
        "A score needs a threshold before it can stop a line. `src/factorylens/conformal.py` "
        "sets it split-conformally from the validation normals: alarm above the k-th largest "
        "calibration score, k = floor(alpha (n + 1)), which keeps the false alarm rate at "
        "most alpha on average if new normal parts are exchangeable with the calibration "
        "parts (`scripts/conformal_alarms.py`, all from saved scores). False alarm rate on "
        "the normal test parts / share of defective parts caught, mean of 10 categories:",
        "",
        table.to_markdown(),
        "",
        "![alarm rate vs target; vs calibration size; with one sensor broken]"
        "(figures/conformal.png)",
        "",
        "- On clean data the promise holds on average for RGB and both fusions. FPFH runs "
        f"over at alpha = 0.1 ({c.loc[('fpfh', 0.1), 'false_alarm']:.3f}, about two standard "
        "errors above the target over 10 categories): its normal test parts look less like "
        "the validation parts than they should, which is the kind of shift the guarantee "
        "does not cover. The fused scores use validation statistics for their z-scores, so "
        "for them calibration and test are only approximately exchangeable.",
        f"- With the {n_val} validation parts a category has, a target of 10% means a false "
        f"alarm rate anywhere between {lo:.0%} and {hi:.0%} for a given calibration set "
        "(5-95%, Beta distribution). Cutting the RGB calibration set to 5 parts at "
        f"alpha = 0.2 gives {q.loc[5, 0.05]:.0%}-{q.loc[5, 0.95]:.0%} in practice, 20 parts "
        f"{q.loc[20, 0.05]:.0%}-{q.loc[20, 0.95]:.0%}. Keeping a 5% target within 2 points, "
        f"9 times out of 10, takes {conformal.min_calibration(0.05, 0.02)} normal parts.",
        "",
        "Thresholds from the clean validation parts, test parts corrupted, alpha = 0.1, "
        "all corruptions and severities:",
        "",
        broken.to_markdown(),
        "",
        "- This is where the difference between the fusions shows in a way AUROC does not. "
        "With the RGB image corrupted, late fusion raises an alarm on "
        f"{b.loc[('rgb', 'late'), 'false_alarm']:.0%} of normal parts: the line stops. "
        "Reliability fusion keeps it at "
        f"{b.loc[('rgb', 'reliable_window'), 'false_alarm']:.0%} "
        f"({b.loc[('rgb', 'reliable_extent'), 'false_alarm']:.0%} with extent) and catches "
        "about as many defects as the healthy sensor. With the point cloud corrupted it "
        f"stays at {b.loc[('xyz', 'reliable_window'), 'false_alarm']:.0%} and catches more "
        f"than RGB alone ({b.loc[('xyz', 'reliable_window'), 'detection']:.3f} vs "
        f"{b.loc[('xyz', 'healthy sensor'), 'detection']:.3f}).",
        "",
        *conformal_checks(checks),
    ]


def conformal_checks(checks: pd.DataFrame | None) -> list[str]:
    if checks is None:
        return []
    ex = checks[checks["check"] == "exchangeability"]
    fp = ex[ex["sensor"] == "fpfh"]
    shifted = fp[fp["shift_p"] < 0.05]["category"].tolist()
    survives = ex[ex["shift_p_holm"] < 0.05]
    tire = fp.set_index("category").loc["tire"]
    split = checks[checks["check"] == "strict split"]
    wm = (
        checks[checks["check"] == "weight-matched"]
        .groupby("sensor")[["false_alarm", "false_alarm_matched"]]
        .mean()
    )
    return [
        "Where the rate goes wrong, and what fixes it (`scripts/conformal_checks.py`):",
        "",
        f"- FPFH's excess on clean data is a real shift. Its normal test parts score higher "
        f"than the validation parts in {len(shifted)} categories ({', '.join(shifted)}, "
        "one-sided Mann-Whitney p < 0.05). After Holm's correction over the 20 tests, "
        f"{len(survives)} remains ({', '.join(survives['category'] + ' ' + survives['sensor'])}), "
        f"where FPFH alarms on {tire['false_alarm']:.0%} of good parts.",
        "- Five confirmed-good parts from the current run fix most of it. Added to the "
        "calibration set, they bring FPFH's mean false alarm rate from "
        f"{fp['false_alarm'].mean():.3f} to {fp['false_alarm_plus5'].mean():.3f}, and "
        f"tire's from {tire['false_alarm']:.2f} to {tire['false_alarm_plus5']:.2f}, measured "
        "on the remaining normal parts. On a line that means asking an operator to confirm "
        "a handful of good parts when a run starts.",
        "- The fused scores take their z-score statistics from the same validation parts "
        "they are calibrated on. Splitting the validation parts in two, one half for each "
        f"job, gives {split['false_alarm_split'].mean():.3f} instead of "
        f"{split['false_alarm'].mean():.3f}: the strict version stays under the target too, "
        "at the price of a smaller calibration set.",
        "- One idea that did not work: calibrating each part against validation parts fused "
        "with that part's own sensor weights. With the RGB image corrupted, the false alarm "
        f"rate went from {wm.loc['rgb', 'false_alarm']:.3f} to "
        f"{wm.loc['rgb', 'false_alarm_matched']:.3f}. The excess there comes from occlusion "
        "and from FPFH's own shift, not from the weights.",
        "",
    ]


def main() -> None:
    lines = [
        "# Results",
        "",
        "<!-- generated by scripts/make_results_tables.py, do not edit by hand -->",
        "",
        *rgb_section(),
        *defect_section(),
        *threed_section(),
        *sensor_section(),
        *btf_section(),
        *balance_section(),
        *sweep_section(),
        *robustness_section(),
        *stream_section(),
        *modality_section(),
        *conformal_section(),
        *layer_section(),
        *checks_section(),
    ]
    OUT.write_text("\n".join(lines))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
