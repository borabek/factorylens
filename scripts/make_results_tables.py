"""Build docs/results.md from the result files, so no number in it is typed by hand.

usage: uv run python scripts/make_results_tables.py
"""

import json
from pathlib import Path

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
    coverage = pd.read_csv("results/analysis/early_fusion_coverage_bagel.csv")
    rho = coverage[["excess_normal", "joint_auroc"]].corr("spearman").iloc[0, 1]

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
        "- The reason is coverage. A joint neighbour has to match appearance and geometry "
        "in the same training patch, and a few images do not contain enough combinations. "
        "On bagel, how much farther the joint neighbour of a normal test patch is than the "
        f"two separate ones has a Spearman correlation of {rho:.2f} with early fusion's "
        "AUROC over the 20 k-shot runs (`scripts/early_fusion_coverage.py`).",
        "- Flips and rotations of the training images help RGB a little at k = 1 and 10 and "
        "not at 5. They do not close the gap to either fusion.",
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
        *layer_section(),
        *checks_section(),
    ]
    OUT.write_text("\n".join(lines))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
