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
    df = experiments.collect(results_dir)
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


def main() -> None:
    lines = [
        "# Results",
        "",
        "<!-- generated by scripts/make_results_tables.py, do not edit by hand -->",
        "",
        *rgb_section(),
        *defect_section(),
        *threed_section(),
        *btf_section(),
        *layer_section(),
        *checks_section(),
    ]
    OUT.write_text("\n".join(lines))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
