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
    df = experiments.collect("results/rgb")
    df = df[df["k"].isna()].set_index("category").sort_index()
    btf = reference("RGB iNet")
    rows = []
    for cat, r in df.iterrows():
        rows.append(
            {
                "category": cat,
                "image AUROC [95% CI]": f"{r.image_auroc:.3f} [{r.image_auroc_ci_low:.2f}, "
                f"{r.image_auroc_ci_high:.2f}]",
                "BTF image": f"{btf.loc[cat, 'image_auroc']:.3f}",
                "pixel AUROC": f"{r.pixel_auroc:.3f}",
                "BTF pixel": f"{btf.loc[cat, 'pixel_auroc']:.3f}",
                "AUPRO": f"{r.aupro:.3f}",
                "BTF AUPRO": f"{btf.loc[cat, 'aupro']:.3f}",
            }
        )
    mean = df[METRICS].mean()
    bmean = btf.loc[df.index, METRICS].mean()
    rows.append(
        {
            "category": "**mean**",
            "image AUROC [95% CI]": f"**{mean.image_auroc:.3f}**",
            "BTF image": f"{bmean.image_auroc:.3f}",
            "pixel AUROC": f"**{mean.pixel_auroc:.3f}**",
            "BTF pixel": f"{bmean.pixel_auroc:.3f}",
            "AUPRO": f"**{mean.aupro:.3f}**",
            "BTF AUPRO": f"{bmean.aupro:.3f}",
        }
    )
    table = pd.DataFrame(rows).to_markdown(index=False)
    diff = (df["image_auroc"] - btf.loc[df.index, "image_auroc"]).abs()
    inside = (
        (btf.loc[df.index, "image_auroc"] >= df["image_auroc_ci_low"])
        & (btf.loc[df.index, "image_auroc"] <= df["image_auroc_ci_high"])
    ).sum()
    weakest = df["image_auroc"].nsmallest(3)
    foam_gap = df.loc["foam", "aupro"] - btf.loc["foam", "aupro"]
    notes = [
        "Reading the numbers:",
        "",
        f"- The reimplementation lands close to the published one. BTF's image AUROC falls inside "
        f"our 95% CI on {inside} of {len(df)} categories, and the largest gap is "
        f"{diff.max():.3f} ({diff.idxmax()}).",
        f"- RGB alone is weakest on {', '.join(weakest.index)} "
        f"({', '.join(f'{v:.2f}' for v in weakest)}). These are organic objects where colour and "
        "texture vary a lot between normal samples, which probably makes a dent or a cut look like "
        "normal variation to an ImageNet feature. This is where 3D should help, if it helps "
        "anywhere.",
        f"- Localisation on foam is {foam_gap:.2f} AUPRO above BTF's number. I have not found the "
        "reason yet (candidates: blur sigma, resizing, coreset details), so treat that gap as "
        "unexplained rather than as an improvement.",
        "",
    ]
    return [
        "## RGB only (PatchCore)",
        "",
        "Full training set, WideResNet-50 layer2 + layer3, 10% coreset, 224 px. "
        "BTF columns are the *RGB iNet* row of Back to the Feature, which uses the same "
        "recipe; their numbers are copied from the paper (`configs/reference/btf.json`).",
        "",
        table,
        "",
        f"<sub>{provenance(df.reset_index(), 'results/rgb')}. CI: 1000 bootstrap "
        "resamples of the test images, stratified by label.</sub>",
        "",
        *notes,
    ]


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
        *layer_section(),
    ]
    OUT.write_text("\n".join(lines))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
