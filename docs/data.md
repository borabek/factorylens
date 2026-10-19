# Data

Nothing in this repo contains dataset files. Everything goes under `data/` (git-ignored) and is downloaded with the scripts below.

## MVTec 3D-AD (main benchmark)

10 object categories scanned with an industrial structured-light sensor. Each sample is an RGB image plus an `xyz` map (float32, metres) aligned pixel for pixel with it. Training and validation contain only defect-free samples. Test samples come with pixel masks, and the mask value encodes the defect type (`class_ids.json`).

| category | train | val | test | image size (W×H) | defect types |
|---|---|---|---|---|---|
| bagel | 244 | 22 | 110 | 800×800 | contamination, crack, hole, combined |
| cable_gland | 223 | 23 | 108 | 400×400 | bent, cut, hole, thread |
| carrot | 286 | 29 | 159 | 800×800 | contamination, crack, cut, hole, combined |
| cookie | 210 | 22 | 131 | 500×500 | contamination, crack, hole, combined |
| dowel | 288 | 34 | 130 | 400×400 | bent, contamination, cut, combined |
| foam | 236 | 27 | 100 | 900×900 | color, contamination, cut, combined |
| peach | 361 | 42 | 132 | 600×600 | contamination, cut, hole, combined |
| potato | 300 | 33 | 114 | 800×800 | contamination, cut, hole, combined |
| rope | 298 | 33 | 101 | 900×400 | contamination, cut, open |
| tire | 210 | 29 | 112 | 600×800 | contamination, cut, hole, combined |
| **total** | **2656** | **294** | **1197** | | |

Image sizes are from the first training sample of each category.

Pixels without a 3D point are stored as zeros, mostly background the sensor could not see. Averaged over all samples, valid coverage ranges from 61% (cable_gland) to 99% (rope).

### What the data looks like

![one defective test sample per category](figures/samples.png)

Numbers from `scripts/eda_mvtec3d.py`, which writes one row per sample to `results/eda/mvtec3d_samples.csv`. Background removal uses the defaults: 2 mm plane threshold, then blobs under 0.02% of the image dropped.

| category | valid 3D points | object height (median) | defect area (median) | defect area (min) | defect px with no 3D point | defect px on the table |
|---|---|---|---|---|---|---|
| bagel | 83% | 39 mm | 0.39% | 0.07% | 12.1% | 0.3% |
| cable_gland | 61% | 38 mm | 0.35% | 0.05% | 9.0% | 0.1% |
| carrot | 98% | 17 mm | 0.20% | 0.05% | 8.6% | 0.4% |
| cookie | 90% | 20 mm | 1.16% | 0.26% | 13.3% | 4.8% |
| dowel | 99% | 9 mm | 0.30% | 0.04% | 3.9% | 13.0% |
| foam | 94% | 43 mm | 0.21% | 0.04% | 4.7% | 0.1% |
| peach | 84% | 34 mm | 0.29% | 0.10% | 12.9% | 0.1% |
| potato | 91% | 35 mm | 0.18% | 0.04% | 3.5% | 0.0% |
| rope | 99% | 12 mm | 0.56% | 0.04% | 10.0% | 7.2% |
| tire | 85% | 113 mm | 0.18% | 0.04% | 3.0% | 0.0% |

The last two columns are means over defective test samples (948 in total).

Things that matter later:

- **Defects are small.** The median defect covers 0.29% of the image, and about 10% of defective samples have under 0.1%. At a 224×224 input that median is roughly 145 pixels, and the smallest defects are a handful. Image-level AUROC can hide a lot here, which is why pixel-level metrics (AUPRO) are part of the plan.
- **The 3D sensor often misses exactly the defect.** On bagel, cookie and peach, 12–13% of defect pixels have no 3D point at all; cracks and holes are where structured light fails. A geometry-only model has nothing to score there, and the RGB image does.
- **Some defects sit off the object.** On dowel and rope, 7–13% of defect pixels end up as background: the table showing through a cut, or contamination lying next to the object. Plane removal drops them by design. This is the reason the threshold is 2 mm and not 5 mm (see `scripts/check_background_removal.py`).
- **Object height varies 10×.** Dowel and rope are about 1 cm tall, tire over 11 cm. Any fixed 3D noise level in the corruption benchmark is relatively much harsher on thin objects, so severities will be expressed in millimetres and reported per category.

![defect area per test sample](figures/defect_sizes.png)

![share of defect pixels without object geometry](figures/defects_off_object.png)

```bash
scripts/download_mvtec3d.sh data/raw/mvtec3d
uv run factorylens data check data/raw/mvtec3d --deep
uv run factorylens data prepare data/raw/mvtec3d   # ~1 min, 0.7 GB
```

`data prepare` writes every split at 224×224 to `data/processed/224px-<hash>/`:
- RGB is resized bilinearly.
- xyz and the masks are resized nearest-neighbour, after background removal at full resolution.
- `<hash>` comes from the preprocessing settings, so a changed threshold gets a fresh directory instead of stale arrays.

Reading bagel's 244 training samples from the cache takes 0.2 s, against 10 s from the raw files.

The full-archive link on the MVTec page was dead when I downloaded the data (2026-10-17), and so were the per-category links for carrot, cookie, foam and peach. I took those four from the Kaggle mirror [`gengchensun/mvtec-3d-ad`](https://www.kaggle.com/datasets/gengchensun/mvtec-3d-ad). The mirror's bagel folder is byte-identical to the official one, and the deep check passes on every category.

Licence: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), so non-commercial use only. The figures above are derived from the dataset and fall under the same licence.

> Bergmann, Jin, Sattlegger, Steger. *The MVTec 3D-AD Dataset for Unsupervised 3D Anomaly Detection and Localization.* VISAPP 2022.

## Future Factories analog dataset (secondary study)

10 Hz signals from a rocket-assembly testbed at the University of South Carolina: four robots (gripper load and position, six joint angles each), conveyor drive temperatures and stoppers, safety I/O and a cycle counter. The data covers about 29 hours on 11–12 December 2023, and cycles are labelled as normal or by which parts were removed from the tray.

```bash
uvx kaggle datasets download ramyharik/ff-2023-12-12-analog-dataset --unzip -p data/raw/ff_analog
```

Licence: CC BY-SA 4.0.

See [research_plan.md](research_plan.md) for why the multimodal (image) version of this dataset is not used for the main comparison.

> Harik et al. *Analog and Multi-modal Manufacturing Datasets Acquired on the Future Factories Platform.* arXiv:2401.15544, 2024.
