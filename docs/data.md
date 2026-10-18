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

On a first look at one training scan per category, between 66% (cable_gland) and 99% (rope) of pixels have a valid 3D point. The rest are zeros, mostly background the sensor could not see.

```bash
scripts/download_mvtec3d.sh data/raw/mvtec3d
uv run python scripts/check_mvtec3d.py data/raw/mvtec3d --deep
```

The full-archive link on the MVTec page was dead when I downloaded the data (2026-10-17), and so were the per-category links for carrot, cookie, foam and peach. I took those four from the Kaggle mirror [`gengchensun/mvtec-3d-ad`](https://www.kaggle.com/datasets/gengchensun/mvtec-3d-ad). The mirror's bagel folder is byte-identical to the official one, and the deep check passes on every category.

Licence: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), so non-commercial use only.

> Bergmann, Jin, Sattlegger, Steger. *The MVTec 3D-AD Dataset for Unsupervised 3D Anomaly Detection and Localization.* VISAPP 2022.

## Future Factories analog dataset (secondary study)

10 Hz signals from a rocket-assembly testbed at the University of South Carolina: four robots (gripper load and position, six joint angles each), conveyor drive temperatures and stoppers, safety I/O and a cycle counter. The data covers about 29 hours on 11–12 December 2023, and cycles are labelled as normal or by which parts were removed from the tray.

```bash
uvx kaggle datasets download ramyharik/ff-2023-12-12-analog-dataset --unzip -p data/raw/ff_analog
```

Licence: CC BY-SA 4.0.

See [research_plan.md](research_plan.md) for why the multimodal (image) version of this dataset is not used for the main comparison.

> Harik et al. *Analog and Multi-modal Manufacturing Datasets Acquired on the Future Factories Platform.* arXiv:2401.15544, 2024.
