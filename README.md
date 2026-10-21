# FactoryLens

[![ci](https://github.com/borabek/factorylens/actions/workflows/ci.yml/badge.svg)](https://github.com/borabek/factorylens/actions/workflows/ci.yml)

Anomaly detection for industrial inspection using an RGB camera and a 3D sensor together, with a focus on what happens when one of the two sensors gets noisy or fails.

This is a work in progress (started October 2026). The research question, the dataset decision and the hypotheses are in [docs/research_plan.md](docs/research_plan.md).

## Where things are

**First results** (full tables in [docs/results.md](docs/results.md)), image AUROC averaged over the 10 categories:

| | this repo | published (Back to the Feature) |
|---|---|---|
| RGB only, PatchCore (5 seeds) | 0.799 ± 0.005 | 0.785 |
| 3D only, FPFH | 0.793 | 0.753 |
| RGB + FPFH concatenated, as in BTF (their preprocessing) | 0.801 | 0.873 |
| RGB + FPFH, halves balanced on the training data (their preprocessing) | **0.923** | |

- The two sensors fail in different places. RGB is near chance on potato (0.56), where 3D gets 0.95. 3D struggles on cable_gland and tire, where RGB is fine.
- Both single-sensor baselines reproduce. The published fusion does not: concatenating the two feature sets as BTF does lands *below* the better single sensor on 8 of 10 categories.
- The FPFH half of the joint features is 12–47 times wider than the RGB half, so the distances are almost entirely 3D. Rescaling it on the training data alone lifts the same fusion to 0.923, above both sensors on 9 of 10 categories and above the published number.

Findings from the data itself, each with the script that produces it:

- **The obvious multimodal dataset is confounded with time.** In Future Factories, almost every anomalous cycle in the camera period comes before every normal one, and elapsed time alone gives an AUROC of 0.986 (`scripts/ff/confound_check.py`). That is why the main study uses MVTec 3D-AD.
- **The 3D sensor is often blind exactly at the defect.** On bagel, cookie and peach, 12–13% of defect pixels have no 3D point (`scripts/eda_mvtec3d.py`). Only the camera can see those.
- **A standard preprocessing threshold deletes defects on thin objects.** BTF's 5 mm background cut removes 38% of the annotated defect pixels on dowel. This repo uses 2 mm (`scripts/check_background_removal.py`).
- **A sensor shortcut that flips sign across days.** In the FF robot data, R01's gripper load separates normal and missing-part cycles on day 1, but only because it drifts, and on day 2 the relationship reverses. R03 shows the real signature of a missing nose cone on both days (`scripts/ff/figures.py`).

![label blocks over time and the period with camera images](docs/figures/ff_timeline.png)

## Usage

```bash
uv run factorylens data check data/raw/mvtec3d --deep   # counts, file pairing, sizes
uv run factorylens data prepare data/raw/mvtec3d        # cached 224 px samples, ~1 min
uv run factorylens data kshot                           # rewrite configs/kshot.json
uv run factorylens ff blocks                            # FF label blocks and day split
uv run factorylens rgb run --category bagel             # PatchCore on one category
uv run factorylens results table results/rgb            # mean metrics over result files
uv run python scripts/make_results_tables.py            # regenerate docs/results.md
```

## Setup

Needs [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
make install   # creates .venv, installs pre-commit hooks
make check     # lint, type check, tests with coverage
```

## Data

Download instructions are in [docs/data.md](docs/data.md). The code downloads public datasets; nothing is redistributed here. The same page has counts, a data card and the figures.

- [MVTec 3D-AD](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad) (CC BY-NC-SA 4.0), the main benchmark
- [Future Factories analog dataset](https://www.kaggle.com/datasets/ramyharik/ff-2023-12-12-analog-dataset) (CC BY-SA 4.0), used for a smaller sensor-only study

## License

Code is MIT. The datasets keep their own licences.
