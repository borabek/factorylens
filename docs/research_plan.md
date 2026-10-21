# Research plan

## Summary

The Future Factories (FF) multimodal dataset does not support a fair comparison of vision-only, sensor-only and fused anomaly detectors. Inside the period covered by camera images, the anomaly label is almost perfectly determined by when a cycle happened. The main study therefore moves to **MVTec 3D-AD**, where an RGB camera and an industrial 3D sensor capture the same objects. The FF *analog* data stays in the project as a smaller sensor-only case study about temporal confounding.

## What I checked in Future Factories (2023-12-12 release)

**Analog dataset (10 Hz, labelled)**
- 7 CSV files, 974,836 rows, 2023-12-11 13:04 to 2023-12-12 18:07 UTC.
- The cycle counter resets after interruptions, so I segmented physical cycles as runs of a constant counter value. This gives 325 full cycles of about 295 s each, which matches the dataset paper.
- Labels are per cycle: 161 normal, 59 `NoNose`, 51 `NoNose,NoBody2`, 54 `NoNose,NoBody2,NoBody1`.
- The labels come in six contiguous blocks:

| Block | Label | Cycles | UTC |
|---|---|---|---|
| 1 | normal | 23 | 11 Dec 13:04 – 15:36 |
| 2 | NoNose | 36 | 11 Dec 15:36 – 19:59 |
| 3 | NoNose, NoBody2 | 51 | 11 Dec 19:59 – 12 Dec 00:16 |
| 4 | NoNose, NoBody2, NoBody1 | 54 | 12 Dec 00:16 – 04:45 |
| 5 | normal | 138 | 12 Dec 04:45 – 16:14 |
| 6 | NoNose | 23 | 12 Dec 16:14 – 18:07 |

**Multimodal dataset (about 2 Hz, two cameras, no labels)**
- 6 Kaggle parts of 56–105 GB each, roughly 580 GB in total. Images are 1.7 MB PNGs in batches of 1000. Each batch has a `data_<n>.json` with timestamps, 44 sensor values and the two image paths.
- The timestamps are local time. The first batch starts 2023-12-11 11:50:52 and the last ends 2023-12-12 11:25:42. With the 4h59m51.1s offset that NSF-MAP used, this window is 16:50 UTC on day 1 to 16:25 UTC on day 2.
- The window misses all of block 1 and almost all of block 6. Inside it, every anomalous cycle except five comes before every normal cycle.

**Shortcut test** (`scripts/ff/confound_check.py`, cycle level)

| Predictor | All analog cycles | Image window only |
|---|---|---|
| hours since start | 0.737 | **0.986** |
| conveyor 1 VFD temperature | 0.507 | 0.600 |
| conveyor 2 VFD temperature | 0.597 | 0.502 |

<sub>Corrected on 2026-10-21. About 1% of the conveyor 2–4 temperature readings turned out to be corrupted values, from 1e-30 to 1e38 and many negative. They are now masked outside 40–140 °F. The first version of this table used them: conveyor 2 read 0.503 and 0.511. The conclusion does not change.</sub>

Elapsed time alone reaches an AUROC of 0.986 inside the image window. Any input that drifts with time could produce a high score without detecting a single missing part: room lighting, camera warm-up, a cable being nudged. A vision model, a sensor model and a fused model would all be scored on that confound, so the A/B/C comparison would not mean anything.

**Other options checked**
- **FF 2024-08-13 (V2):** an 8-hour normal run with no anomaly annotations (`Description` is empty), CC BY-NC-SA 4.0.
- **TaaS rocket assembly line:** a repackaging of the same FF recordings, so it has the same problem.
- **MulSen-AD** (RGB + infrared thermography + point cloud, CVPR 2025): a promising three-sensor dataset. Its licence is not stated, it ships as a single RAR on Hugging Face, and it has far less published tooling. I kept it as possible future work.

## Decision

**Main study: MVTec 3D-AD** (CC BY-NC-SA 4.0, about 13.2 GB, 10 categories).
- Each sample has an RGB image and an xyz map from an industrial structured-light sensor, aligned pixel to pixel. Training data is normal only, test defects have pixel masks, and the defect type is known.
- Published baselines (e.g. Back to the Feature, Horwitz & Hoshen 2022) let me check that my implementation is correct before I trust any new result.
- It is small enough for the M4 laptop with frozen pretrained features.

**Secondary case study: FF analog data, sensor only.**
- Normal cycles and `NoNose` cycles occur on both days (blocks 1–2 and 5–6). That allows a cross-day split: train on day 1, test on day 2.
- The case study reports the time-only baselines next to the real detectors, to show how easy it is to get a high score for the wrong reason.
- The dashboard replays these sensor streams in a "line monitor" view, clearly labelled as a separate dataset.

## Revised research questions

**Main question.** With limited normal training data, does fusing RGB images with 3D geometry detect and localise surface defects better than either sensor alone? How gracefully does each model degrade when one sensor is noisy or missing?

- **RQ1 Detection and localisation.**
  - Models: RGB-only (A), 3D-only (B) and fused (C).
  - Metrics: image-level AUROC, pixel-level AUROC and AUPRO@30%, per category, with bootstrap confidence intervals.
- **RQ2 Data efficiency.** The same comparison with k ∈ {1, 5, 10, 50, all} normal training samples per category, using 5 random subsets for each k.
- **RQ3 Robustness.** Depth noise, missing 3D points, RGB blur, brightness shifts and a fully missing modality. Compare naive fusion with normalised score fusion.
- **RQ4 Defect characterisation.** Which defect types each sensor catches and which it misses (e.g. colour-only contamination versus geometric holes and cuts), and whether the per-modality anomaly maps point at the annotated region.
- **Alarm thresholds.** Split-conformal thresholds set on validation normals, then the empirical false-alarm rate on test normals compared with the target.

## Hypotheses (written before running anything)

1. Fusion beats the better single modality on mean image-level AUROC. The gain is largest in categories where defects are mostly geometric but texture varies (e.g. foam, rope).
2. The advantage of fusion shrinks as k grows, because the RGB-only memory bank catches up with more data.
3. Naive feature concatenation degrades more than either single modality when one sensor is corrupted. Normalised late fusion degrades more gracefully.
4. Conformal thresholds keep the empirical false-alarm rate within about ±2 points of the target on most categories. Categories with very few validation normals will miss it.

If any of these turn out false, I will report them as false.
