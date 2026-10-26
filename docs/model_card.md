# Model card: FactoryLens inspector

What `factorylens fit` produces, what it is for, and where it should not be trusted. The numbers come from the results pages linked below, not from new runs.

## What it is

One file per product. It holds:

- two memory-bank anomaly detectors: PatchCore features from an ImageNet WideResNet-50 for the RGB image, and FPFH descriptors for the point cloud;
- reliability-aware late fusion of the two (`src/factorylens/fusion.py`);
- a split-conformal alarm threshold.

It is fitted on normal parts only. No defect labels are used anywhere in fitting, calibration or the reliability signals.

## Intended use

- Research and teaching: comparing RGB, 3D and fused detection, and studying what happens when one sensor degrades.
- A starting point for a line that has a few hundred normal parts per product and wants an alarm with a stated false alarm rate. The inspector is not validated on any real production line.

## Training data

- MVTec 3D-AD, one category at a time: 210–361 normal training parts and 22–42 normal validation parts per category.
- The licence is CC BY-NC-SA 4.0. A model fitted on it carries features of the dataset in its memory banks, so it should be treated as covered by the same licence: no commercial use, share alike. No fitted models are distributed with this repository; `models/` is in `.gitignore`.

## Performance (MVTec 3D-AD test sets, mean of 10 categories)

| | value | source |
|---|---|---|
| image AUROC, clean | 0.876 | [results.md](results.md), robustness |
| image AUROC, RGB corrupted (FactoryLens-C) | 0.788 | same |
| image AUROC, 3D corrupted | 0.832 | same |
| false alarm rate at alpha = 0.1, clean | 0.089 | results.md, alarm thresholds |
| false alarm rate at alpha = 0.1, RGB corrupted | 0.185 | same |
| share of defective parts caught at alpha = 0.1, clean | 0.668 | same |

Early fusion scores higher on clean data (0.910) but collapses when a sensor degrades. That trade-off is the reason the inspector uses reliability fusion. Per-category numbers vary a lot, from 0.70 (tire) to 0.98 (dowel, rope): see results.md.

## Known limitations

- **Local corruptions.** An occluder or smudge on the camera looks like a defect, and the default reliability signal does not catch it (occlusion AUROC 0.745 vs 0.794 for the healthy sensor). The optional extent signal fixes that at a cost of 0.007 AUROC on clean data.
- **Reaction time.** Sensor health is judged over the last 16 parts. After a sensor fails, it takes a median of 7 parts to down-weight it. A shorter window reacts faster but switches healthy sensors off more often (results.md, "A sensor that fails mid-run").
- **Calibration size.** With about 30 validation parts, a 10% false alarm target can land anywhere between 3% and 20% for a given calibration set. The guarantee also assumes new normal parts look like the validation parts. That fails for FPFH on tire, where the 3D detector on its own alarms on 44% of good parts. Adding five confirmed-good parts from the current run to the calibration set brings it down to 16%.
- **Dataset artefacts.** In several MVTec 3D-AD categories, normal and defective test images differ in lighting ([data.md](data.md)). A control run showed the detectors here do not exploit it, but the absolute numbers are from this dataset only.
- **Hardware.** About 105 ms per part on an M4 laptop (MPS). This was not measured on industrial or edge hardware.

## Ethical considerations

The inspector decides whether a manufactured part looks anomalous. It processes no personal data. The main risk is over-trust: an alarm rate promised on 30 calibration parts is a noisy estimate, and a silently degraded sensor changes what the model sees. Both are reported per part (`p_value`, `weight_rgb`, `weight_3d`, `sensor_alert`), so an operator can see when not to rely on the decision.
