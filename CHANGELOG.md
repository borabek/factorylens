# Changelog

## 1.0.0 (2026-10-29)

First release. Everything below was built and evaluated on MVTec 3D-AD, with the Future Factories analog data as a smaller sensor-only study.

**Detectors and fusion**
- PatchCore on RGB, FPFH on point clouds, early fusion (concatenated, with the two halves balanced on the training data) and late fusion.
- Reliability-aware late fusion: a label-free health signal per sensor, judged over the last 16 parts, and a weighted Stouffer combination. Optional extent signal for local corruptions.
- Split-conformal alarm thresholds with a stated false alarm rate.

**Benchmarks**
- A reproduction of Back to the Feature: both single-sensor halves reproduce, the fusion does not, and feature scale explains why.
- A k-shot sweep (1 to all training images, 5 draws) for all four detectors.
- FactoryLens-C: 10 one-sensor corruptions at 5 severities and a missing sensor for each, seeded and shared as code.
- Mid-run sensor failure replay, conformal checks, a lighting control and per-defect modality importance.
- CPU latency and memory: per stage, against coreset size and thread count, plus an ONNX export of the backbone.

**Using a model**
- `factorylens fit / inspect / serve`: one file per product with banks, fusion statistics and the alarm threshold, loaded with `weights_only=True`; optional float16 banks.
- FastAPI service (`/inspect`, `/status`, `/reset`, `/health`) and a Gradio demo. A precomputed version runs as a static Hugging Face Space.
- Model card and a write-up of how the inspector would sit on a production line.

**Paper**
- A workshop-style paper in `paper/`. Every table and every number quoted in the text is generated from the result files, and CI fails if they drift.
