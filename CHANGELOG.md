# Changelog

## 1.2.2 (2026-11-02)

- The paper's conclusion said the reliability signal "can sit on top of learned fusion models". That is untested, and for detectors that predict one sensor from the other it probably can't say which sensor broke, because every map depends on both. The paper now says so.

## 1.2.1 (2026-11-02)

- A negative result in the paper and on the results page: with the camera corrupted, five confirmed-good parts bring false alarms from 18.5% to 13.2% and defects caught from 57.8% to 52.4%, which misses the bar set in advance (at most 12%, at least 55%). The paper's conclusion said confirmed parts "repaired the alarm rate"; that holds for a shift between validation and test parts, not for a broken sensor, and it now says so.
- CI: pull requests run on Linux with all extras, so the FPFH tests run there too, and a broken open3d install fails loudly. macOS has its own workflow after merges and weekly, because GitHub sometimes assigns no runner and cancels the job, which blocked merges.

## 1.2.0 (2026-11-02)

- `fit --fault-test`: an optional sequential test for sensor faults instead of the median window (`conformal.FaultTest`, conformal martingales on the health signal). On 47 conditions it did not see while its settings were fixed, it notices a fault in 3 parts instead of 7 and lifts the AUROC after the fault from 0.788 to 0.801 (p = 0.002). It catches fewer defects at the same threshold, and a flag stays up until a reset. The window stays the default, so published numbers don't move.

Found by testing the installed service end to end, with real scans and with what a broken sensor could send:

- A dead 3D sensor on a raw scan (zeros, NaN or infinite everywhere) made every request fail with a 500, so the window never filled and `sensor_alert` never fired. It now goes through as a part with no points. With one sensor dead throughout, the fused AUROC on raw bagel and tire scans equals the other sensor's alone. The benchmark corrupts preprocessed data, which is why it never hit this; its numbers don't change.
- NaN and infinite values count as missing points, as zeros do.
- `uv sync --extra serve` gave a service whose `/health` said ok and whose `/inspect` always failed, and `--extra demo` gave a demo that couldn't start. The demo extra now brings matplotlib, the docs name the extras needed together, `make install` includes the 3D features, and `serve` and `demo` say what is missing before starting.
- `alpha` outside 0-1 gave a wrong threshold without a word (1.5 alarmed above the median), and `--coreset-ratio 0` was ignored. Both are refused now. `fit` warns when there are too few validation parts for `alpha`, so the model would never alarm.
- Refused instead of answered wrongly: integer depth, parts of the wrong size, a non-uint8 image. A file that isn't a model gets a clear error, and the CLI reports bad files without a traceback.
- `/health` calls the version that fitted the model `fitted_with`.

## 1.1.0 (2026-10-29)

The inspector watches its own reliability on a line.

- `sensor_alert` is `"rgb+3d"` when both sensors look off. Both are still kept at full weight, but the alert no longer stays empty in exactly the case where neither sensor can be trusted.
- `GET /status` checks the alarm rate over the last 100 parts against the range expected from the calibration set (beta-binomial, 99%), so drift in normal parts shows up on a line monitor.
- `Inspector.confirm_good` and `POST /confirm` add a part an operator found good to the calibration set. Every decision now carries a `part` number for this.

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
