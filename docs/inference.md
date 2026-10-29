# Using a fitted model

The research code fits and scores whole test sets. For a line, `factorylens.inspector.Inspector` packs one product's fitted model into a single file and inspects parts one at a time, in the order they arrive.

## Fit and save

```bash
uv run factorylens fit data/raw/mvtec3d --category bagel --alpha 0.1
```

This builds the RGB and FPFH memory banks from the training parts. On the validation parts it fits:

- the scaling for late fusion,
- the references for the reliability signals,
- a split-conformal alarm threshold for the target false alarm rate `alpha`.

Everything goes into `models/bagel.pt`, about 80 MB for bagel. Add `--half` to store the banks as float16: 40 MB, with scores moving by at most 0.006 on bagel, the same AUROC and every alarm decision unchanged. The ImageNet backbone weights are not stored; torchvision loads them as usual. Model files are loaded with `torch.load(weights_only=True)`, so a file can only contain tensors and plain values, not code.

## Inspect parts

```bash
uv run factorylens inspect models/bagel.pt part1.png part1.tiff part2.png part2.tiff
```

Each pair is an RGB image and an organised point cloud (H × W × 3 float TIFF in metres, as MVTec 3D-AD stores it), both at full resolution. The background plane is removed and both are resized exactly as in training. Each part gives one line of JSON:

| field | meaning |
|---|---|
| `score` | fused anomaly score, in standard deviations of the validation parts |
| `p_value` | conformal p-value against the validation parts |
| `alarm` | `score` above the threshold for `alpha` |
| `weight_rgb`, `weight_3d` | how much each sensor counted for this part (0–1) |
| `health_rgb`, `health_3d` | how far each sensor's anomaly map sits beyond anything seen on the validation parts, in their SDs; ≤ 0 looks normal |
| `score_rgb`, `score_3d` | the two single-sensor scores |
| `sensor_alert` | `"rgb"`, `"3d"` or `"rgb+3d"` when that sensor counted for less than half, or when both looked off (both are then kept at full weight); `null` otherwise |
| `part` | number of the part since the model was loaded, for `POST /confirm` |
| `timings_ms` | milliseconds per stage for this part and in total (HTTP only; `Inspector.timings` in Python, in seconds) |

The weights come from the median health over the last 16 parts. So a sensor that starts failing is switched off after a few parts, and one bad part does not switch it off on its own. The first parts after starting have a shorter history. `Inspector.reset()` (or `POST /reset`) clears it, for example after a sensor has been repaired.

## HTTP

```bash
uv sync --extra serve
uv run factorylens serve models/bagel.pt --port 8000
curl -X POST localhost:8000/inspect -F rgb=@part1.png -F xyz=@part1.tiff
```

- `GET /health` shows the model and its settings.
- `POST /inspect` returns the JSON above. Add `?with_map=true` to get the anomaly map as well.
- `GET /status` reports each sensor's health and weight over the current window and the median time per part. After 20 parts it also gives the alarm rate over the last 100 against the range expected from the calibration set (`alarm_rate`, `alarm_rate_expected`, `alarm_rate_high`). A rate above the range means more defects than usual or normal parts drifting away from the calibration parts.
- `POST /confirm?part=N` adds part N to the calibration set after an operator has checked it and found it good. Only one of the last 100 parts inspected with both sensors at full weight can be added, and each part only once. `Inspector.save` keeps the added parts.
- `POST /reset` clears the window and the alarm history.

Requests are handled one at a time, because the window is shared state.

## Checks

- A float32 model loaded from disk gives the same RGB and FPFH scores as the sweep's full-data runs for the same category, to the last digit (bagel).
- `scripts/verify_inspectors.py` fits, saves (float16) and reloads an inspector for every category, then streams the test parts through it in a random order, as they would arrive on a line. Single-sensor scores stay within 0.2% of the sweep's. Results from commit 40b1276, in `results/analysis/inspectors.csv`:

| category    | AUROC | false alarms | defects caught | ms per part | file (MB) |
|:------------|------:|:-------------|:---------------|------------:|----------:|
| bagel       | 0.926 | 9%           | 74%            | 100         | 40        |
| cable_gland | 0.880 | 0%           | 33%            | 64          | 37        |
| carrot      | 0.975 | 11%          | 94%            | 60          | 47        |
| cookie      | 0.822 | 18%          | 73%            | 101         | 35        |
| dowel       | 0.982 | 0%           | 89%            | 64          | 48        |
| foam        | 0.805 | 0%           | 49%            | 101         | 39        |
| peach       | 0.761 | 19%          | 51%            | 89          | 60        |
| potato      | 0.929 | 5%           | 75%            | 66          | 50        |
| rope        | 0.979 | 19%          | 96%            | 74          | 49        |
| tire        | 0.701 | 8%           | 34%            | 74          | 35        |
| **mean**    | 0.876 | 8.9%         | 67%            | 79          | 44        |

- The means match the benchmark through a different code path, which runs the whole test set at once: AUROC 0.876, false alarms 0.089, defects caught 0.668 (results.md, robustness and alarm thresholds).
- Per category, the false alarm rate at a 10% target ranges from 0% to 19%. That is the spread the Beta distribution predicts for 22–42 calibration parts, so it is not a fault of a particular category.
- Fitting takes 50–130 s per category. Inspection takes 60–100 ms per part on an M4 MacBook Air with the backbone on the GPU (MPS), plus background removal on the raw scan. On the CPU alone a raw scan takes about 150 ms in total; [latency.md](latency.md) breaks that down by stage and shows what the coreset size, threads, float16 and ONNX change. These are laptop numbers, not a claim about factory hardware.

See also the [model card](model_card.md).
