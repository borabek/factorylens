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

Everything goes into `models/bagel.pt`, about 80 MB for bagel. The ImageNet backbone weights are not stored; torchvision loads them as usual.

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

The weights come from the median health over the last 16 parts. So a sensor that starts failing is switched off after a few parts, and one bad part does not switch it off on its own. The first parts after starting have a shorter history. `Inspector.reset()` (or `POST /reset`) clears it, for example after a sensor has been repaired.

## HTTP

```bash
uv sync --extra serve
uv run factorylens serve models/bagel.pt --port 8000
curl -X POST localhost:8000/inspect -F rgb=@part1.png -F xyz=@part1.tiff
```

- `GET /health` shows the model and its settings.
- `POST /inspect` returns the JSON above. Add `?with_map=true` to get the anomaly map as well.
- `POST /reset` clears the window.

Requests are handled one at a time, because the window is shared state.

## Checks

- A model loaded from disk gives the same RGB and FPFH scores as the sweep's full-data runs for the same category, to the last digit.
- On bagel's test parts, in the dataset's order, it scores AUROC 0.921, with 9.1% false alarms at a 10% target and 73% of defective parts caught.
- On an M4 MacBook Air (MPS), one preprocessed part takes about 105 ms. A raw 800 × 800 scan including background removal takes about 150 ms. These are laptop numbers, not a claim about factory hardware.
