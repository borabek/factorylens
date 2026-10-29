# Latency and memory

How long one part takes, where the time goes, how much memory it needs, and what the size of the memory banks costs and buys. Measured with `scripts/benchmark_latency.py` at commit e77925b; the numbers are in `results/analysis/latency_*.csv`.

**Machine:** an M4 MacBook Air (4 performance + 6 efficiency cores, 16 GB, no fan), with nothing else running. These are laptop numbers. An industrial edge PC has a different CPU and different memory bandwidth, so absolute times will differ. The breakdown by stage, and what changes it, should carry over better.

```bash
uv sync --extra 3d --group bench
OMP_NUM_THREADS=4 uv run python scripts/benchmark_latency.py all data/raw/mvtec3d \
    --categories bagel cable_gland carrot cookie dowel foam peach potato rope tire
```

## One part, stage by stage

The bagel inspector (10% coreset, float16 file) loaded on the CPU, given 30 raw test scans one at a time after 3 warm-up parts. Median milliseconds per part. Each setting runs in its own process, so the thread count also reaches open3d's OpenMP and the peak memory belongs to that setting alone.

| stage | 1 thread | 2 threads | 4 threads | 4 threads, GPU (MPS) |
|---|---:|---:|---:|---:|
| preprocess (background plane, resize) | 30 | 30 | 30 | 30 |
| backbone (WideResNet-50, layers 2–3) | 35 | 29 | 26 | 4 |
| RGB bank search + map | 32 | 29 | 28 | 34 |
| FPFH features | 57 | 55 | 56 | 56 |
| FPFH bank search + map | 11 | 8 | 7 | 7 |
| fusion, p-value, alarm | 0.7 | 0.7 | 0.6 | 0.6 |
| **total** | **166** | **152** | **148** | **132** |
| 90th percentile | 174 | 160 | 158 | 140 |

- **FPFH is the largest stage, at 38%,** and it runs in open3d on the CPU whatever device the network uses. Together with preprocessing it is 58% of the time.
- **More threads barely help** (166 → 148 ms). The backbone and the bank search are already fast on one thread here, probably because PyTorch's matrix products use Apple's Accelerate library, and open3d's FPFH hardly speeds up with more threads.
- **The GPU makes the backbone 6× faster but the part only 11% faster,** because the stages that dominate don't run on it. A GPU is not the first thing to buy for this pipeline.
- **The first part is slow.** Loading a model file takes 0.6 s, and the first inspection pays for memory allocation. A service should load the model and inspect one part before the line starts.

## Memory

| | peak RSS |
|---|---:|
| Python with PyTorch, open3d and the package imported | 455 MB |
| + model loaded (backbone, both banks) | 980 MB |
| + 30 parts inspected | 1.49 GB |

- **Float16 banks halve the file (40 MB instead of 81 MB) but not the memory.** They are converted back to float32 when loaded, because float16 nearest-neighbour search on the CPU is slow or unsupported. Peak memory is the same within 3%.
- A third of the peak is the libraries themselves. The rest is the backbone, the banks and the working memory of one inspection. The backbone is the same for every product. A station that switches between products could keep one backbone and load only the banks (40 MB each), but the code loads a complete inspector per file today.

## Coreset size: the cheapest speed-up

The memory banks keep a greedy coreset of the training patches, 10% by default, as in PatchCore. Thinning them further makes the nearest-neighbour search proportionally faster. The question is what it costs in detection. All ten categories, banks of 1% to 20% from one greedy run (k-center selection is a prefix process, so the first m picks don't depend on how many come after), the same validation calibration and the same window as the inspector:

| coreset | bank size (MB) | search per part (ms) | image AUROC | false alarms (α = 0.1) | defects caught |
|---:|---:|---:|---:|---:|---:|
| 1% | 9 | 5 | 0.878 | 11.1% | 66% |
| 2.5% | 22 | 10 | 0.870 | 9.6% | 65% |
| 5% | 44 | 19 | 0.874 | 9.3% | 65% |
| 10% (default) | 88 | 38 | 0.876 | 8.9% | 67% |
| 20% | 176 | 75 | 0.878 | 11.7% | 69% |

- **From 1% to 20%, the mean AUROC stays within 0.008, with no trend.** Per category, going from 10% to 1% moves it by −0.02 to +0.04: cable_gland and tire lose a little, foam and peach gain. Nothing here says that smaller banks are worse.
- **At 1% the search takes 5 ms instead of 38, and the banks take 9 MB instead of 88.** That removes about a fifth of the time per part on this machine (148 → about 115 ms).
- The false alarm rate moves between 9% and 12% with no trend in bank size. That is the noise expected from 22–42 calibration parts ([results.md](results.md), alarm thresholds), not an effect of the coreset.
- The default stays at 10%, because every result in the paper and the docs uses it. `factorylens fit --coreset-ratio 0.01` gives the small version for a station that needs the time. This is one dataset with one feature extractor; a product with more varied normal parts may need a larger bank, so the check is worth repeating on new data.

## ONNX export of the backbone

The backbone exported with `torch.onnx.export` and run with onnxruntime on the CPU, at 224 px, one image:

| threads | PyTorch (ms) | onnxruntime (ms) | largest difference in the features |
|---:|---:|---:|---:|
| 1 | 29 | 19 | < 1e-5 (relative) |
| 2 | 25 | 20 | < 1e-5 |
| 4 | 22 | 21 | < 1e-5 |

The export is exact, and onnxruntime is a third faster with one thread. With four threads the two are about equal. ONNX matters more for getting the network onto hardware without Python than for speed on this laptop. The weights (100 MB) are the same either way. The inspector itself still uses PyTorch: the exported graph is only the backbone, and FPFH and the bank search would need their own ports.

## What did not help: FPFH on a second thread

FPFH and the RGB side don't depend on each other, so computing them at the same time looks like it should hide most of FPFH's 56 ms. It doesn't. On a second thread the total went from 149 to only 146 ms, because the backbone slowed from 27 to 80 ms. open3d holds Python's GIL while it computes, so the RGB thread waits. This happens with one thread per side too, so it isn't competition for cores. In a separate process the part took 101 ms instead of 111 (without preprocessing). That isn't worth a second process of several hundred MB, probably because macOS put the worker on the efficiency cores. The stages therefore run in sequence, with a comment in `Inspector.inspect` saying why.

## What this means for a line

At about 150 ms a part on four CPU threads, one station could inspect roughly 6 parts a second, 400 a minute, if parts arrive evenly and nothing else runs. With a 1% coreset it is about 8 a second. [architecture.md](architecture.md) describes where this runs and what happens when it doesn't answer in time.
