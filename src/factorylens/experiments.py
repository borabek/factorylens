"""Running a detector on one category and recording what came out.

Each run writes two things:
  results/<model>/<name>.json   metrics + config + provenance (committed)
  outputs/scores/<model>/<name>.npz   per-image scores and pixel maps (local,
                                      reused later for fusion and analysis)
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from factorylens.data import cache, subsets
from factorylens.eval import metrics
from factorylens.models.detectors import Detector, RGBDetector
from factorylens.models.patchcore import PatchCoreConfig
from factorylens.repro import config_hash, load_result, run_info, save_result


def evaluate(
    labels: np.ndarray, scores: np.ndarray, masks: np.ndarray, maps: np.ndarray
) -> dict[str, float]:
    lo, hi = metrics.bootstrap_ci(labels, scores)
    return {
        "image_auroc": metrics.image_auroc(labels, scores),
        "image_auroc_ci_low": lo,
        "image_auroc_ci_high": hi,
        "pixel_auroc": metrics.pixel_auroc(masks, maps),
        "aupro": metrics.aupro(masks, maps),
    }


def run_name(config: dict[str, Any]) -> str:
    k = config["k"] if config["k"] is not None else "all"
    return f"{config['category']}_k{k}_s{config['seed']}_{config_hash(config)}"


def run(
    detector: Detector,
    root: str | Path,
    cache_dir: str | Path,
    category: str,
    k: int | None = None,
    seed: int = 0,
    settings: cache.Settings | None = None,
    kshot_path: str | Path = "configs/kshot.json",
    results_dir: str | Path = "results/rgb",
    scores_dir: str | Path | None = None,
    extra_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit `detector` on one category, score its test split, save metrics (and scores)."""
    run = run_info()  # before anything else, see save_result()
    settings = settings or cache.Settings()
    train = cache.get(root, cache_dir, category, "train", settings)
    if k is not None:
        train = cache.take(train, subsets.load(kshot_path)[category][str(seed)][str(k)])
    test = cache.get(root, cache_dir, category, "test", settings)

    start = time.perf_counter()
    detector.fit(train)
    fit_s = time.perf_counter() - start
    start = time.perf_counter()
    scores, maps = detector.score(test)
    score_s = time.perf_counter() - start

    config = {
        "category": category,
        "k": k,
        "seed": seed,
        **(extra_config or {}),
        "preprocess": settings.as_dict(),
        **detector.config_dict(),
    }
    result = evaluate(test.is_anomalous, scores, test.mask, maps)
    result |= {
        "n_train": len(train),
        "n_test": len(test),
        "bank_size": detector.bank_size,
        "fit_s": fit_s,
        "score_ms_per_image": 1000 * score_s / len(test),
    }
    name = run_name(config)
    save_result(Path(results_dir) / f"{name}.json", result, config, run=run)
    if scores_dir is not None:
        Path(scores_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            Path(scores_dir) / f"{name}.npz",
            scores=scores,
            maps=maps.astype(np.float16),
            labels=test.is_anomalous,
            defect=np.array(test.defect),
            stem=np.array(test.stem),
        )
    return result


def run_rgb(
    root: str | Path,
    cache_dir: str | Path,
    category: str,
    k: int | None = None,
    seed: int = 0,
    model_config: PatchCoreConfig | None = None,
    kshot_path: str | Path = "configs/kshot.json",
    results_dir: str | Path = "results/rgb",
    scores_dir: str | Path | None = "outputs/scores/rgb",
    device: str | None = None,
    pretrained: bool = True,
) -> dict[str, Any]:
    model_config = replace(model_config or PatchCoreConfig(), seed=seed)
    detector = RGBDetector(model_config, device=device, pretrained=pretrained)
    return run(
        detector,
        root,
        cache_dir,
        category,
        k=k,
        seed=seed,
        kshot_path=kshot_path,
        results_dir=results_dir,
        scores_dir=scores_dir,
        extra_config={"pretrained": pretrained},
    )


def collect(results_dir: str | Path) -> pd.DataFrame:
    """One row per result file, config fields and metrics side by side."""
    rows = []
    for path in sorted(Path(results_dir).glob("*.json")):
        record = load_result(path)
        config = {k: v for k, v in record["config"].items() if not isinstance(v, dict)}
        if isinstance(config.get("layers"), list):
            config["layers"] = "+".join(config["layers"])
        rows.append({**config, **record["metrics"], "file": path.name})
    return pd.DataFrame(rows)
