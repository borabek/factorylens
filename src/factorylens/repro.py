"""Seeding, and writing results together with what produced them.

Every number in the report should trace back to a JSON file written by
save_result(), which records the config, a hash of it, the git commit and the
library versions alongside the metrics.
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

TRACKED_PACKAGES = ("factorylens", "numpy", "pillow", "tifffile", "torch", "scikit-learn")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)


def config_hash(config: Mapping[str, Any]) -> str:
    """Short hash that ignores key order, so equal configs always hash the same."""
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def git_state(cwd: str | Path | None = None) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def package_versions() -> dict[str, str]:
    versions = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def run_info(cwd: str | Path | None = None) -> dict[str, Any]:
    """Environment and code state. Call it when a run *starts*.

    git state is read from the checkout the package was imported from, not
    the current directory, so it describes the code that is actually running.
    """
    if cwd is None:
        cwd = Path(__file__).resolve().parent
    return {
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_state(cwd),
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "packages": package_versions(),
    }


def save_result(
    path: str | Path,
    metrics: Mapping[str, Any],
    config: Mapping[str, Any],
    run: Mapping[str, Any] | None = None,
) -> Path:
    """Write metrics with their config and run info. NaN or inf in metrics raises ValueError.

    Pass `run` captured with run_info() at the start of a long job. Reading it
    at save time records whatever the checkout looks like an hour later.
    """
    path = Path(path)
    record = {
        "config": dict(config),
        "config_hash": config_hash(config),
        "metrics": dict(metrics),
        "run": dict(run) if run is not None else run_info(),
    }
    # allow_nan=False: a NaN score is a bug somewhere upstream, not a result
    text = json.dumps(record, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")
    return path


def load_result(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        record: dict[str, Any] = json.load(f)
    return record
