"""FPFH point-cloud features (Rusu et al., 2009) for the 3D-only detector.

FPFH describes the local shape around a point with a 33-bin histogram of the
angles between its normal and its neighbours' normals. Settings follow Back to
the Feature: normals from neighbours within 10 cm (at most 30 points), FPFH
within 25 cm (at most 100). For objects a few cm across the radii are never
the limit, so in practice the scale is set by the 30 and 100 nearest points.

Computing FPFH takes about half a second per scan, so the 28x28 patch
features are cached on disk, keyed by the settings and the preprocessing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from factorylens.data.cache import Prepared
from factorylens.data.mvtec3d import valid_points
from factorylens.repro import config_hash

N_BINS = 33


@dataclass(frozen=True)
class FPFHConfig:
    voxel_size: float = 0.05  # BTF's name; normal radius = 2x, feature radius = 5x
    normal_nn: int = 30
    feature_nn: int = 100
    grid: int = 28
    pad: bool = False  # BTF averages 3x3 without padding (224 -> 222) before pooling

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def fpfh_map(xyz: np.ndarray, config: FPFHConfig | None = None) -> np.ndarray:
    """(H, W, 3) organised point cloud -> (H, W, 33) FPFH, zeros where there is no point."""
    import open3d as o3d  # optional dependency: uv sync --extra 3d

    config = config or FPFHConfig()
    valid = valid_points(xyz)
    out = np.zeros((*xyz.shape[:2], N_BINS), np.float32)
    if valid.sum() < config.normal_nn:
        return out
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz[valid].astype(np.float64)))
    search = o3d.geometry.KDTreeSearchParamHybrid
    pc.estimate_normals(search(radius=2 * config.voxel_size, max_nn=config.normal_nn))
    # open3d's stubs only list the overload with explicit indices (and on Linux CI
    # open3d isn't installed at all, so the ignore is unused there)
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(  # type: ignore[call-arg, unused-ignore]
        pc, search(radius=5 * config.voxel_size, max_nn=config.feature_nn)
    )
    out[valid] = np.asarray(fpfh.data).T
    return out


def to_patches(maps: np.ndarray, config: FPFHConfig | None = None) -> torch.Tensor:
    """(N, H, W, 33) per-point features -> (N, grid, grid, 33) patch features."""
    config = config or FPFHConfig()
    x = torch.from_numpy(maps).permute(0, 3, 1, 2)
    x = F.avg_pool2d(x, 3, stride=1, padding=1 if config.pad else 0)
    x = F.adaptive_avg_pool2d(x, config.grid)
    return x.permute(0, 2, 3, 1).contiguous()


def patches_for(
    prepared: Prepared, config: FPFHConfig | None = None, cache_dir: str | Path | None = None
) -> torch.Tensor:
    """Patch features for every sample of a prepared split, read from disk if cached."""
    config = config or FPFHConfig()
    path = None
    if cache_dir is not None:
        key = config_hash({"fpfh": config.as_dict(), "preprocess": prepared.meta.get("settings")})
        name = f"{prepared.meta.get('category')}_{prepared.meta.get('split')}.pt"
        path = Path(cache_dir) / key / name
        if path.exists():
            cached: torch.Tensor = torch.load(path)
            if len(cached) == len(prepared):
                return cached
    maps = np.stack([fpfh_map(x, config) for x in prepared.xyz])
    patches = to_patches(maps, config)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(patches, path)
    return patches
