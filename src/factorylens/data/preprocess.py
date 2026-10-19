"""Resizing and background removal for RGB + xyz pairs.

The objects in MVTec 3D-AD sit on a flat surface that fills the image border,
and that surface is tilted relative to the camera (z changes by 1-4 cm along
the border), so a fixed depth cut does not work. Instead a plane is fitted to
the border pixels with RANSAC and everything close to it is dropped, which is
the same idea as Back to the Feature (Horwitz & Hoshen, 2022).

BTF drops points within 5 mm of the plane. On dowel (thin rods lying flat)
that removes about 38% of the annotated defect pixels, against 14% at 2 mm,
so 2 mm is the default here and BTF_THRESHOLD is kept for reproducing their
numbers. scripts/check_background_removal.py has the per-category numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from factorylens.data.mvtec3d import valid_points

DEFAULT_THRESHOLD = 0.002
BTF_THRESHOLD = 0.005


def resize_rgb(rgb: np.ndarray, size: int) -> np.ndarray:
    """Square resize with antialiasing, (size, size, 3) uint8."""
    return np.asarray(Image.fromarray(rgb).resize((size, size), Image.Resampling.BILINEAR))


def resize_xyz(xyz: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour square resize.

    Bilinear interpolation would average real points with the (0, 0, 0)
    entries next to them and invent geometry halfway to the camera, so the
    only safe option here is to pick existing points.
    """
    h, w = xyz.shape[:2]
    rows = ((np.arange(size) + 0.5) * h / size).astype(int)
    cols = ((np.arange(size) + 0.5) * w / size).astype(int)
    return np.asarray(xyz[rows[:, None], cols[None, :]])


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(
        (size, size), Image.Resampling.NEAREST
    )
    return np.asarray(resized) > 0


@dataclass(frozen=True)
class Plane:
    normal: np.ndarray  # unit vector
    offset: float  # plane is normal @ p + offset = 0

    def distance(self, points: np.ndarray) -> np.ndarray:
        """Signed distance of (..., 3) points to the plane."""
        return np.asarray(points @ self.normal + self.offset)


def _plane_through(p: np.ndarray) -> Plane | None:
    normal = np.cross(p[1] - p[0], p[2] - p[0])
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        return None  # collinear sample
    normal = normal / norm
    return Plane(normal, float(-normal @ p[0]))


def _least_squares_plane(points: np.ndarray) -> Plane:
    centroid = points.mean(axis=0)
    # the normal is the direction of least variance
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vt[-1]
    return Plane(normal, float(-normal @ centroid))


def fit_plane(
    points: np.ndarray,
    threshold: float = 0.002,
    iterations: int = 200,
    seed: int = 0,
) -> Plane:
    """RANSAC plane fit to (N, 3) points, refined by least squares on the inliers."""
    if len(points) < 3:
        raise ValueError("need at least 3 points to fit a plane")
    rng = np.random.default_rng(seed)
    best, best_count = None, -1
    for _ in range(iterations):
        plane = _plane_through(points[rng.choice(len(points), 3, replace=False)])
        if plane is None:
            continue
        count = int((np.abs(plane.distance(points)) < threshold).sum())
        if count > best_count:
            best, best_count = plane, count
    if best is None:
        raise ValueError("all sampled point triples were collinear")
    inliers = points[np.abs(best.distance(points)) < threshold]
    return _least_squares_plane(inliers)


def border_mask(shape: tuple[int, int], width: int) -> np.ndarray:
    mask = np.zeros(shape, bool)
    mask[:width] = mask[-width:] = True
    mask[:, :width] = mask[:, -width:] = True
    return mask


def remove_background(
    xyz: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    border: int = 10,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, Plane]:
    """Zero out points within `threshold` metres of the background plane.

    Returns the cleaned xyz, the foreground mask and the fitted plane.
    """
    valid = valid_points(xyz)
    ring = border_mask(valid.shape, border) & valid
    plane = fit_plane(xyz[ring], seed=seed)
    foreground = valid & (np.abs(plane.distance(xyz)) >= threshold)
    cleaned = np.where(foreground[..., None], xyz, 0).astype(np.float32)
    return cleaned, foreground, plane


def to_point_cloud(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N, 3) valid points and their (N, 2) row/col pixel positions."""
    rows, cols = np.nonzero(valid_points(xyz))
    return xyz[rows, cols], np.stack([rows, cols], axis=1)
