"""Reliability-aware late fusion.

Late fusion adds the RGB and 3D anomaly scores after standardising each on
the validation normals. A sensor that has gone *dead* drops out of that sum on
its own (every image gets the same score), but a *noisy* sensor keeps adding
variation that has nothing to do with defects (docs/results.md, robustness).

This module estimates, per test image and without labels, whether each
sensor still looks like it did on the validation normals, and down-weights the
one that does not. The signal is the *background level* of each sensor's
anomaly map: a low quantile of the map over the object. A real defect covers
a small part of the object and raises the top of the map; a degraded sensor
raises the whole map. For the point cloud, the number of points is a second
signal, because missing points look like background to FPFH rather than like
an anomaly.

Each signal is turned into a z-score against the validation normals. A
sensor's weight is 1 while its z-score is within what the validation images
themselves reach, and falls linearly to 0 over `ramp` standard deviations
beyond that. Nothing is fitted on test data or labels.

The two standardised scores are combined with weighted Stouffer:
(w_r z_r + w_g z_g) / sqrt(w_r^2 + w_g^2). With equal weights this ranks
images exactly like plain late fusion, so on clean data nothing changes
unless a sensor looks off. If both sensors look off, both keep weight 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Scored = tuple[np.ndarray, np.ndarray]  # image scores (N,), maps (N, H, W)


@dataclass(frozen=True)
class ReliabilityConfig:
    quantile: float = 0.25  # of the anomaly map over the object
    ramp: float = 3.0  # SDs beyond the worst validation image at which a weight reaches 0


def background_level(maps: np.ndarray, foreground: np.ndarray, quantile: float) -> np.ndarray:
    """Per image, a low quantile of the map over the object (the whole map if there is none)."""
    out = np.empty(len(maps))
    for i, (m, fg) in enumerate(zip(maps, foreground, strict=True)):
        out[i] = np.quantile(m[fg] if fg.any() else m, quantile)
    return out


@dataclass
class Reference:
    """Distribution of one signal over the validation normals."""

    mean: float
    std: float
    worst: float  # largest z-score any validation image reaches

    @classmethod
    def fit(cls, values: np.ndarray) -> Reference:
        mean, std = float(values.mean()), float(values.std()) + 1e-12
        return cls(mean, std, float(((values - mean) / std).max()))

    def z(self, values: np.ndarray) -> np.ndarray:
        return np.asarray((values - self.mean) / self.std)


def _standardise(scored: Scored, val: Scored) -> Scored:
    s, m = scored
    vs, vm = val
    return (s - vs.mean()) / (vs.std() + 1e-12), (m - vm.mean()) / (vm.std() + 1e-12)


@dataclass
class Weights:
    rgb: np.ndarray
    geo: np.ndarray
    u_rgb: np.ndarray  # how far beyond the validation normals, in their SDs
    u_geo: np.ndarray


class ReliabilityFusion:
    def __init__(self, config: ReliabilityConfig | None = None) -> None:
        self.config = config or ReliabilityConfig()

    def fit(
        self, rgb_val: Scored, geo_val: Scored, foreground: np.ndarray, points: np.ndarray
    ) -> ReliabilityFusion:
        """Validation-normal scores of both sensors, their object masks and point counts."""
        q = self.config.quantile
        self.rgb_val, self.geo_val = rgb_val, geo_val
        self.ref_rgb = Reference.fit(background_level(rgb_val[1], foreground, q))
        self.ref_geo = Reference.fit(background_level(geo_val[1], foreground, q))
        self.ref_points = Reference.fit(-points.astype(float))  # fewer points = higher
        return self

    def _weight(self, u: np.ndarray, worst: float) -> np.ndarray:
        return np.clip(1 - (u - worst) / self.config.ramp, 0.0, 1.0)

    def weights(
        self, rgb: Scored, geo: Scored, foreground: np.ndarray, points: np.ndarray
    ) -> Weights:
        q = self.config.quantile
        u_rgb = self.ref_rgb.z(background_level(rgb[1], foreground, q))
        u_level = self.ref_geo.z(background_level(geo[1], foreground, q))
        u_points = self.ref_points.z(-points.astype(float))
        w_rgb = self._weight(u_rgb, self.ref_rgb.worst)
        w_geo = np.minimum(
            self._weight(u_level, self.ref_geo.worst), self._weight(u_points, self.ref_points.worst)
        )
        both_off = (w_rgb == 0) & (w_geo == 0)
        w_rgb[both_off], w_geo[both_off] = 1.0, 1.0
        u_geo = np.maximum(u_level - self.ref_geo.worst, u_points - self.ref_points.worst)
        return Weights(w_rgb, w_geo, u_rgb - self.ref_rgb.worst, u_geo)

    def fuse(
        self, rgb: Scored, geo: Scored, foreground: np.ndarray, points: np.ndarray
    ) -> tuple[Scored, Weights]:
        w = self.weights(rgb, geo, foreground, points)
        zr, zg = _standardise(rgb, self.rgb_val), _standardise(geo, self.geo_val)
        norm = np.sqrt(w.rgb**2 + w.geo**2)
        scores = (w.rgb * zr[0] + w.geo * zg[0]) / norm
        maps = (w.rgb[:, None, None] * zr[1] + w.geo[:, None, None] * zg[1]) / norm[:, None, None]
        return (scores, maps), w
