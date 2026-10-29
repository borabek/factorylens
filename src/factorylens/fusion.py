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

With `window` set, a sensor's state is read from the median signal over the
current image and the previous window-1 images instead of the current image
alone. A real sensor fault lasts for many parts, and a defect-like local
corruption (a smudge, an occluder) then switches the sensor off for the whole
stretch instead of for some images and not others. Test images are processed
in a fixed random order (`order_seed`), because the dataset groups them by
defect type and a window over that order would mix the classes.

The two standardised scores are combined with weighted Stouffer:
(w_r z_r + w_g z_g) / sqrt(w_r^2 + w_g^2). With equal weights this ranks
images exactly like plain late fusion, so on clean data nothing changes
unless a sensor looks off. If both sensors look off, both keep weight 1.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

Scored = tuple[np.ndarray, np.ndarray]  # image scores (N,), maps (N, H, W)


@dataclass(frozen=True)
class ReliabilityConfig:
    quantile: float = 0.25  # of the anomaly map over the object
    ramp: float = 3.0  # SDs beyond the worst validation image at which a weight reaches 0
    window: int | None = None  # images; None = every image on its own
    order_seed: int = 0
    # a second, high quantile: how much of the object is lit up. Catches local
    # corruptions (an occluder) that leave the background level alone. A large
    # real defect lifts it too, which is why it is only safe with a window.
    extent: float | None = None


def running_median(values: np.ndarray, window: int, order: np.ndarray) -> np.ndarray:
    """Median of each value and the window-1 values before it, visiting them in `order`."""
    out = np.empty(len(values))
    seq = values[order]
    for i in range(len(seq)):
        out[order[i]] = np.median(seq[max(0, i - window + 1) : i + 1])
    return out


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


@dataclass
class Scale:
    """Mean and spread of one sensor's image scores and map values on the validation normals."""

    score_mean: float
    score_std: float
    map_mean: float
    map_std: float

    @classmethod
    def fit(cls, val: Scored) -> Scale:
        s, m = val
        return cls(float(s.mean()), float(s.std()) + 1e-12, float(m.mean()), float(m.std()) + 1e-12)

    def __call__(self, scored: Scored) -> Scored:
        s, m = scored
        return (s - self.score_mean) / self.score_std, (m - self.map_mean) / self.map_std


@dataclass
class Weights:
    rgb: np.ndarray
    geo: np.ndarray
    u_rgb: np.ndarray  # how far beyond the validation normals, in their SDs
    u_geo: np.ndarray
    # both sensors looked off, so both were put back to full weight
    both_off: np.ndarray | None = None


class ReliabilityFusion:
    def __init__(self, config: ReliabilityConfig | None = None) -> None:
        self.config = config or ReliabilityConfig()

    def fit(
        self, rgb_val: Scored, geo_val: Scored, foreground: np.ndarray, points: np.ndarray
    ) -> ReliabilityFusion:
        """Validation-normal scores of both sensors, their object masks and point counts."""
        q = self.config.quantile
        self.scale_rgb, self.scale_geo = Scale.fit(rgb_val), Scale.fit(geo_val)
        self.refs = {
            "rgb": Reference.fit(background_level(rgb_val[1], foreground, q)),
            "geo": Reference.fit(background_level(geo_val[1], foreground, q)),
            "points": Reference.fit(-points.astype(float)),  # fewer points = higher
        }
        if self.config.extent is not None:
            e = self.config.extent
            self.refs["rgb_extent"] = Reference.fit(background_level(rgb_val[1], foreground, e))
            self.refs["geo_extent"] = Reference.fit(background_level(geo_val[1], foreground, e))
        return self

    def state(self) -> dict[str, Any]:
        """Everything a fitted fusion needs, as plain numbers (for saving a model)."""
        return {
            "config": asdict(self.config),
            "scale_rgb": asdict(self.scale_rgb),
            "scale_geo": asdict(self.scale_geo),
            "refs": {k: asdict(v) for k, v in self.refs.items()},
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> ReliabilityFusion:
        f = cls(ReliabilityConfig(**state["config"]))
        f.scale_rgb, f.scale_geo = Scale(**state["scale_rgb"]), Scale(**state["scale_geo"])
        f.refs = {k: Reference(**v) for k, v in state["refs"].items()}
        return f

    def _weight(self, u: np.ndarray) -> np.ndarray:
        """1 up to the worst validation image (u = 0), 0 at `ramp` SDs beyond it."""
        return np.clip(1 - u / self.config.ramp, 0.0, 1.0)

    def _beyond(self, name: str, values: np.ndarray) -> np.ndarray:
        ref = self.refs[name]
        return ref.z(values) - ref.worst

    def signals(
        self, rgb: Scored, geo: Scored, foreground: np.ndarray, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per image, how far each sensor is beyond anything seen on the validation normals."""
        q = self.config.quantile
        u_rgb = self._beyond("rgb", background_level(rgb[1], foreground, q))
        u_geo = np.maximum(
            self._beyond("geo", background_level(geo[1], foreground, q)),
            self._beyond("points", -points.astype(float)),
        )
        if self.config.extent is not None:
            e = self.config.extent
            u_rgb = np.maximum(
                u_rgb, self._beyond("rgb_extent", background_level(rgb[1], foreground, e))
            )
            u_geo = np.maximum(
                u_geo, self._beyond("geo_extent", background_level(geo[1], foreground, e))
            )
        return u_rgb, u_geo

    def weights_from(self, u_rgb: np.ndarray, u_geo: np.ndarray) -> Weights:
        w_rgb, w_geo = self._weight(u_rgb), self._weight(u_geo)
        both_off = (w_rgb == 0) & (w_geo == 0)
        w_rgb[both_off], w_geo[both_off] = 1.0, 1.0
        return Weights(w_rgb, w_geo, u_rgb, u_geo, both_off)

    def weights(
        self, rgb: Scored, geo: Scored, foreground: np.ndarray, points: np.ndarray
    ) -> Weights:
        u_rgb, u_geo = self.signals(rgb, geo, foreground, points)
        if self.config.window:
            order = np.random.default_rng(self.config.order_seed).permutation(len(u_rgb))
            u_rgb = running_median(u_rgb, self.config.window, order)
            u_geo = running_median(u_geo, self.config.window, order)
        return self.weights_from(u_rgb, u_geo)

    def combine(self, rgb: Scored, geo: Scored, w: Weights) -> Scored:
        """Weighted Stouffer of the two standardised scores and maps."""
        zr, zg = self.scale_rgb(rgb), self.scale_geo(geo)
        norm = np.sqrt(w.rgb**2 + w.geo**2)
        scores = (w.rgb * zr[0] + w.geo * zg[0]) / norm
        maps = (w.rgb[:, None, None] * zr[1] + w.geo[:, None, None] * zg[1]) / norm[:, None, None]
        return scores, maps

    def fuse(
        self, rgb: Scored, geo: Scored, foreground: np.ndarray, points: np.ndarray
    ) -> tuple[Scored, Weights]:
        w = self.weights(rgb, geo, foreground, points)
        return self.combine(rgb, geo, w), w
