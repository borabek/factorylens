"""FactoryLens-C: corruptions of the RGB image and the point cloud, 5 severities each.

The idea follows ImageNet-C (Hendrycks & Dietterich, 2019), where the RGB
settings come from, extended to the 3D sensor and to the two sensors falling
out of step. Every corruption hits one sensor only, so a fusion method can be
judged on whether it notices that one input got worse.

All corruptions act on the preprocessed 224 px arrays: background already
removed, xyz in metres with zeros where there is no point. For the 3D ones
that is a simplification (a real sensor would be noisy before the plane fit
too), chosen so that the clean and corrupted runs share their preprocessing.

Randomness is seeded from (seed, corruption, severity, sample), so the
same call always gives the same corrupted data, in any order. Only the code
and seeds are published, never corrupted copies of the dataset.
"""

from __future__ import annotations

import io
import zlib
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from PIL import Image
from scipy import ndimage

from factorylens.data.cache import Prepared
from factorylens.data.mvtec3d import valid_points

Rng = np.random.Generator
VERSION = 1  # bump when a corruption changes, so cached features are not reused

# RGB, from ImageNet-C at 224 px
NOISE_SIGMA = (0.08, 0.12, 0.18, 0.26, 0.38)  # fraction of the 0-255 range
BLUR_RADIUS = (3, 4, 6, 8, 10)  # disk kernel, px
BRIGHTNESS = (0.1, 0.2, 0.3, 0.4, 0.5)  # added to [0, 1] intensities
JPEG_QUALITY = (25, 18, 15, 10, 7)
OCCLUDED = (0.02, 0.05, 0.1, 0.15, 0.25)  # share of the image under a grey square

# 3D. The MVTec 3D-AD objects are 4-10 cm across, scanned at about 0.2 mm/px.
DEPTH_SIGMA_M = (0.00025, 0.0005, 0.001, 0.002, 0.004)
HOLES = (0.05, 0.1, 0.2, 0.3, 0.5)  # share of the object's points removed in blobs
DEPTH_STEP_M = (0.0005, 0.001, 0.002, 0.003, 0.005)
TILT_DEG = (1.0, 2.0, 4.0, 6.0, 10.0)
SHIFT_PX = (1, 2, 4, 6, 8)  # xyz moved against the RGB image


def _rgb_float(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.float32) / 255.0


def _to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)


def gaussian_noise(rgb: np.ndarray, severity: int, rng: Rng) -> np.ndarray:
    x = _rgb_float(rgb)
    return _to_uint8(x + rng.normal(0.0, NOISE_SIGMA[severity - 1], x.shape))


def defocus_blur(rgb: np.ndarray, severity: int, rng: Rng) -> np.ndarray:
    r = BLUR_RADIUS[severity - 1]
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    disk = (xx**2 + yy**2 <= r**2).astype(np.float32)
    disk /= disk.sum()
    x = _rgb_float(rgb)
    out = np.stack([ndimage.convolve(x[..., c], disk, mode="reflect") for c in range(3)], -1)
    return _to_uint8(out)


def brightness(rgb: np.ndarray, severity: int, rng: Rng) -> np.ndarray:
    return _to_uint8(_rgb_float(rgb) + BRIGHTNESS[severity - 1])


def jpeg(rgb: np.ndarray, severity: int, rng: Rng) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=JPEG_QUALITY[severity - 1])
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def occlusion(
    rgb: np.ndarray, severity: int, rng: Rng, centre: np.ndarray | None = None
) -> np.ndarray:
    """A mid-grey square, centred on a random object pixel when one is given."""
    h, w = rgb.shape[:2]
    side = int(round(np.sqrt(OCCLUDED[severity - 1] * h * w)))
    if centre is None:
        cy, cx = rng.integers(0, h), rng.integers(0, w)
    else:
        cy, cx = centre
    top = int(np.clip(cy - side // 2, 0, h - side))
    left = int(np.clip(cx - side // 2, 0, w - side))
    out = rgb.copy()
    out[top : top + side, left : left + side] = 128
    return out


def depth_noise(xyz: np.ndarray, severity: int, rng: Rng) -> np.ndarray:
    out = xyz.copy()
    valid = valid_points(xyz)
    out[valid, 2] += rng.normal(0.0, DEPTH_SIGMA_M[severity - 1], valid.sum()).astype(xyz.dtype)
    return out


def holes(xyz: np.ndarray, severity: int, rng: Rng) -> np.ndarray:
    """Remove round blobs of points until the target share of the object is gone."""
    out = xyz.copy()
    valid = valid_points(xyz)
    target = HOLES[severity - 1] * valid.sum()
    removed = np.zeros_like(valid)
    yy, xx = np.mgrid[: xyz.shape[0], : xyz.shape[1]]
    rows, cols = np.nonzero(valid)
    while (removed & valid).sum() < target:
        i = rng.integers(len(rows))
        # blobs scale with the object: 5-15% of the side of a square of the same area
        radius = rng.uniform(0.05, 0.15) * np.sqrt(valid.sum())
        removed |= (yy - rows[i]) ** 2 + (xx - cols[i]) ** 2 <= radius**2
    out[removed] = 0
    return out


def quantise(xyz: np.ndarray, severity: int, rng: Rng) -> np.ndarray:
    """Depth resolution of a cheaper sensor: z rounded to a fixed step."""
    out = xyz.copy()
    valid = valid_points(xyz)
    step = DEPTH_STEP_M[severity - 1]
    z = np.round(out[valid, 2] / step) * step
    # a point that lands exactly on z = 0 with x = y = 0 would read as missing
    out[valid, 2] = np.where(z == 0, step, z)
    return out


def tilt(xyz: np.ndarray, severity: int, rng: Rng) -> np.ndarray:
    """Rotate the object about its centroid by a small angle around a random horizontal axis."""
    out = xyz.copy()
    valid = valid_points(xyz)
    if not valid.any():
        return out
    pts = xyz[valid].astype(np.float64)
    centre = pts.mean(axis=0)
    phi = rng.uniform(0, 2 * np.pi)
    axis = np.array([np.cos(phi), np.sin(phi), 0.0])
    theta = np.deg2rad(TILT_DEG[severity - 1])
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rot = np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * k @ k
    out[valid] = ((pts - centre) @ rot.T + centre).astype(xyz.dtype)
    return out


def misregister(xyz: np.ndarray, severity: int, rng: Rng) -> np.ndarray:
    """Shift the xyz image against the RGB image, as after a knock to one sensor."""
    d = SHIFT_PX[severity - 1]
    angle = rng.uniform(0, 2 * np.pi)
    dy, dx = int(round(d * np.sin(angle))), int(round(d * np.cos(angle)))
    if dy == 0 and dx == 0:
        dx = d
    out = np.zeros_like(xyz)
    h, w = xyz.shape[:2]
    src = xyz[max(0, -dy) : h - max(0, dy), max(0, -dx) : w - max(0, dx)]
    out[max(0, dy) : max(0, dy) + src.shape[0], max(0, dx) : max(0, dx) + src.shape[1]] = src
    return out


@dataclass(frozen=True)
class Corruption:
    name: str
    sensor: str  # "rgb" or "xyz"
    apply: Callable[..., np.ndarray]
    severities: int = 5


CORRUPTIONS = {
    c.name: c
    for c in [
        Corruption("gaussian_noise", "rgb", gaussian_noise),
        Corruption("defocus_blur", "rgb", defocus_blur),
        Corruption("brightness", "rgb", brightness),
        Corruption("jpeg", "rgb", jpeg),
        Corruption("occlusion", "rgb", occlusion),
        Corruption("rgb_missing", "rgb", lambda rgb, s, rng: np.zeros_like(rgb), severities=1),
        Corruption("depth_noise", "xyz", depth_noise),
        Corruption("holes", "xyz", holes),
        Corruption("quantise", "xyz", quantise),
        Corruption("tilt", "xyz", tilt),
        Corruption("misregister", "xyz", misregister),
        Corruption("xyz_missing", "xyz", lambda xyz, s, rng: np.zeros_like(xyz), severities=1),
    ]
}


def rng_for(seed: int, name: str, severity: int, stem: str) -> Rng:
    key = [seed, zlib.crc32(name.encode()), severity, zlib.crc32(stem.encode())]
    return np.random.default_rng(key)


def corrupt(prepared: Prepared, name: str, severity: int, seed: int = 0) -> Prepared:
    """A corrupted copy of a prepared split. Masks and labels are left alone."""
    c = CORRUPTIONS[name]
    if not 1 <= severity <= c.severities:
        raise ValueError(f"{name} has severities 1-{c.severities}, got {severity}")
    rgb, xyz = prepared.rgb.copy(), prepared.xyz.copy()
    for i, stem in enumerate(prepared.stem):
        # stems repeat across the defect folders of a test split
        rng = rng_for(seed, name, severity, f"{prepared.defect[i]}/{stem}")
        if c.sensor == "rgb":
            if name == "occlusion" and prepared.foreground[i].any():
                rows, cols = np.nonzero(prepared.foreground[i])
                j = rng.integers(len(rows))
                rgb[i] = occlusion(rgb[i], severity, rng, centre=np.array([rows[j], cols[j]]))
            else:
                rgb[i] = c.apply(rgb[i], severity, rng)
        else:
            xyz[i] = c.apply(xyz[i], severity, rng)
    foreground = valid_points(xyz) if c.sensor == "xyz" else prepared.foreground
    meta = dict(prepared.meta)
    meta["settings"] = {
        **(prepared.meta.get("settings") or {}),
        "corruption": {"name": name, "severity": severity, "seed": seed, "version": VERSION},
    }
    meta["split"] = f"{prepared.meta.get('split')}-{name}{severity}-s{seed}"
    return replace(prepared, rgb=rgb, xyz=xyz, foreground=foreground, meta=meta)
