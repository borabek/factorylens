"""A fitted detector for one product, for use on a line.

`Inspector.fit` builds everything the robustness benchmark builds for one
category (RGB and FPFH memory banks, validation statistics, reliability
references) and adds an alarm threshold, split-conformal on the validation
normals. `inspect` then takes one part at a time, in arrival order, and keeps
the reliability window as state: the sensor weights of each part depend on
the parts before it, as they would on a line.

`save`/`load` write and read a single file with the banks and the numbers;
the ImageNet backbone weights come from torchvision as usual.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from factorylens import __version__, conformal
from factorylens.data import cache
from factorylens.data.cache import Prepared
from factorylens.data.mvtec3d import valid_points
from factorylens.fusion import ReliabilityConfig, ReliabilityFusion, Weights
from factorylens.models import fpfh
from factorylens.models.memory_bank import MemoryBank
from factorylens.models.patchcore import PatchCore, PatchCoreConfig
from factorylens.sweep import SweepConfig

FORMAT = 1
RECENT = 100  # parts kept for the alarm rate and for confirming good parts
MIN_FOR_RATE = 20  # parts before the alarm rate is judged


@dataclass(frozen=True)
class InspectorConfig:
    sweep: SweepConfig = field(default_factory=SweepConfig)
    reliability: ReliabilityConfig = field(default_factory=lambda: ReliabilityConfig(window=16))
    alpha: float = 0.1  # target false alarm rate


@dataclass
class Inspection:
    score: float  # fused, in validation SDs
    p_value: float  # conformal, against the validation normals
    alarm: bool
    weight_rgb: float
    weight_3d: float
    health_rgb: float  # SDs beyond the worst validation part (<= 0: looks normal)
    health_3d: float
    score_rgb: float
    score_3d: float
    anomaly_map: np.ndarray = field(repr=False)  # (S, S) fused, in validation SDs
    # each sensor's own map, on the same scale, before weighting
    map_rgb: np.ndarray | None = field(default=None, repr=False)
    map_3d: np.ndarray | None = field(default=None, repr=False)
    part: int = 0  # number of this part since the model was loaded, for confirm_good()
    # both sensors looked off; both are then kept at full weight, but it needs a person
    both_off: bool = field(default=False, repr=False)

    @property
    def sensor_alert(self) -> str | None:
        """Which sensor needs looking at: counted at less than half weight for this part."""
        return _alert(self.weight_rgb, self.weight_3d, self.both_off)

    def as_dict(self, with_map: bool = False) -> dict[str, Any]:
        skip = {"anomaly_map", "map_rgb", "map_3d", "both_off"}
        d: dict[str, Any] = {k: _plain(v) for k, v in asdict(self).items() if k not in skip}
        d["sensor_alert"] = self.sensor_alert
        if with_map:
            d["anomaly_map"] = np.round(self.anomaly_map, 3).tolist()
        return d


def _alert(weight_rgb: float, weight_3d: float, both_off: bool) -> str | None:
    if both_off:
        return "rgb+3d"
    off = [name for name, w in (("rgb", weight_rgb), ("3d", weight_3d)) if w < 0.5]
    return "+".join(off) if off else None


def _plain(v: Any) -> bool | int | float:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    return v if isinstance(v, int) else float(v)


def _sweep_from(d: dict[str, Any]) -> SweepConfig:
    return SweepConfig(
        rgb=PatchCoreConfig(**{**d["rgb"], "layers": tuple(d["rgb"]["layers"])}),
        fpfh=fpfh.FPFHConfig(**d["fpfh"]),
        settings=cache.Settings(**d["settings"]),
    )


class Inspector:
    def __init__(
        self,
        config: InspectorConfig | None = None,
        device: str | None = None,
        pretrained: bool = True,
    ) -> None:
        self.config = config or InspectorConfig()
        self.pretrained = pretrained
        self.rgb_model = PatchCore(self.config.sweep.rgb, device=device, pretrained=pretrained)
        bank = self.config.sweep.rgb.bank_config()
        self.rgb_bank, self.geo_bank = MemoryBank(bank), MemoryBank(bank)
        self.meta: dict[str, Any] = {}
        self.parts_seen = 0
        # fused score of each recent part, or None if a sensor was down-weighted for it
        self.recent: dict[int, float | None] = {}
        self.reset()

    # -- fitting ---------------------------------------------------------------

    def _geo_features(self, p: Prepared) -> torch.Tensor:
        geo = fpfh.to_patches(
            np.stack([fpfh.fpfh_map(x, self.config.sweep.fpfh) for x in p.xyz]),
            self.config.sweep.fpfh,
        )
        return geo.to(self.rgb_model.device)

    def _features(self, p: Prepared) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rgb_model.embed(p.rgb), self._geo_features(p)

    def fit(self, train: Prepared, validation: Prepared) -> Inspector:
        tr_rgb, tr_geo = self._features(train)
        va_rgb, va_geo = self._features(validation)
        self.size = (train.rgb.shape[1], train.rgb.shape[2])
        self.rgb_bank.fit(tr_rgb)
        self.geo_bank.fit(tr_geo)
        self.calibrate(validation, va_rgb, va_geo)
        self.meta = {
            "category": train.meta.get("category"),
            "n_train": len(train),
            "n_validation": len(validation),
            "factorylens": __version__,
        }
        return self

    def calibrate(self, validation: Prepared, va_rgb: torch.Tensor, va_geo: torch.Tensor) -> None:
        """Fusion statistics and the alarm threshold, for banks that are already set."""
        rgb_val = self.rgb_bank.score(va_rgb, self.size)
        geo_val = self.geo_bank.score(va_geo, self.size)
        self.fusion = ReliabilityFusion(self.config.reliability).fit(
            rgb_val, geo_val, validation.foreground, valid_points(validation.xyz).sum(axis=(1, 2))
        )
        # every validation part sits inside its own reference range, so its weights are 1
        ones, zeros = np.ones(len(validation)), np.zeros(len(validation))
        self.calibration = self.fusion.combine(rgb_val, geo_val, Weights(ones, ones, zeros, zeros))[
            0
        ]
        self.threshold = conformal.threshold(self.calibration, self.config.alpha)

    # -- inspecting --------------------------------------------------------------

    def reset(self) -> None:
        """Forget the reliability window, e.g. after a sensor was repaired."""
        window = self.config.reliability.window or 1
        self.history: dict[str, deque[float]] = {
            "rgb": deque(maxlen=window),
            "geo": deque(maxlen=window),
            "seconds": deque(maxlen=window),
        }
        self.alarms: deque[bool] = deque(maxlen=RECENT)
        # seconds per stage for the last part; exact on the CPU, where nothing runs async
        self.timings: dict[str, float] = {}

    def status(self) -> dict[str, Any]:
        """Sensor health over the current window and the recent alarm rate, for a line monitor.

        The alarm rate counts every alarm, defects included, so a rate above the
        expected range means either more defects than usual or normal parts that
        no longer look like the calibration parts (drift). Either is worth a look.
        """
        out: dict[str, Any] = {
            "parts_in_window": len(self.history["rgb"]),
            "calibration_parts": len(self.calibration),
            "threshold": self.threshold if math.isfinite(self.threshold) else None,
        }
        if out["parts_in_window"]:
            u_rgb, u_geo = np.median(self.history["rgb"]), np.median(self.history["geo"])
            w = self.fusion.weights_from(np.array([u_rgb]), np.array([u_geo]))
            assert w.both_off is not None
            out |= {
                "health_rgb": float(u_rgb),
                "health_3d": float(u_geo),
                "weight_rgb": float(w.rgb[0]),
                "weight_3d": float(w.geo[0]),
                "sensor_alert": _alert(w.rgb[0], w.geo[0], bool(w.both_off[0])),
                "ms_per_part": 1000 * float(np.median(self.history["seconds"])),
            }
        if len(self.alarms) >= MIN_FOR_RATE:
            rate = float(np.mean(self.alarms))
            lo, hi = conformal.alarm_rate_range(
                len(self.calibration), self.config.alpha, len(self.alarms)
            )
            out |= {
                "alarm_rate": rate,
                "alarm_rate_expected": [lo, hi],
                "alarm_rate_high": rate > hi,
            }
        return out

    def confirm_good(self, part: int) -> None:
        """An operator checked a recent part and it is good: add it to the calibration set.

        Only parts inspected with both sensors at full weight can be added, because
        only then is their score on the same scale as the calibration scores. More
        calibration parts make the alarm rate tighter, and parts from the current
        run correct a shift between the validation parts and the line.
        """
        if part not in self.recent:
            raise KeyError(f"part {part} is not among the last {RECENT} parts inspected")
        score = self.recent.pop(part)
        if score is None:
            raise ValueError(f"a sensor was down-weighted for part {part}, so it can't calibrate")
        self.calibration = np.append(self.calibration, score)
        self.threshold = conformal.threshold(self.calibration, self.config.alpha)
        self.meta["confirmed"] = self.meta.get("confirmed", 0) + 1

    def inspect(self, rgb: np.ndarray, xyz: np.ndarray, foreground: np.ndarray) -> Inspection:
        """One preprocessed part: (S, S, 3) uint8 image, (S, S, 3) xyz, (S, S) object mask."""
        part = Prepared(
            rgb[None],
            xyz[None],
            foreground[None],
            np.zeros((1, *foreground.shape), bool),
            ["unknown"],
            ["part"],
        )
        timings, start = {}, time.perf_counter()
        last = start

        def lap(stage: str) -> None:
            nonlocal last
            now = time.perf_counter()
            timings[stage], last = now - last, now

        # one stage after the other: open3d holds the GIL, so FPFH on a second thread
        # slows the backbone down as much as it saves (docs/latency.md)
        f_rgb = self.rgb_model.embed(part.rgb)
        lap("backbone")
        r = self.rgb_bank.score(f_rgb, self.size)
        lap("rgb_bank")
        f_geo = self._geo_features(part)
        lap("fpfh")
        g = self.geo_bank.score(f_geo, self.size)
        lap("fpfh_bank")
        points = valid_points(part.xyz).sum(axis=(1, 2))
        u_rgb, u_geo = self.fusion.signals(r, g, part.foreground, points)
        self.history["rgb"].append(float(u_rgb[0]))
        self.history["geo"].append(float(u_geo[0]))
        w = self.fusion.weights_from(
            np.array([np.median(self.history["rgb"])]), np.array([np.median(self.history["geo"])])
        )
        scores, maps = self.fusion.combine(r, g, w)
        score = float(scores[0])
        self.parts_seen += 1
        healthy = w.rgb[0] == 1 and w.geo[0] == 1
        self.recent[self.parts_seen] = score if healthy else None
        self.recent.pop(self.parts_seen - RECENT, None)
        self.alarms.append(score > self.threshold)
        assert w.both_off is not None
        out = Inspection(
            score=score,
            p_value=float(conformal.p_values(self.calibration, scores)[0]),
            alarm=score > self.threshold,
            weight_rgb=float(w.rgb[0]),
            weight_3d=float(w.geo[0]),
            health_rgb=float(w.u_rgb[0]),
            health_3d=float(w.u_geo[0]),
            score_rgb=float(r[0][0]),
            score_3d=float(g[0][0]),
            anomaly_map=maps[0].astype(np.float32),
            map_rgb=self.fusion.scale_rgb(r)[1][0].astype(np.float32),
            map_3d=self.fusion.scale_geo(g)[1][0].astype(np.float32),
            part=self.parts_seen,
            both_off=bool(w.both_off[0]),
        )
        lap("fusion")
        timings["total"] = last - start
        self.history["seconds"].append(timings["total"])
        self.timings = timings
        return out

    def inspect_raw(self, rgb: np.ndarray, xyz: np.ndarray) -> Inspection:
        """A scan as it comes from the sensors, at full resolution."""
        start = time.perf_counter()
        rgb, xyz, fg = cache.prepare_sample(rgb, xyz, self.config.sweep.settings)
        prepared = time.perf_counter() - start
        out = self.inspect(rgb, xyz, fg)
        self.timings = {"preprocess": prepared, **self.timings}
        self.timings["total"] += prepared
        self.history["seconds"][-1] += prepared
        return out

    # -- saving ------------------------------------------------------------------

    def save(self, path: str | Path, half: bool = False) -> None:
        """half: store the banks as float16, about half the file, scores change by ~1e-3."""
        assert self.rgb_bank.bank is not None and self.geo_bank.bank is not None
        dtype = torch.float16 if half else torch.float32
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": FORMAT,
                "config": {
                    "sweep": {
                        "rgb": self.config.sweep.rgb.as_dict(),
                        "fpfh": self.config.sweep.fpfh.as_dict(),
                        "settings": asdict(self.config.sweep.settings),
                    },
                    "alpha": self.config.alpha,
                },
                "pretrained": self.pretrained,
                # ImageNet weights come from torchvision; anything else has to travel with the file
                "backbone": None if self.pretrained else self.rgb_model.backbone.state_dict(),
                "size": list(self.size),
                "rgb_bank": self.rgb_bank.bank.cpu().to(dtype),
                "geo_bank": self.geo_bank.bank.cpu().to(dtype),
                "fusion": self.fusion.state(),
                "calibration": self.calibration.tolist(),
                "meta": self.meta,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> Inspector:
        # tensors and plain values only, so a model file cannot run code when loaded
        state = torch.load(path, weights_only=True)
        if state["format"] != FORMAT:
            raise ValueError(f"model file format {state['format']}, this version reads {FORMAT}")
        fusion = ReliabilityFusion.from_state(state["fusion"])
        config = InspectorConfig(
            sweep=_sweep_from(state["config"]["sweep"]),
            reliability=fusion.config,
            alpha=state["config"]["alpha"],
        )
        inspector = cls(config, device=device, pretrained=state["pretrained"])
        if state["backbone"] is not None:
            inspector.rgb_model.backbone.load_state_dict(state["backbone"])
        device_ = inspector.rgb_model.device
        inspector.rgb_bank.bank = state["rgb_bank"].float().to(device_)
        inspector.geo_bank.bank = state["geo_bank"].float().to(device_)
        inspector.size = tuple(state["size"])
        inspector.fusion = fusion
        inspector.calibration = np.asarray(state["calibration"])
        inspector.threshold = conformal.threshold(inspector.calibration, config.alpha)
        inspector.meta = state["meta"]
        return inspector
