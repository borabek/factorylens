"""The three detectors behind one interface: fit on a prepared train split,
score a prepared test split, get image scores and pixel maps back.

rgb   PatchCore on the colour image
fpfh  the same memory bank on FPFH point-cloud features
btf   Back to the Feature: RGB and FPFH patch features concatenated as they
      are, one memory bank on the result
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from factorylens.data.cache import Prepared
from factorylens.models import fpfh
from factorylens.models.memory_bank import BankConfig, MemoryBank
from factorylens.models.patchcore import BTF_RGB, PatchCore, PatchCoreConfig


class Detector(Protocol):
    name: str

    def fit(self, train: Prepared) -> Detector: ...

    def score(self, test: Prepared) -> tuple[np.ndarray, np.ndarray]: ...

    def config_dict(self) -> dict[str, Any]: ...

    @property
    def bank_size(self) -> int: ...


def _size(p: Prepared) -> tuple[int, int]:
    return (p.rgb.shape[1], p.rgb.shape[2])


class RGBDetector:
    name = "rgb"

    def __init__(
        self,
        config: PatchCoreConfig | None = None,
        device: str | None = None,
        pretrained: bool = True,
    ) -> None:
        self.model = PatchCore(config, device=device, pretrained=pretrained)

    def fit(self, train: Prepared) -> RGBDetector:
        self.model.fit(train.rgb)
        return self

    def score(self, test: Prepared) -> tuple[np.ndarray, np.ndarray]:
        return self.model.score(test.rgb)

    def config_dict(self) -> dict[str, Any]:
        return {"model": "patchcore_rgb", **self.model.config.as_dict()}

    @property
    def bank_size(self) -> int:
        return len(self.model.bank) if self.model.bank is not None else 0


@dataclass(frozen=True)
class FPFHDetectorConfig:
    features: fpfh.FPFHConfig = field(default_factory=fpfh.FPFHConfig)
    bank: BankConfig = field(default_factory=BankConfig)


class FPFHDetector:
    name = "fpfh"

    def __init__(
        self, config: FPFHDetectorConfig | None = None, feature_cache: str | Path | None = None
    ) -> None:
        self.config = config or FPFHDetectorConfig()
        self.memory = MemoryBank(self.config.bank)
        self.feature_cache = feature_cache

    def fit(self, train: Prepared) -> FPFHDetector:
        self.memory.fit(fpfh.patches_for(train, self.config.features, self.feature_cache))
        return self

    def score(self, test: Prepared) -> tuple[np.ndarray, np.ndarray]:
        patches = fpfh.patches_for(test, self.config.features, self.feature_cache)
        return self.memory.score(patches, _size(test))

    def config_dict(self) -> dict[str, Any]:
        return {
            "model": "fpfh",
            "fpfh": self.config.features.as_dict(),
            "bank": self.config.bank.__dict__,
        }

    @property
    def bank_size(self) -> int:
        return len(self.memory.bank) if self.memory.bank is not None else 0


def spread(x: torch.Tensor) -> float:
    """Mean distance of a feature vector to the mean vector."""
    flat = x.reshape(-1, x.shape[-1])
    return float((flat - flat.mean(0)).norm(dim=1).mean())


class BTFDetector:
    """RGB and FPFH patches side by side, as in the BTF paper.

    FPFH histograms are in the hundreds while ImageNet features are around 1,
    so with the default fpfh_weight=1 (BTF's choice) distances in this joint
    space are dominated by the 3D part. fpfh_weight scales the FPFH half
    before concatenation; "balanced" picks the factor on the training patches
    so that both halves have the same spread.
    """

    name = "btf"

    def __init__(
        self,
        rgb: PatchCoreConfig = BTF_RGB,
        features: fpfh.FPFHConfig | None = None,
        device: str | None = None,
        pretrained: bool = True,
        feature_cache: str | Path | None = None,
        fpfh_weight: float | str = 1.0,
    ) -> None:
        if isinstance(fpfh_weight, str) and fpfh_weight != "balanced":
            raise ValueError(f"fpfh_weight must be a number or 'balanced', not {fpfh_weight!r}")
        self.rgb = PatchCore(rgb, device=device, pretrained=pretrained)
        self.features = features or fpfh.FPFHConfig(grid=rgb.grid or 28)
        self.memory = MemoryBank(rgb.bank_config())
        self.feature_cache = feature_cache
        self.fpfh_weight = fpfh_weight
        self.fitted_weight: float | None = None if fpfh_weight == "balanced" else float(fpfh_weight)

    def parts(self, p: Prepared) -> tuple[torch.Tensor, torch.Tensor]:
        """RGB and FPFH patch features of a split, before weighting."""
        rgb = self.rgb.embed(p.rgb)
        geo = fpfh.patches_for(p, self.features, self.feature_cache).to(rgb.device)
        if rgb.shape[1:3] != geo.shape[1:3]:
            raise ValueError(
                f"grids differ: rgb {tuple(rgb.shape[1:3])}, fpfh {tuple(geo.shape[1:3])}"
            )
        return rgb, geo

    def patches(self, p: Prepared) -> torch.Tensor:
        rgb, geo = self.parts(p)
        if self.fitted_weight is None:
            raise RuntimeError("call fit() first, the balanced weight comes from the train split")
        return torch.cat([rgb, geo * self.fitted_weight], dim=-1)

    def fit(self, train: Prepared) -> BTFDetector:
        if self.fpfh_weight == "balanced":
            rgb, geo = self.parts(train)
            self.fitted_weight = spread(rgb) / spread(geo)
        self.memory.fit(self.patches(train))
        return self

    def score(self, test: Prepared) -> tuple[np.ndarray, np.ndarray]:
        return self.memory.score(self.patches(test), _size(test))

    def config_dict(self) -> dict[str, Any]:
        return {
            "model": "btf",
            "rgb": self.rgb.config.as_dict(),
            "fpfh": self.features.as_dict(),
            "fpfh_weight": self.fpfh_weight,
            "fpfh_weight_fitted": self.fitted_weight,
        }

    @property
    def bank_size(self) -> int:
        return len(self.memory.bank) if self.memory.bank is not None else 0
