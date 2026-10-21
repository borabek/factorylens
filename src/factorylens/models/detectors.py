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


class BTFDetector:
    """RGB and FPFH patches side by side, unscaled, as in the BTF paper.

    FPFH histograms are in the hundreds while ImageNet features are around 1,
    so distances in this joint space are dominated by the 3D part. That is
    kept on purpose here: this class reproduces BTF, it does not fix it.
    """

    name = "btf"

    def __init__(
        self,
        rgb: PatchCoreConfig = BTF_RGB,
        features: fpfh.FPFHConfig | None = None,
        device: str | None = None,
        pretrained: bool = True,
        feature_cache: str | Path | None = None,
    ) -> None:
        self.rgb = PatchCore(rgb, device=device, pretrained=pretrained)
        self.features = features or fpfh.FPFHConfig(grid=rgb.grid or 28)
        self.memory = MemoryBank(rgb.bank_config())
        self.feature_cache = feature_cache

    def _patches(self, p: Prepared) -> torch.Tensor:
        rgb = self.rgb.embed(p.rgb)
        geo = fpfh.patches_for(p, self.features, self.feature_cache).to(rgb.device)
        if rgb.shape[1:3] != geo.shape[1:3]:
            raise ValueError(
                f"grids differ: rgb {tuple(rgb.shape[1:3])}, fpfh {tuple(geo.shape[1:3])}"
            )
        return torch.cat([rgb, geo], dim=-1)

    def fit(self, train: Prepared) -> BTFDetector:
        self.memory.fit(self._patches(train))
        return self

    def score(self, test: Prepared) -> tuple[np.ndarray, np.ndarray]:
        return self.memory.score(self._patches(test), _size(test))

    def config_dict(self) -> dict[str, Any]:
        return {
            "model": "btf",
            "rgb": self.rgb.config.as_dict(),
            "fpfh": self.features.as_dict(),
        }

    @property
    def bank_size(self) -> int:
        return len(self.memory.bank) if self.memory.bank is not None else 0
