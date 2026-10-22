"""Cycle-level anomaly detectors for the Future Factories robot signals.

Every assembly cycle becomes a fixed-length profile: the signals binned to 1 s
since the cycle started, (seconds, channels). All detectors are fitted on
normal cycles only and give one score per cycle, higher = more anomalous.

  profile_z   how far each second sits from the normal mean, in normal
              standard deviations, averaged over the cycle
  iforest     Isolation Forest on 10 s averages of the profile
  forecast    a ridge model predicts each next second from the previous few
              and from where in the cycle it is (Fourier terms of the phase);
              the score is the cycle's mean squared prediction error

Two baselines go with them. `Level` looks only at each channel's mean over the
cycle, so it sees slow drift but not the shape of the motion. Time of day is
the other one, computed in the script, because in this data the labels come
in contiguous blocks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge

from factorylens.data import ff

JOINTS = ("S", "L", "U", "R", "B", "T")
ROBOTS = ("R01", "R02", "R03", "R04")


def robot_channels(robot: str) -> tuple[str, ...]:
    return (
        f"I_{robot}_Gripper_Load",
        f"I_{robot}_Gripper_Pot",
        *(f"M_{robot}_{j}JointAngle_Degree" for j in JOINTS),
    )


ROBOT_CHANNELS = tuple(c for r in ROBOTS for c in robot_channels(r))
TEMPERATURES = tuple(f"Q_VFD{i}_Temperature" for i in range(1, 5))


def cycle_profiles(
    df: pd.DataFrame, cycles: pd.DataFrame, channels: tuple[str, ...], seconds: int = 295
) -> np.ndarray:
    """(n_cycles, seconds, channels): per-second medians, gaps filled from neighbours."""
    ids = ff.cycle_ids(df["Q_Cell_CycleCount"]).to_numpy()
    out = np.empty((len(cycles), seconds, len(channels)))
    for i, (cid, c) in enumerate(cycles.iterrows()):
        part = df.loc[ids == cid, list(channels)].astype(float)
        t = (part.index - c["start"]).total_seconds().astype(int)
        binned = part.groupby(t).median().reindex(range(seconds))
        out[i] = binned.ffill().bfill().to_numpy()
    return out


def usable(profiles: np.ndarray) -> np.ndarray:
    """Channels that were logged and moved in every one of these cycles."""
    logged = np.isfinite(profiles).all(axis=(0, 1))
    moved = np.nan_to_num(profiles).std(axis=(0, 1)) > 0
    return np.asarray(logged & moved)


@dataclass
class Standardiser:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, profiles: np.ndarray) -> Standardiser:
        flat = profiles.reshape(-1, profiles.shape[-1])
        return cls(np.nanmean(flat, axis=0), np.nanstd(flat, axis=0) + 1e-9)

    def __call__(self, profiles: np.ndarray) -> np.ndarray:
        return np.asarray(np.nan_to_num((profiles - self.mean) / self.std))


class ProfileZ:
    def fit(self, train: np.ndarray) -> ProfileZ:
        self.mean = train.mean(axis=0)
        self.std = train.std(axis=0) + 1e-6 * np.abs(train).mean() + 1e-9
        return self

    def score(self, profiles: np.ndarray) -> np.ndarray:
        return np.asarray(np.abs((profiles - self.mean) / self.std).mean(axis=(1, 2)))


class Level(ProfileZ):
    """Profile z-score on cycle means only: what drift alone can do."""

    def fit(self, train: np.ndarray) -> Level:
        super().fit(train.mean(axis=1, keepdims=True))
        return self

    def score(self, profiles: np.ndarray) -> np.ndarray:
        return super().score(profiles.mean(axis=1, keepdims=True))


def demean(profiles: np.ndarray) -> np.ndarray:
    """Subtract each cycle's own channel means, which removes slow drift."""
    return profiles - profiles.mean(axis=1, keepdims=True)


class IForest:
    def __init__(self, bin_s: int = 10, seed: int = 0) -> None:
        self.bin_s, self.seed = bin_s, seed

    def _features(self, profiles: np.ndarray) -> np.ndarray:
        n, s, c = profiles.shape
        s = s - s % self.bin_s
        binned = profiles[:, :s].reshape(n, s // self.bin_s, self.bin_s, c).mean(axis=2)
        return np.asarray(self.scale(binned).reshape(n, -1))

    def fit(self, train: np.ndarray) -> IForest:
        self.scale = Standardiser.fit(train)
        self.model = IsolationForest(n_estimators=300, random_state=self.seed)
        self.model.fit(self._features(train))
        return self

    def score(self, profiles: np.ndarray) -> np.ndarray:
        return np.asarray(-self.model.score_samples(self._features(profiles)))


class Forecast:
    """Next-second prediction from lags and the phase within the cycle."""

    def __init__(self, lags: int = 5, harmonics: int = 6, alpha: float = 1.0) -> None:
        self.lags, self.harmonics, self.alpha = lags, harmonics, alpha

    def _design(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n, s, c = z.shape
        t = np.arange(self.lags, s)
        phase = 2 * np.pi * t / s
        fourier = np.column_stack(
            [f(k * phase) for k in range(1, self.harmonics + 1) for f in (np.sin, np.cos)]
        )
        lagged = np.stack([z[:, self.lags - j : s - j] for j in range(1, self.lags + 1)], axis=-1)
        x = np.concatenate(
            [
                lagged.reshape(n, len(t), c * self.lags),
                np.broadcast_to(fourier, (n, *fourier.shape)),
            ],
            axis=-1,
        )
        return x, z[:, self.lags :]

    def fit(self, train: np.ndarray) -> Forecast:
        self.scale = Standardiser.fit(train)
        x, y = self._design(self.scale(train))
        self.model = Ridge(alpha=self.alpha).fit(
            x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1])
        )
        return self

    def residuals(self, profiles: np.ndarray) -> np.ndarray:
        """(n_cycles, seconds - lags, channels) prediction errors."""
        x, y = self._design(self.scale(profiles))
        pred = self.model.predict(x.reshape(-1, x.shape[-1])).reshape(y.shape)
        return np.asarray(y - pred)

    def score(self, profiles: np.ndarray) -> np.ndarray:
        return np.asarray((self.residuals(profiles) ** 2).mean(axis=(1, 2)))


DETECTORS = {"profile_z": ProfileZ, "iforest": IForest, "forecast": Forecast}
