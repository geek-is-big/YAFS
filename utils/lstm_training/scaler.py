from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class StandardScaler:
    """Global StandardScaler for RSSI values in dBm."""

    rssi_mean: float
    rssi_std: float

    @classmethod
    def fit(cls, values: np.ndarray) -> "StandardScaler":
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0:
            raise ValueError("Cannot fit StandardScaler on empty values.")
        mean = float(np.mean(array))
        std = float(np.std(array))
        if std <= 0.0:
            std = 1.0
        return cls(rssi_mean=mean, rssi_std=std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((np.asarray(values, dtype=np.float32) - self.rssi_mean) / self.rssi_std).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float32) * self.rssi_std + self.rssi_mean).astype(np.float32)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"rssi_mean": self.rssi_mean, "rssi_std": self.rssi_std}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "StandardScaler":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(rssi_mean=float(payload["rssi_mean"]), rssi_std=float(payload["rssi_std"]))
