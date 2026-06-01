from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.lstm_dataset_gen.compat import haversine_distance_m, rssi_from_distance_dbm
from utils.lstm_dataset_gen.graph import load_fog_nodes
from .scaler import StandardScaler


RSSI_COLUMN_PATTERN = re.compile(r"^RSSI_FOG_(\d+)$")


def sorted_rssi_columns(frame: pd.DataFrame) -> List[str]:
    pairs = []
    for column in frame.columns:
        match = RSSI_COLUMN_PATTERN.match(str(column))
        if match:
            pairs.append((int(match.group(1)), str(column)))
    if not pairs:
        raise ValueError("No RSSI_FOG_<id> columns found.")
    return [column for _, column in sorted(pairs)]


def fog_ids_from_columns(columns: Sequence[str]) -> List[int]:
    fog_ids = []
    for column in columns:
        match = RSSI_COLUMN_PATTERN.match(str(column))
        if match is None:
            raise ValueError(f"Invalid RSSI column name: {column}")
        fog_ids.append(int(match.group(1)))
    return fog_ids


def load_rssi_frame(csv_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    frame = pd.read_csv(csv_path)
    columns = sorted_rssi_columns(frame)
    return frame, columns


def build_lstm_input_from_route_window(rssi_window: np.ndarray, use_delta: bool) -> np.ndarray:
    rssi = np.asarray(rssi_window, dtype=np.float32)
    if rssi.ndim != 2:
        raise ValueError("rssi_window must have shape (history_len, num_fog_nodes).")
    if not use_delta:
        return rssi
    delta = np.diff(rssi, axis=0, prepend=rssi[:1])
    return np.concatenate([rssi, delta.astype(np.float32)], axis=1).astype(np.float32)


class RouteWindowDataset:
    """Lazy route-grouped sliding-window dataset for RSSI forecasting."""

    def __init__(
        self,
        frame: pd.DataFrame,
        rssi_columns: Sequence[str],
        history_len: int,
        pred_horizon: int,
        use_delta: bool,
        scaler: Optional[StandardScaler] = None,
    ) -> None:
        self.rssi_columns = list(rssi_columns)
        self.history_len = int(history_len)
        self.pred_horizon = int(pred_horizon)
        self.use_delta = bool(use_delta)
        self.scaler = scaler
        self.routes: List[np.ndarray] = []
        self.index: List[Tuple[int, int]] = []

        if "route_id" not in frame.columns:
            frame = frame.copy()
            frame["route_id"] = "route_0000"

        sort_columns = ["route_id"]
        if "time" in frame.columns:
            sort_columns.append("time")
        frame = frame.sort_values(sort_columns)

        for _, group in frame.groupby("route_id", sort=False):
            values = group.loc[:, self.rssi_columns].to_numpy(dtype=np.float32)
            if scaler is not None:
                values = scaler.transform(values)
            route_index = len(self.routes)
            self.routes.append(values)
            max_start = len(values) - self.history_len - self.pred_horizon
            for start in range(max_start + 1):
                self.index.append((route_index, start))

    def __len__(self) -> int:
        return len(self.index)

    @property
    def num_fog_nodes(self) -> int:
        return len(self.rssi_columns)

    def __getitem__(self, idx: int):
        import torch

        route_index, start = self.index[idx]
        route = self.routes[route_index]
        history_end = start + self.history_len
        target_end = history_end + self.pred_horizon
        x = build_lstm_input_from_route_window(route[start:history_end], self.use_delta)
        y = route[history_end:target_end]
        return torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))


def build_fixed_route_rssi_frame(route_csv_path: Path, fog_csv_path: Path) -> pd.DataFrame:
    route = pd.read_csv(route_csv_path)
    lat_col, lon_col = _lat_lon_columns(route.columns)
    fog_nodes = load_fog_nodes(Path(fog_csv_path))

    rows = []
    for step, (_, row) in enumerate(route.iterrows()):
        latitude = float(row[lat_col])
        longitude = float(row[lon_col])
        output = {
            "route_id": "fixed_eval_route",
            "time": step,
            "latitude": latitude,
            "longitude": longitude,
        }
        for fog in fog_nodes:
            distance_m = haversine_distance_m(latitude, longitude, fog.latitude, fog.longitude)
            output[f"RSSI_FOG_{fog.fog_id}"] = rssi_from_distance_dbm(distance_m)
        rows.append(output)
    return pd.DataFrame(rows)


def _lat_lon_columns(columns: Iterable[str]) -> Tuple[str, str]:
    by_lower = {str(column).strip().lower(): str(column) for column in columns}
    lat = by_lower.get("latitude") or by_lower.get("lat")
    lon = by_lower.get("longitude") or by_lower.get("lon") or by_lower.get("lng")
    if lat is None or lon is None:
        raise ValueError("Route CSV must contain Latitude/Longitude columns.")
    return lat, lon
