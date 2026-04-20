from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config as cfg
from yafs.placement import JSONPlacement


@dataclass
class CityFogDevice:
    dataset_id: int
    topo_id: int
    latitude: float
    longitude: float
    details: str


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute great-circle distance in meters.
    """
    r = 6371000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    d_phi = np.radians(lat2 - lat1)
    d_lam = np.radians(lon2 - lon1)

    a = np.sin(d_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lam / 2.0) ** 2
    return 2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def interpolate_user_path(
    waypoints: List[Tuple[float, float]],
    speed_mps: float = cfg.USER_SPEED_MPS,
    step_seconds: int = cfg.SIM_STEP_SECONDS,
) -> List[Tuple[float, float]]:
    """
    Interpolate user waypoints to one position per simulation step.
    """
    if not waypoints:
        return []
    if len(waypoints) == 1:
        return [waypoints[0]]

    step_distance = speed_mps * step_seconds
    points: List[Tuple[float, float]] = [waypoints[0]]
    current = np.array(waypoints[0], dtype=float)
    target_idx = 1

    while target_idx < len(waypoints):
        target = np.array(waypoints[target_idx], dtype=float)
        segment_m = haversine_distance_m(current[0], current[1], target[0], target[1])
        if segment_m <= step_distance:
            current = target
            points.append((float(current[0]), float(current[1])))
            target_idx += 1
            continue

        ratio = step_distance / segment_m
        current = current + ratio * (target - current)
        points.append((float(current[0]), float(current[1])))

    return points


class MobilityModel:
    def __init__(self, user_trace: List[Tuple[float, float]], fog_devices: List[CityFogDevice]):
        self.user_trace = user_trace
        self.fog_devices = fog_devices
        self._nearest_cache: Dict[int, int] = {}

    def user_position(self, step: int) -> Tuple[float, float]:
        if not self.user_trace:
            return 0.0, 0.0
        idx = min(max(step, 0), len(self.user_trace) - 1)
        return self.user_trace[idx]

    def distances_for_step(self, step: int) -> List[Tuple[CityFogDevice, float]]:
        lat, lon = self.user_position(step)
        return [
            (fog, haversine_distance_m(lat, lon, fog.latitude, fog.longitude))
            for fog in self.fog_devices
        ]

    def nearest_fog_topology_node(self, step: int) -> Optional[int]:
        if step in self._nearest_cache:
            return self._nearest_cache[step]
        distances = self.distances_for_step(step)
        if not distances:
            return None
        nearest = min(distances, key=lambda item: item[1])[0].topo_id
        self._nearest_cache[step] = nearest
        return nearest


class MobilityPlacement(JSONPlacement):
    """
    Keep the Fog module attached to the nearest city fog device to the moving user.
    """

    def __init__(self, mobility_model: MobilityModel, app_name: str, **kwargs):
        super().__init__(**kwargs)
        self.mobility_model = mobility_model
        self.app_name = app_name
        self.last_fog_topo_id = None

    def initial_allocation(self, sim, app_name):
        super().initial_allocation(sim, app_name)
        nearest = self.mobility_model.nearest_fog_topology_node(step=0)
        if nearest is not None and app_name == self.app_name:
            self._move_fog_module(sim, nearest)

    def run(self, sim):
        step = int(sim.env.now)
        nearest = self.mobility_model.nearest_fog_topology_node(step)
        if nearest is None or nearest == self.last_fog_topo_id:
            return
        self._move_fog_module(sim, nearest)

    def _move_fog_module(self, sim, new_topo_id: int) -> None:
        deployed = list(sim.alloc_module.get(self.app_name, {}).get("Fog", []))
        for des in deployed:
            sim.undeploy_module(self.app_name, "Fog", des)
        app = sim.apps[self.app_name]
        sim.deploy_module(self.app_name, "Fog", app.services["Fog"], [new_topo_id])
        self.last_fog_topo_id = new_topo_id


def load_city_fog_devices(dataset_dir: Path, topo_start_id: int = 1000) -> List[CityFogDevice]:
    df = pd.read_csv(dataset_dir / "edgeResources-melbCBD.csv")
    fog_df = df[df["Level"] > 0].copy()
    fog_devices: List[CityFogDevice] = []
    for idx, row in fog_df.reset_index(drop=True).iterrows():
        fog_devices.append(
            CityFogDevice(
                dataset_id=int(row["ID"]),
                topo_id=topo_start_id + idx,
                latitude=float(row["Latitude"]),
                longitude=float(row["Longitude"]),
                details=str(row["Details"]),
            )
        )
    return fog_devices


def load_user_trace(
    dataset_dir: Path,
    speed_mps: float = cfg.USER_SPEED_MPS,
    step_seconds: int = cfg.SIM_STEP_SECONDS,
) -> List[Tuple[float, float]]:
    df = pd.read_csv(dataset_dir / "usersLocation-melbCBD_1.csv")
    waypoints = list(zip(df["Latitude"].astype(float), df["Longitude"].astype(float)))
    return interpolate_user_path(waypoints, speed_mps=speed_mps, step_seconds=step_seconds)
