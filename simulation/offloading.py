"""
    Offloading simulation
"""
import os
import time
import json
import random
import logging.config
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

import pandas as pd
import numpy as np

from yafs.core import Sim
from yafs.application import Application, Message, fractional_selectivity
from yafs.topology import Topology
from yafs.population import Statical
from yafs.placement import JSONPlacement
from yafs.selection import First_ShortestPath
from yafs.distribution import deterministic_distribution, deterministicDistributionStartPoint
from yafs.stats import Stats

# from jsonAllocation import JSONPopulation

class TimeUnit(Enum):
    SECOND = 1
    MILLISECOND = 2

APP_NAME = "CityScenario"
TIME_UNIT = TimeUnit.SECOND
USER_SPEED_MPS = 1.4
SIM_STEP_SECONDS = 1
ENABLE_ANIMATION = True
ANIMATION_STEP_STRIDE = 5
ANIMATION_MAX_FRAMES = 300
ANIMATION_FORMAT = "gif"  # "mp4" or "gif"
PRINT_NEAREST_DISTANCES = False
NEAREST_NODES_TO_PRINT = 3
RSSI_REFERENCE_DISTANCE_M = 1.0
RSSI_AT_REFERENCE_DBM = -20.0
RSSI_ENVIRONMENT_COEFF = 2.7
WIFI_RETRANSMISSION_RTT_S = 0.03  # 30ms

# WiFi throughput model (YAFS BW units are MB/s).
WIFI_MAX_BW_MBPS = 54.0 / 8.0
WIFI_MEDIUM_BW_MBPS = 11.0 / 8.0
WIFI_MIN_BW_MBPS = 1.0 / 8.0

# Galaxy S4 WiFi tail model.
WIFI_TAIL_TIME_S = 0.210
WIFI_TAIL_POWER_W = 0.289

# Sensor WiFi model.
SENSOR_WIFI_TAIL_TIME_S = 0.18
SENSOR_WIFI_TAIL_POWER_W = 0.1212
SENSOR_WIFI_PROMOTION_TIME_S = 0.30
SENSOR_WIFI_PROMOTION_POWER_W = 0.2425

# Sensor BLE model (sensor <-> edge, near distance ~0.5m).
SENSOR_BLE_TAIL_TIME_S = 4.77
SENSOR_BLE_TAIL_POWER_W = 0.0341
SENSOR_BLE_TX_POWER_W = 0.1115
SENSOR_BLE_RX_POWER_W = 0.1172
SENSOR_BLE_DATA_BW_MBPS = 0.305

# Task1 profile (ECG classification)
TASK1_ID = "Task1"
TASK1_PERIOD_S = 10
TASK1_COMPLEXITY_MI = 500
TASK1_DATA_SIZE_KB = 36
TASK1_MAX_RESPONSE_TIME_S = 15
TASK1_CLASSIFICATION = "critical analysis"
TASK_EXECUTION_MODE = "EDGE"  # EDGE | FOG
CURRENT_EXECUTION_MODE = TASK_EXECUTION_MODE
OPTIMIZATION_METHOD = "LP"


def node_model_name(base_name: str) -> str:
    return f"{base_name}-{OPTIMIZATION_METHOD.lower()}"

ALPHA_LATENCY = 0.0
BETA_ENERGY = 1.0
# OFFLOADING_DECISION_PERIOD_S = TASK1_PERIOD_S
OFFLOADING_DECISION_PERIOD_S = 1
TASK1_MIGRATION_STATE_SIZE_KB = 20.0


def get_task_processing_module() -> str:
    mode = str(TASK_EXECUTION_MODE).upper()
    if mode == "EDGE":
        return "Mobile"
    if mode == "FOG":
        return "Fog"
    raise ValueError(f"Unsupported TASK_EXECUTION_MODE={TASK_EXECUTION_MODE}")


def task_mode_selectivity(threshold: float, mode: str) -> bool:
    """
    Runtime switch between EDGE/FOG processing chains.
    """
    return str(mode).upper() == str(CURRENT_EXECUTION_MODE).upper() and fractional_selectivity(threshold)


def kb_to_mb(kb: float) -> float:
    return kb / 1024.0


def mi_to_instructions(mi: float) -> float:
    return mi * 10**6


def solve_lp_two_mode(c_edge: float, c_fog: float) -> str:
    """
    LP:
      min c_edge*x_edge + c_fog*x_fog
      s.t. x_edge + x_fog = 1, x>=0
    """
    return "EDGE" if c_edge <= c_fog else "FOG"


def build_simulation_tag() -> str:
    alpha_str = str(ALPHA_LATENCY).replace(".", "p")
    beta_str = str(BETA_ENERGY).replace(".", "p")
    return f"{APP_NAME}_{OPTIMIZATION_METHOD}_a{alpha_str}_b{beta_str}"


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
    r = 6371000.0  # meters
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    d_phi = np.radians(lat2 - lat1)
    d_lam = np.radians(lon2 - lon1)

    a = np.sin(d_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lam / 2.0) ** 2
    return 2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def rssi_from_distance_dbm(
    distance_m: float,
    d0_m: float = RSSI_REFERENCE_DISTANCE_M,
    rssi0_dbm: float = RSSI_AT_REFERENCE_DBM,
    n: float = RSSI_ENVIRONMENT_COEFF,
) -> float:
    """
    Log-distance path loss model:
    RSSI = RSSI0 - 10*n*log10(d/d0)
    """
    # BLE Beacons for Indoor Positioning at an Interactive IoT-Based Smart Museum - arXiv:2001.07686 [cs.NI]
    # Surpassing Bluetooth Low Energy Limitations on Distance Determination - DOI: 10.1109/EPEPEMC.2016.7752104
    if d0_m <= 0:
        raise ValueError("Reference distance d0_m must be > 0")
    # Avoid log10(0) and cap near-field values to RSSI0 when d <= d0.
    effective_d = max(distance_m, d0_m)
    return rssi0_dbm - (10.0 * n * np.log10(effective_d / d0_m))


def seconds_to_tu(seconds: float) -> float:
    if TIME_UNIT == TimeUnit.SECOND:
        return seconds
    if TIME_UNIT == TimeUnit.MILLISECOND:
        return seconds * 1000.0
    raise ValueError("Unknown TIME_UNIT")


def tu_to_seconds(time_unit_value: float) -> float:
    if TIME_UNIT == TimeUnit.SECOND:
        return time_unit_value
    if TIME_UNIT == TimeUnit.MILLISECOND:
        return time_unit_value / 1000.0
    raise ValueError("Unknown TIME_UNIT")


def wifi_throughput_mbps_from_rssi(rssi_dbm: float) -> float:
    """
    Throughput model from RSSI:
    - [-50, -70] dBm -> 54 Mbps
    - [-70, -85] dBm -> 11 Mbps
    - -90 dBm -> 1 Mbps
    with linear interpolation between -85 and -90.
    """
    # Characterizing and modeling the impact of wireless signal strength on smartphone battery drain
    # DOI: 10.1145/2494232.2466586
    if rssi_dbm >= -70.0:
        return ips_to_ipt(WIFI_MAX_BW_MBPS)
    if rssi_dbm >= -85.0:
        return ips_to_ipt(WIFI_MEDIUM_BW_MBPS)
    if rssi_dbm <= -90.0:
        return ips_to_ipt(WIFI_MIN_BW_MBPS)

    # Linear interpolation on (-90, -85) for intermediate weak-signal values.
    # rssi=-90 -> 1 Mbps, rssi=-85 -> 11 Mbps
    x = np.array([-90.0, -85.0], dtype=float)
    y = np.array([WIFI_MIN_BW_MBPS, WIFI_MEDIUM_BW_MBPS], dtype=float)
    return ips_to_ipt(float(np.interp(rssi_dbm, x, y)))


def tcp_retransmission_rate_from_rssi(rssi_dbm: float) -> float:
    """
    TCP retransmission rate model with piecewise linear interpolation:
    -60 dBm: 0%
    -70 dBm: 20%
    -80 dBm: 30%
    -85 dBm: 55%
    -90 dBm: 90%
    """
    # Characterizing and modeling the impact of wireless signal strength on smartphone battery drain
    # DOI: 10.1145/2494232.2466586
    if rssi_dbm >= -60.0:
        return 0.0
    if rssi_dbm <= -90.0:
        return 0.9

    x = np.array([-90.0, -85.0, -80.0, -70.0, -60.0], dtype=float)
    y = np.array([0.9, 0.55, 0.3, 0.2, 0.0], dtype=float)
    return float(np.interp(rssi_dbm, x, y))


def galaxy_s4_wifi_powers_w_from_rssi(rssi_dbm: float) -> Tuple[float, float]:
    """
    RSSI-dependent WiFi Tx/Rx power model (Galaxy S4), values in Watts.

    Source points (mW):
      -50: Tx 654, Rx 451
      -60: Tx 723, Rx 528
      -70: Tx 1019, Rx 592
      -80: Tx 1113, Rx 633
      -85: Tx 892, Rx 514
    For -90 and below we extrapolate the -80..-85 downtrend once and clamp.
    """
    x = np.array([-90.0, -85.0, -80.0, -70.0, -60.0, -50.0], dtype=float)

    # Extrapolated -90 values from -80..-85 trend (decreasing on weak signal).
    tx_mw_m90 = 671.0
    rx_mw_m90 = 395.0

    tx_mw = np.array([tx_mw_m90, 892.0, 1113.0, 1019.0, 723.0, 654.0], dtype=float)
    rx_mw = np.array([rx_mw_m90, 514.0, 633.0, 592.0, 528.0, 451.0], dtype=float)

    if rssi_dbm <= -90.0:
        tx = tx_mw_m90
        rx = rx_mw_m90
    elif rssi_dbm >= -50.0:
        tx = tx_mw[-1]
        rx = rx_mw[-1]
    else:
        tx = float(np.interp(rssi_dbm, x, tx_mw))
        rx = float(np.interp(rssi_dbm, x, rx_mw))

    return tx / 1000.0, rx / 1000.0


def transport_protocol_from_message_name(message_name: str) -> str:
    """
    Message naming convention:
    - names containing 'TCP' -> retransmission-aware transport
    - names containing 'UDP' -> best-effort transport
    - default fallback: UDP
    """
    upper = str(message_name).upper()
    if "TCP" in upper:
        return "TCP"
    if "UDP" in upper:
        return "UDP"
    return "UDP"


def expected_tcp_retries_per_packet(retransmission_rate: float) -> float:
    """
    Expected retries for geometric success model.
    """
    clamped = min(max(retransmission_rate, 0.0), 0.999999)
    return clamped / (1.0 - clamped)


def energy_wh_per_mb(throughput_mb_s: float, tx_power_w: float, rx_power_w: float) -> float:
    """
    One-way transfer energy (TX+RX) per transferred MB.
    """
    safe_bw = max(throughput_mb_s, 1e-12)
    transfer_time_tu = 1.0 / safe_bw
    return (watt_to_wpt(tx_power_w) + watt_to_wpt(rx_power_w)) * transfer_time_tu


def fixed_overhead_energy_wh(power_w: float, duration_s: float) -> float:
    return watt_to_wpt(power_w) * seconds_to_tu(duration_s)


def tail_energy_wh_per_transfer() -> float:
    """
    Fixed WiFi tail energy after data exchange completion.
    """
    return watt_to_wpt(WIFI_TAIL_POWER_W) * seconds_to_tu(WIFI_TAIL_TIME_S)


def sensor_wifi_data_powers_w_from_rssi(rssi_dbm: float) -> Tuple[float, float, bool]:
    """
    Sensor WiFi data power model (W) by RSSI.
    At <= -70 dBm we mark link as unreliable for this sensor antenna.
    """
    # Characterizing Smartwatch Usage in the Wild - DOI: 10.1145/3081333.3081351
    # Values in mW:
    # -42: Tx 669.1, Rx 378.5
    # -55: Tx 672.8, Rx 343.0
    # -65: Tx 840.7, Rx 252.3
    # -70: hard to communicate
    if rssi_dbm <= -70.0:
        return np.nan, np.nan, False

    x = np.array([-65.0, -55.0, -42.0], dtype=float)
    tx_mw = np.array([840.7, 672.8, 669.1], dtype=float)
    rx_mw = np.array([252.3, 343.0, 378.5], dtype=float)

    if rssi_dbm <= -65.0:
        tx = tx_mw[0]
        rx = rx_mw[0]
    elif rssi_dbm >= -42.0:
        tx = tx_mw[-1]
        rx = rx_mw[-1]
    else:
        tx = float(np.interp(rssi_dbm, x, tx_mw))
        rx = float(np.interp(rssi_dbm, x, rx_mw))

    return tx / 1000.0, rx / 1000.0, True


def sensor_wifi_energy_wh_per_mb(rssi_dbm: float, throughput_mb_s: float) -> Tuple[float, float, float, float, float, float, bool]:
    """
    Returns total per-MB energy and components for sensor WiFi:
    data + promotion + tail.
    """
    tx_w, rx_w, available = sensor_wifi_data_powers_w_from_rssi(rssi_dbm)
    if not available:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False

    data_wh_per_mb = energy_wh_per_mb(throughput_mb_s, tx_w, rx_w)
    promotion_wh = fixed_overhead_energy_wh(SENSOR_WIFI_PROMOTION_POWER_W, SENSOR_WIFI_PROMOTION_TIME_S)
    tail_wh = fixed_overhead_energy_wh(SENSOR_WIFI_TAIL_POWER_W, SENSOR_WIFI_TAIL_TIME_S)
    total_wh_per_mb = data_wh_per_mb + promotion_wh + tail_wh
    return total_wh_per_mb, data_wh_per_mb, promotion_wh, tail_wh, tx_w, rx_w, True


def sensor_ble_energy_wh_per_mb() -> Tuple[float, float, float]:
    """
    Sensor BLE per-MB energy model (data + tail).
    """
    # Characterizing Smartwatch Usage in the Wild - DOI: 10.1145/3081333.3081351
    # Sensors could be connected via BLE with Edge device, with typical distance around 0.5m
    # and Tx/Rx power around 110mW. BLE tail is long (4.77s) but low power (34mW).
    data_wh_per_mb = energy_wh_per_mb(
        throughput_mb_s=ips_to_ipt(SENSOR_BLE_DATA_BW_MBPS),
        tx_power_w=SENSOR_BLE_TX_POWER_W,
        rx_power_w=SENSOR_BLE_RX_POWER_W,
    )
    tail_wh = fixed_overhead_energy_wh(SENSOR_BLE_TAIL_POWER_W, SENSOR_BLE_TAIL_TIME_S)
    total_wh_per_mb = data_wh_per_mb + tail_wh
    return total_wh_per_mb, data_wh_per_mb, tail_wh


def interpolate_user_path(
    waypoints: List[Tuple[float, float]],
    speed_mps: float = USER_SPEED_MPS,
    step_seconds: int = SIM_STEP_SECONDS,
) -> List[Tuple[float, float]]:
    """
    Interpolates user waypoints to one position per simulation step.
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
    Keeps the Fog module attached to the nearest city fog device to the moving user.
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


class MobilityDistanceMonitor:
    """
    Stores per-second distances between the user and every city fog device.
    """
    def __init__(self, mobility_model: MobilityModel, output_path: Path, flush_every_steps: int = 30):
        self.mobility_model = mobility_model
        self.output_path = output_path
        self.flush_every_steps = flush_every_steps
        self.buffer = []
        self.header_written = False
        self.app_name = APP_NAME
        self.mobile_node_id = 2

    def _current_fog_topology_id(self, sim) -> Optional[int]:
        fog_des = sim.alloc_module.get(self.app_name, {}).get("Fog", [])
        if not fog_des:
            return None
        return sim.alloc_DES.get(fog_des[0], None)

    def _current_mobile_topology_id(self, sim) -> Optional[int]:
        mobile_des = sim.alloc_module.get(self.app_name, {}).get("Mobile", [])
        if not mobile_des:
            return None
        return sim.alloc_DES.get(mobile_des[0], None)

    def _current_task_processing_topology_id(self, sim) -> Optional[int]:
        mode = str(CURRENT_EXECUTION_MODE).upper()
        if mode == "FOG":
            return self._current_fog_topology_id(sim)
        return self._current_mobile_topology_id(sim)

    def _current_execution_mode(self, sim) -> str:
        """
        Runtime mode follows LP decision, validated by actual placement.
        """
        mode = str(CURRENT_EXECUTION_MODE).upper()
        if mode == "FOG" and self._current_fog_topology_id(sim) is not None:
            return "FOG"
        if self._current_mobile_topology_id(sim) is None:
            return "EDGE"
        return "EDGE"

    def run(self, sim):
        step = int(sim.env.now)
        user_lat, user_lon = self.mobility_model.user_position(step)
        distances = self.mobility_model.distances_for_step(step)
        if not distances:
            return

        distances_sorted = sorted(distances, key=lambda item: item[1])
        nearest_topo_id = distances_sorted[0][0].topo_id

        if PRINT_NEAREST_DISTANCES:
            nearest_n = distances_sorted[:max(1, NEAREST_NODES_TO_PRINT)]
            nearest_info = ", ".join(
                f"{fog.topo_id}:{dist_m:.2f}m" for fog, dist_m in nearest_n
            )
            print(f"[t={step:4d}s] nearest_fog_nodes -> {nearest_info}")

        execution_mode = self._current_execution_mode(sim)
        current_fog_topology_id = self._current_fog_topology_id(sim)
        current_mobile_topology_id = self._current_mobile_topology_id(sim)
        task_processing_topology_id = self._current_task_processing_topology_id(sim)
        sensor_ble_total_wh_per_mb, sensor_ble_data_wh_per_mb, sensor_ble_tail_wh = sensor_ble_energy_wh_per_mb()

        for fog, distance_m in distances:
            rssi_dbm = rssi_from_distance_dbm(distance_m)
            wifi_bw_mb_s = wifi_throughput_mbps_from_rssi(rssi_dbm)
            tcp_retx_rate = tcp_retransmission_rate_from_rssi(rssi_dbm)
            tcp_expected_retries = expected_tcp_retries_per_packet(tcp_retx_rate)
            # Approximation: each retransmission adds one RTT.
            tcp_expected_extra_latency_tu = tcp_expected_retries * seconds_to_tu(WIFI_RETRANSMISSION_RTT_S)
            tcp_expected_extra_latency_s = tu_to_seconds(tcp_expected_extra_latency_tu)

            udp_effective_bw_mb_s = wifi_bw_mb_s
            tcp_effective_bw_mb_s = wifi_bw_mb_s * (1.0 - tcp_retx_rate)

            tx_power_w, rx_power_w = galaxy_s4_wifi_powers_w_from_rssi(rssi_dbm)
            tail_energy_wh = tail_energy_wh_per_transfer()

            udp_energy_wh_per_mb = energy_wh_per_mb(
                throughput_mb_s=udp_effective_bw_mb_s,
                tx_power_w=tx_power_w,
                rx_power_w=rx_power_w,
            ) + tail_energy_wh
            tcp_energy_wh_per_mb = energy_wh_per_mb(
                throughput_mb_s=max(tcp_effective_bw_mb_s, 1e-12),
                tx_power_w=tx_power_w,
                rx_power_w=rx_power_w,
            ) + tail_energy_wh

            sensor_wifi_total_wh_per_mb, sensor_wifi_data_wh_per_mb, sensor_wifi_promotion_wh, sensor_wifi_tail_wh, sensor_wifi_tx_w, sensor_wifi_rx_w, sensor_wifi_available = sensor_wifi_energy_wh_per_mb(
                rssi_dbm=rssi_dbm,
                throughput_mb_s=wifi_bw_mb_s,
            )

            # Active sensor link depends on current execution mode.
            if execution_mode == "FOG" and sensor_wifi_available:
                sensor_active_link = "WIFI"
                sensor_active_energy_wh_per_mb = sensor_wifi_total_wh_per_mb
            else:
                sensor_active_link = "BLE"
                sensor_active_energy_wh_per_mb = sensor_ble_total_wh_per_mb

            # Dynamically update Mobile -> Fog link BW from RSSI model.
            if sim.topology.G.has_edge(2, fog.topo_id):
                sim.topology.G[2][fog.topo_id]["BW"] = wifi_bw_mb_s

            self.buffer.append(
                {
                    "sim_time_s": step,
                    "user_latitude": user_lat,
                    "user_longitude": user_lon,
                    "fog_dataset_id": fog.dataset_id,
                    "fog_topology_id": fog.topo_id,
                    "fog_latitude": fog.latitude,
                    "fog_longitude": fog.longitude,
                    "distance_m": distance_m,
                    "rssi_dbm": rssi_dbm,
                    "wifi_bw_mb_s": wifi_bw_mb_s,
                    "udp_effective_bw_mb_s": udp_effective_bw_mb_s,
                    "tcp_retransmission_rate": tcp_retx_rate,
                    "tcp_expected_retries": tcp_expected_retries,
                    "tcp_expected_extra_latency_tu": tcp_expected_extra_latency_tu,
                    "tcp_expected_extra_latency_s": tcp_expected_extra_latency_s,
                    "tcp_effective_bw_mb_s": tcp_effective_bw_mb_s,
                    "wifi_tx_power_w": tx_power_w,
                    "wifi_rx_power_w": rx_power_w,
                    "wifi_tail_energy_wh_per_transfer": tail_energy_wh,
                    "udp_energy_wh_per_mb": udp_energy_wh_per_mb,
                    "tcp_energy_wh_per_mb": tcp_energy_wh_per_mb,
                    "execution_mode": execution_mode,
                    "sensor_active_link": sensor_active_link,
                    "sensor_active_energy_wh_per_mb": sensor_active_energy_wh_per_mb,
                    "sensor_wifi_available": int(sensor_wifi_available),
                    "sensor_wifi_tx_power_w": sensor_wifi_tx_w,
                    "sensor_wifi_rx_power_w": sensor_wifi_rx_w,
                    "sensor_wifi_energy_wh_per_mb": sensor_wifi_total_wh_per_mb,
                    "sensor_wifi_data_energy_wh_per_mb": sensor_wifi_data_wh_per_mb,
                    "sensor_wifi_promotion_energy_wh_per_transfer": sensor_wifi_promotion_wh,
                    "sensor_wifi_tail_energy_wh_per_transfer": sensor_wifi_tail_wh,
                    "sensor_ble_energy_wh_per_mb": sensor_ble_total_wh_per_mb,
                    "sensor_ble_data_energy_wh_per_mb": sensor_ble_data_wh_per_mb,
                    "sensor_ble_tail_energy_wh_per_transfer": sensor_ble_tail_wh,
                    "current_fog_topology_id": current_fog_topology_id,
                    "current_mobile_topology_id": current_mobile_topology_id,
                    "task_processing_topology_id": task_processing_topology_id,
                    "is_nearest": int(fog.topo_id == nearest_topo_id),
                }
            )

        if (step % self.flush_every_steps) == 0:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        df = pd.DataFrame(self.buffer)
        mode = "a" if self.header_written else "w"
        df.to_csv(self.output_path, mode=mode, header=(not self.header_written), index=False)
        self.header_written = True
        self.buffer.clear()


class OffloadingDecisionMonitor:
    """
    Solves LP offloading decision periodically and updates CURRENT_EXECUTION_MODE.
    Logs decision metrics and post-factum migration energy.
    """
    def __init__(self, mobility_model: MobilityModel, output_path: Path):
        self.mobility_model = mobility_model
        self.output_path = output_path
        self.buffer = []
        self.header_written = False
        self.last_mode = str(CURRENT_EXECUTION_MODE).upper()

    def run(self, sim):
        global CURRENT_EXECUTION_MODE

        step = int(sim.env.now)
        distances = self.mobility_model.distances_for_step(step)
        if not distances:
            return

        nearest_fog, nearest_distance_m = min(distances, key=lambda item: item[1])
        fog_des = sim.alloc_module.get(APP_NAME, {}).get("Fog", [])
        current_fog_topology_id = sim.alloc_DES.get(fog_des[0], None) if fog_des else None
        mobile_des = sim.alloc_module.get(APP_NAME, {}).get("Mobile", [])
        current_mobile_topology_id = sim.alloc_DES.get(mobile_des[0], None) if mobile_des else None
        rssi_fog_dbm = rssi_from_distance_dbm(nearest_distance_m)
        wifi_bw_mb_s = wifi_throughput_mbps_from_rssi(rssi_fog_dbm)
        tcp_retx_rate = tcp_retransmission_rate_from_rssi(rssi_fog_dbm)
        attempts_factor = 1.0 / max(1.0 - tcp_retx_rate, 1e-12)

        task_input_mb = kb_to_mb(TASK1_DATA_SIZE_KB)
        task_result_mb = 740.0 / (1024.0 * 1024.0)
        migration_mb = kb_to_mb(TASK1_MIGRATION_STATE_SIZE_KB)

        # Link delays (TCP expected).
        ble_bw_mb_s = ips_to_ipt(SENSOR_BLE_DATA_BW_MBPS)
        delay_sensor_edge = task_input_mb / max(ble_bw_mb_s, 1e-12)
        delay_edge_fog = task_result_mb / max(wifi_bw_mb_s, 1e-12) * attempts_factor
        delay_sensor_fog = task_input_mb / max(wifi_bw_mb_s, 1e-12) * attempts_factor

        mobile_ipt = ips_to_ipt(1.9 * 10**9)
        fog_ipt = ips_to_ipt(500 * 10**6)
        proc_instructions = mi_to_instructions(TASK1_COMPLEXITY_MI)
        proc_delay_edge = proc_instructions / max(mobile_ipt, 1e-12)
        proc_delay_fog = proc_instructions / max(fog_ipt, 1e-12)

        # L objective term.
        latency_edge = delay_sensor_edge + proc_delay_edge + delay_edge_fog
        latency_fog = delay_sensor_fog + proc_delay_fog

        # E objective term.
        # EDGE mode:
        # sensor tx(BLE) + edge rx(BLE) + edge processing + edge tx->fog(WiFi).
        sensor_ble_tx_wh = watt_to_wpt(SENSOR_BLE_TX_POWER_W) * (task_input_mb / max(ble_bw_mb_s, 1e-12))
        edge_ble_rx_wh = watt_to_wpt(SENSOR_BLE_RX_POWER_W) * (task_input_mb / max(ble_bw_mb_s, 1e-12))
        edge_proc_wh = watt_to_wpt(1.3) * proc_delay_edge
        edge_wifi_tx_w, edge_wifi_rx_w = galaxy_s4_wifi_powers_w_from_rssi(rssi_fog_dbm)
        edge_tx_to_fog_wh = watt_to_wpt(edge_wifi_tx_w) * (task_result_mb / max(wifi_bw_mb_s, 1e-12) * attempts_factor)
        edge_tx_to_fog_wh += tail_energy_wh_per_transfer()
        energy_edge = sensor_ble_tx_wh + edge_ble_rx_wh + edge_proc_wh + edge_tx_to_fog_wh

        # FOG mode:
        # sensor tx->fog (WiFi) + edge rx from fog (result, WiFi). Fog energy excluded.
        sensor_wifi_tx_w, _, sensor_wifi_available = sensor_wifi_data_powers_w_from_rssi(rssi_fog_dbm)
        if sensor_wifi_available:
            sensor_tx_to_fog_wh = watt_to_wpt(sensor_wifi_tx_w) * (task_input_mb / max(wifi_bw_mb_s, 1e-12) * attempts_factor)
            sensor_tx_to_fog_wh += fixed_overhead_energy_wh(SENSOR_WIFI_PROMOTION_POWER_W, SENSOR_WIFI_PROMOTION_TIME_S)
            sensor_tx_to_fog_wh += fixed_overhead_energy_wh(SENSOR_WIFI_TAIL_POWER_W, SENSOR_WIFI_TAIL_TIME_S)
        else:
            # Infeasible link for sensor WiFi at weak RSSI.
            sensor_tx_to_fog_wh = 1e9

        edge_rx_from_fog_wh = watt_to_wpt(edge_wifi_rx_w) * (task_result_mb / max(wifi_bw_mb_s, 1e-12) * attempts_factor)
        edge_rx_from_fog_wh += tail_energy_wh_per_transfer()
        energy_fog = sensor_tx_to_fog_wh + edge_rx_from_fog_wh

        c_edge = ALPHA_LATENCY * latency_edge + BETA_ENERGY * energy_edge
        c_fog = ALPHA_LATENCY * latency_fog + BETA_ENERGY * energy_fog
        chosen_mode = solve_lp_two_mode(c_edge, c_fog)

        migration_energy_wh = 0.0
        if chosen_mode != self.last_mode:
            if chosen_mode == "FOG":
                # EDGE -> FOG: state goes from edge to fog, charge edge TX energy.
                migration_energy_wh = watt_to_wpt(edge_wifi_tx_w) * (migration_mb / max(wifi_bw_mb_s, 1e-12) * attempts_factor)
                migration_energy_wh += tail_energy_wh_per_transfer()
            else:
                # FOG -> EDGE: state returns from fog to edge, charge edge RX energy.
                migration_energy_wh = watt_to_wpt(edge_wifi_rx_w) * (migration_mb / max(wifi_bw_mb_s, 1e-12) * attempts_factor)
                migration_energy_wh += tail_energy_wh_per_transfer()

        CURRENT_EXECUTION_MODE = chosen_mode
        self.last_mode = chosen_mode

        self.buffer.append(
            {
                "sim_time_s": step,
                "nearest_fog_topology_id": nearest_fog.topo_id,
                "current_fog_topology_id": current_fog_topology_id,
                "current_mobile_topology_id": current_mobile_topology_id,
                "nearest_fog_distance_m": nearest_distance_m,
                "nearest_fog_rssi_dbm": rssi_fog_dbm,
                "wifi_bw_mb_s": wifi_bw_mb_s,
                "tcp_retransmission_rate": tcp_retx_rate,
                "latency_edge_tu": latency_edge,
                "latency_fog_tu": latency_fog,
                "energy_edge_wh": energy_edge,
                "energy_fog_wh": energy_fog,
                "objective_edge": c_edge,
                "objective_fog": c_fog,
                "chosen_mode": chosen_mode,
                "migration_energy_wh": migration_energy_wh,
            }
        )

        self.flush()

    def flush(self):
        if not self.buffer:
            return
        df = pd.DataFrame(self.buffer)
        mode = "a" if self.header_written else "w"
        df.to_csv(self.output_path, mode=mode, header=(not self.header_written), index=False)
        self.header_written = True
        self.buffer.clear()


def load_city_fog_devices(dataset_dir: Path, topo_start_id: int = 1000) -> List[CityFogDevice]:
    df = pd.read_csv(dataset_dir / "edgeResources-melbCBD.csv")
    # Keep city resources (exclude central datacenter level).
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


def load_user_trace(dataset_dir: Path) -> List[Tuple[float, float]]:
    df = pd.read_csv(dataset_dir / "usersLocation-melbCBD_1.csv")
    waypoints = list(zip(df["Latitude"].astype(float), df["Longitude"].astype(float)))
    return interpolate_user_path(waypoints, speed_mps=USER_SPEED_MPS, step_seconds=SIM_STEP_SECONDS)
        
def watt_to_wpt(watt: float) -> float:
    if TIME_UNIT == TimeUnit.SECOND:
        return watt / 3600.0
    if TIME_UNIT == TimeUnit.MILLISECOND:
        return watt / (3600.0 * 1000.0)
    raise ValueError("Unknown TIME_UNIT")
        
def ips_to_ipt(ips: float) -> float:
    if TIME_UNIT == TimeUnit.SECOND:
        return ips
    if TIME_UNIT == TimeUnit.MILLISECOND:
        return ips * 10 ** -3
    raise ValueError("Unknown TIME_UNIT")

def create_topology(city_fog_devices: List[CityFogDevice]) -> Topology:
    """
    TOPOLOGY
    """
    topology_json = {}
    topology_json["entity"] = []
    topology_json["link"] = []

    # "COST" only used to calculate get_cost_cloud() which is currently commmented
    # IPT - Instructions Per Time Unit, where Time Unit is what is 1 in simulation
    # In out case Time Unit is 1 second so IPT = MIPS and WATT = Power/3600
    # For the case when Time Unit is 1 millisecond the IPT = MIPS * 10 ^ -3

    # Fog - PCEngines ALIX 3D2 (500MHzx86 CPU, 256MB of RAM):
    # - IPT: 500 * 10^6 instructions per second (500 MIPS) -> 500 * 10^3 IPT
    # - RAM: 256 MB
    # - CPU Power Consumption: 0.9W (AMD GeodeTM LX Processors Data Book)
    # - Link Power: Trans 4.9W / Recv 3.7W
    ##   Achilles and the Tortoise: Power Consumption in IEEE 802.11n and IEEE 802.11g Networks
    ##.  https://www.robertoriggio.net/papers/greencom2013.pdf
    #
    # Modile - Galaxy S4 (1.9GHz Quad-Core Krait 300 CPU, 2GB of RAM):
    # - IPT: 1.9 * 10^9 instructions per second (1900 MIPS) -> 1.9 * 10^6 IPT
    # - RAM: 2 GB
    # - CPU Power Consumption: 1.3W (1890Mhz, full utilization)
    # - WiFi Link Power: -50dBm Trans: 654mW / Recv: 451mW
    #                    -80dBm Trans: 1113mW / Recv: 633mW
    # - BLE (4.0) Link Power: Recv: 174mW
    ##   Smartphone Energy Drain in the Wild: Analysis and Implications
    ##.  https://engineering.purdue.edu/~ychu/publications/TR-ECE-15-03.pdf
    #
    # SmartWatch - LG Urbane watch (768GHz Quad-Core ARM Cortex-A7 CPU, 512MB of RAM):
    # - IPT: 768 * 10^6 instructions per second -> 768 * 10^3 IPT
    # - RAM: 512 MB
    # - CPU Power Consumption: 361mW
    # - WiFi Link Power: Trans: 739.9 / Recv: 400.1mW
    # - BLE (4.1) Link Power: Trans: 180.7mW / Recv: 174.9mW
    ##   Poster: Measuring and Optimizing Android Smartwatch Energy Consumption
    ##.  https://dl.acm.org/doi/10.1145/2973750.2985259
    # Static topology endpoints: cloud/mobile/ecg-sensor.
    # Dynamic fog offloading is done to city fog devices only.
    cloud_dev    = {"id": 0, "model": node_model_name("cloud-device"), "type": "CLOUD", "IPT": 5000 * 10**6, "RAM": 40000, "WATT": 0.0}
    mobile_dev = {"id": 2, "model": node_model_name("mobile-device"), "type": "EDGE",
                  "IPT": ips_to_ipt(1.9 * 10**9),
                  "RAM": 2000,
                  "WATT": watt_to_wpt(1.3)}
    ecg_dev = {"id": 4, "model": node_model_name("ecg-device"), "type": "IOT",
               "IPT": ips_to_ipt(768 * 10**6),
               "RAM": 256,
               "WATT": watt_to_wpt(0.361)}

    topology_json["entity"].append(cloud_dev)
    topology_json["entity"].append(mobile_dev)
    topology_json["entity"].append(ecg_dev)

    for fog in city_fog_devices:
        topology_json["entity"].append(
            {
                "id": fog.topo_id,
                "model": node_model_name("city-fog-device"),
                "type": "FOG",
                "IPT": ips_to_ipt(500 * 10**6),
                "RAM": 256,
                "WATT": watt_to_wpt(0.9),
                "LATITUDE": fog.latitude,
                "LONGITUDE": fog.longitude,
                "DETAILS": fog.details,
            }
        )
    
    # BLE 4.0/4.1 Modulation Rate: 1 Mb/s, Max Throughput: 0.305 Mb/s
    ##  Data Transmission Efficiency in Bluetooth Low Energy Versions
    ## https://www.mdpi.com/1424-8220/19/17/3746

    # 100 Mbit/s = 12,5 Mb/s
    # with open(f"{APP_NAME}/networkDefinition.json", "r") as f:
        # data = json.load(f)
        # for link in data['link']:
        #     link['WATT_TRANS'] = watt_to_wpt(link['WATT_TRANS'])
        #     link['WATT_RECV'] = watt_to_wpt(link['WATT_RECV'])
        #     topology_json["link"].append(link)

    # ECG sensor -> Mobile (BLE).
    topology_json["link"].append(
        {
            "PR": 0,
            "s": 4,
            "BW": ips_to_ipt(0.305),
            "d": 2,
            "WATT_TRANS": watt_to_wpt(0.1807),
            "WATT_RECV": watt_to_wpt(0.1749),
        }
    )

    # Mobile -> each city fog (wireless uplink).
    for fog in city_fog_devices:
        topology_json["link"].append(
            {
                "PR": 0,
                "s": 2,
                "BW": ips_to_ipt(12.5),
                "d": fog.topo_id,
                "WATT_TRANS": watt_to_wpt(0.654),
                "WATT_RECV": watt_to_wpt(3.7),
            }
        )

    # ECG sensor -> each city fog (WiFi uplink, used in FOG execution mode).
    for fog in city_fog_devices:
        topology_json["link"].append(
            {
                "PR": 0,
                "s": 4,
                "BW": ips_to_ipt(12.5),
                "d": fog.topo_id,
                "WATT_TRANS": watt_to_wpt(0.6728),
                "WATT_RECV": watt_to_wpt(3.7),
            }
        )

    # Every city fog device -> cloud.
    for fog in city_fog_devices:
        topology_json["link"].append(
            {
                "PR": 0,
                "s": fog.topo_id,
                "BW": ips_to_ipt(12.5),
                "d": 0,
                "WATT_TRANS": watt_to_wpt(4.9),
                "WATT_RECV": watt_to_wpt(3.7),
            }
        )

    t = Topology()
    t.load(topology_json)
    validate_topology_constraints(t)

    return t


def validate_topology_constraints(topology: Topology) -> None:
    """
    Enforce architectural constraint:
    Edge must not communicate with Cloud over a direct link.
    Cloud connectivity is allowed only via Fog nodes (wired Fog-Cloud).
    """
    mobile_node_id = 2
    cloud_node_id = 0
    if topology.G.has_edge(mobile_node_id, cloud_node_id):
        raise ValueError(
            "Invalid topology: direct Mobile(2)-Cloud(0) link is forbidden. "
            "Edge-to-Cloud traffic must go through Fog."
        )


def create_application():
    # APPLICATION
    a = Application(name=APP_NAME)

    a.set_modules([
        {"Cloud": {"Type": Application.TYPE_SINK}},
        {"Fog": {"RAM": 1024, "Type": Application.TYPE_MODULE}},
        {"Mobile": {"RAM": 1024, "Type": Application.TYPE_MODULE}},
        {"EcgSensor": {"Type": Application.TYPE_MODULE}},
        {"Virtual-ECG-Gen": {"Type": Application.TYPE_SOURCE}},
    ])

    with open(Path(__file__).parent / APP_NAME / "appDefinition.json", "r") as f:
        data = json.load(f)

    messages = {}
    for message in data["message"]:
        m = Message(
            message["name"],
            message["s"],
            message["d"],
            instructions=message["instructions"],
            bytes=message["bytes"],
        )
        messages[message["name"]] = m
        if message["s"] == "None":
            a.add_source_messages(m)

    for tx in data["transmission"]:
        tx_mode = str(tx.get("mode", "BOTH")).upper()
        if "message_out" in tx:
            threshold = tx.get("fractional", 1.0)
            if tx_mode in ("EDGE", "FOG"):
                a.add_service_module(
                    tx["module"],
                    messages[tx["message_in"]],
                    messages[tx["message_out"]],
                    task_mode_selectivity,
                    threshold=threshold,
                    mode=tx_mode,
                )
            else:
                a.add_service_module(
                    tx["module"],
                    messages[tx["message_in"]],
                    messages[tx["message_out"]],
                    fractional_selectivity,
                    threshold=threshold,
                )
        else:
            a.add_service_module(tx["module"], messages[tx["message_in"]])

    return a


def generate_mobility_animation(
    mobility_model: MobilityModel,
    output_path: Path,
    placement_history_path: Optional[Path] = None,
    simulated_until_step: Optional[int] = None,
    step_stride: int = ANIMATION_STEP_STRIDE,
):
    """
    Creates a lightweight animation for user movement, nearest fog selection,
    and real task processing placement (EDGE/FOG) if history is provided.
    """
    if not mobility_model.user_trace or not mobility_model.fog_devices:
        return

    effective_stride = max(1, step_stride)
    n_steps = len(mobility_model.user_trace)
    fog_lats = np.array([f.latitude for f in mobility_model.fog_devices])
    fog_lons = np.array([f.longitude for f in mobility_model.fog_devices])
    fog_by_topo_id = {f.topo_id: f for f in mobility_model.fog_devices}

    placement_by_step: Dict[int, Dict[str, object]] = {}
    if placement_history_path is not None and placement_history_path.exists():
        df = pd.read_csv(placement_history_path)
        for step, g in df.groupby("sim_time_s", as_index=False):
            row = g.iloc[0]
            placement_by_step[int(step)] = {
                "execution_mode": str(row.get("execution_mode", "EDGE")).upper(),
                "task_processing_topology_id": row.get("task_processing_topology_id", np.nan),
            }
        if simulated_until_step is None and not df.empty:
            simulated_until_step = int(df["sim_time_s"].max())

    if simulated_until_step is not None:
        n_steps = min(n_steps, max(1, int(simulated_until_step) + 1))

    est_frames = n_steps // effective_stride
    if est_frames > ANIMATION_MAX_FRAMES:
        effective_stride = int(np.ceil(float(n_steps) / float(ANIMATION_MAX_FRAMES)))
    frames = list(range(0, n_steps, effective_stride))

    fig, ax = plt.subplots(figsize=(8, 8))
    method_tag = OPTIMIZATION_METHOD.upper()
    ax.scatter(fog_lons, fog_lats, s=10, c="lightgray", label=f"City fog nodes ({method_tag})")
    path_lons = np.array([p[1] for p in mobility_model.user_trace])
    path_lats = np.array([p[0] for p in mobility_model.user_trace])
    ax.plot(path_lons, path_lats, linewidth=1.0, color="#2a9d8f", alpha=0.4, label=f"User path ({method_tag})")

    user_point, = ax.plot([], [], "o", color="#e76f51", markersize=8, label=f"User ({method_tag})")
    nearest_point, = ax.plot([], [], "o", color="#264653", markersize=8, label="Nearest fog (distance)")
    active_point, = ax.plot([], [], marker="*", color="#1d3557", markersize=11, linestyle="None", label=f"Active processing node ({method_tag})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper right")

    def _update(step: int):
        lat, lon = mobility_model.user_position(step)
        nearest_topo = mobility_model.nearest_fog_topology_node(step)
        nearest = next((f for f in mobility_model.fog_devices if f.topo_id == nearest_topo), None)
        user_point.set_data([lon], [lat])
        if nearest is not None:
            nearest_point.set_data([nearest.longitude], [nearest.latitude])

        mode = str(CURRENT_EXECUTION_MODE).upper()
        placement = placement_by_step.get(step)
        if placement is not None:
            mode = str(placement.get("execution_mode", mode)).upper()
            topo_id = placement.get("task_processing_topology_id", np.nan)
            if mode == "FOG" and pd.notna(topo_id):
                fog = fog_by_topo_id.get(int(topo_id))
                if fog is not None:
                    active_point.set_data([fog.longitude], [fog.latitude])
                else:
                    active_point.set_data([], [])
            else:
                # EDGE processing is on the mobile device at user's location.
                active_point.set_data([lon], [lat])
        else:
            if mode == "FOG":
                active_point.set_data([], [])
            else:
                active_point.set_data([lon], [lat])

        ax.set_title(f"{APP_NAME} | {method_tag} | t={step}s | mode={mode}")
        return user_point, nearest_point, active_point

    ani = FuncAnimation(fig, _update, frames=frames, interval=80, blit=False, repeat=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if ANIMATION_FORMAT.lower() == "gif":
            ani.save(output_path.with_suffix(".gif"), writer=PillowWriter(fps=8))
        else:
            ani.save(output_path.with_suffix(".mp4"), writer=FFMpegWriter(fps=12))
    except Exception as ex:
        logging.warning("Animation export failed (%s). Falling back to final snapshot.", ex)
        lat, lon = mobility_model.user_position(n_steps - 1)
        ax.plot([lon], [lat], "o", color="#e76f51", markersize=8)
        ax.set_title("Final user position")
        fig.savefig(output_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)


def main(stop_time, it,folder_results):
    global CURRENT_EXECUTION_MODE
    CURRENT_EXECUTION_MODE = str(TASK_EXECUTION_MODE).upper()
    sim_tag = build_simulation_tag()
    trace_basename = f"sim_trace_{sim_tag}"

    dataset_dir = Path(__file__).parent / f"{APP_NAME}/dataset"
    city_fog_devices = load_city_fog_devices(dataset_dir)
    user_trace = load_user_trace(dataset_dir)
    mobility_model = MobilityModel(user_trace=user_trace, fog_devices=city_fog_devices)

    """
    TOPOLOGY
    """
    t = create_topology(city_fog_devices)

    print(t.G.nodes()) # nodes id can be str or int

    # Plotting the graph
    pos=nx.spring_layout(t.G)
    nx.draw_networkx(t.G, pos, with_labels=True)
    nx.draw_networkx_edge_labels(t.G, pos,alpha=0.5,font_size=5,verticalalignment="top")


    """
    APPLICATION or SERVICES
    """
    # dataApp = json.load(open('data/appDefinition.json'))
    # apps = create_applications_from_json(dataApp)
    app = create_application()

    """
    SERVICE PLACEMENT 
    """
    initial_fog_node = mobility_model.nearest_fog_topology_node(step=0)
    placementJson = {
        "initialAllocation": [
            {"app": APP_NAME, "module_name": "Fog", "id_resource": initial_fog_node if initial_fog_node is not None else 0},
            {"app": APP_NAME, "module_name": "Mobile", "id_resource": 2},
            {"app": APP_NAME, "module_name": "EcgSensor", "id_resource": 4}
        ]
    }
    placement_dist = deterministic_distribution(name="MobilityPlacementTick", time=SIM_STEP_SECONDS)
    placement = MobilityPlacement(
        name=f"MobilityPlacement_{OPTIMIZATION_METHOD}",
        json=placementJson,
        activation_dist=placement_dist,
        mobility_model=mobility_model,
        app_name=APP_NAME,
    )

    
    """
    POPULATION algorithm
    """
    pop = Statical("Statical")
    #For each type of sink modules we set a deployment on some type of devices
    #A control sink consists on:
    #  args:
    #     model (str): identifies the device or devices where the sink is linked
    #     number (int): quantity of sinks linked in each device
    #     module (str): identifies the module from the app who receives the messages
    pop.set_sink_control({"model": node_model_name("cloud-device"), "number":1, "module":app.get_sink_modules()})

    #In addition, a source includes a distribution function:
    dDistribution1 = deterministic_distribution(name="Deterministic", time=TASK1_PERIOD_S)
    pop.set_src_control({"model": node_model_name("ecg-device"), "number":1, "message": app.get_message("M.TASK1.TCP.Generation"), "distribution": dDistribution1})

    # populationJSON = {
    #     "sinks": [
    #         {"app": APP_NAME, "module_name": "Cloud", "id_resource": 0},
    #     ],
    #     "sources":[
    #         {"app": APP_NAME, "message":"M.SW-M", "time":100,"id_resource":3},
    #     ]

    # }
    # pop = JSONPopulation(name="Statical",json=populationJSON,iteration=0)

    
    """
    Defining ROUTING algorithm to define how path messages in the topology among modules
    """
    selectorPath = First_ShortestPath()

    
    """
    SIMULATION ENGINE
    """
    s = Sim(t, default_results_path=folder_results + trace_basename)

    
    """
    Deploy services == APP's modules
    """
    s.deploy_app2(app, placement, pop, selectorPath)

    distance_output = Path(folder_results) / f"user_fog_distances_{sim_tag}.csv"
    distance_monitor = MobilityDistanceMonitor(mobility_model=mobility_model, output_path=distance_output)
    monitor_dist = deterministicDistributionStartPoint(
        name="MobilityDistanceTick",
        start=0,
        time=SIM_STEP_SECONDS,
    )
    s.deploy_monitor(
        f"MobilityDistanceMonitor_{OPTIMIZATION_METHOD}",
        distance_monitor.run,
        monitor_dist,
        sim=s,
    )

    decision_output = Path(folder_results) / f"offloading_decisions_{sim_tag}.csv"
    offloading_monitor = OffloadingDecisionMonitor(
        mobility_model=mobility_model,
        output_path=decision_output,
    )
    decision_dist = deterministicDistributionStartPoint(
        name="OffloadingDecisionTick",
        start=0,
        time=OFFLOADING_DECISION_PERIOD_S,
    )
    s.deploy_monitor(
        f"OffloadingDecisionMonitor_{OPTIMIZATION_METHOD}",
        offloading_monitor.run,
        decision_dist,
        sim=s,
    )


    """
    RUNNING
    """
    logging.info(" Performing simulation: %i " % it)
    s.run(stop_time)  # To test deployments put test_initial_deploy a TRUE
    distance_monitor.flush()
    offloading_monitor.flush()
    s.print_debug_assignaments()

    if ENABLE_ANIMATION:
        generate_mobility_animation(
            mobility_model=mobility_model,
            output_path=Path(folder_results) / f"mobility_placement_{sim_tag}.gif",
            placement_history_path=distance_output,
            simulated_until_step=max(0, int(stop_time) - 1),
            step_stride=ANIMATION_STEP_STRIDE,
        )

    s1 = Stats(defaultPath=os.path.join(os.getcwd(), folder_results, trace_basename))

    # Consumption and number of bytes report
    s1.showResults(total_time=stop_time, topology=t, multiplier=1)

    # Latency report
    print("\nLatency Report (in time unit):")
    latency = pd.concat([s1.times("time_latency"), s1.times("time_service"), s1.times("time_total_response")], axis=1)
    latency.loc["Total"] = latency.sum()
    print(latency)


if __name__ == '__main__':
    LOGGING_CONFIG = Path(__file__).parent / 'logging.ini'
    logging.config.fileConfig(LOGGING_CONFIG)

    folder_results = Path("results/")
    folder_results.mkdir(parents=True, exist_ok=True)
    folder_results = str(folder_results)+"/"

    nIterations = 1  # iteration for each experiment
    simulationDuration = 3600
    if APP_NAME == "CityScenario":
        dataset_dir = Path(__file__).parent / f"{APP_NAME}/dataset"
        user_trace = load_user_trace(dataset_dir)
        simulationDuration = len(user_trace)

    # Iteration for each experiment changing the seed of randoms
    for iteration in range(nIterations):
        random.seed(iteration)
        logging.info("Running experiment it: - %i" % iteration)

        start_time = time.time()
        main(stop_time=simulationDuration,
             it=iteration,folder_results=folder_results)

        print("\n--- %s seconds ---" % (time.time() - start_time))

    print("Simulation Done!")
    sim_tag = build_simulation_tag()
    trace_basename = f"sim_trace_{sim_tag}"
  
    # Analysing the results. 
    dfl = pd.read_csv(folder_results + trace_basename + "_link.csv")
    print("Number of total messages between nodes: %i"%len(dfl))

    df = pd.read_csv(folder_results + trace_basename + ".csv")
    print("Number of requests handled by deployed services: %i"%len(df))
