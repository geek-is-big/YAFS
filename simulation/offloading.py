"""
    Offloading simulation
"""
import os
import time
import json
import random
import argparse
import logging.config
from typing import List, Tuple, Optional

import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt

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

import config as cfg
from mobility import (
    CityFogDevice,
    MobilityModel,
    load_city_fog_devices,
    load_user_trace,
)
from placement_edge_fog import LPOptimizationPlacement
from utils import (
    generate_energy_decision_plot,
    generate_mobility_animation,
    ips_to_ipt,
    seconds_to_tu,
    watt_to_wpt,
)

# from jsonAllocation import JSONPopulation

DEFAULT_APP = "MobileScenario"
DYNAMIC_SCENARIOS = {"CityScenario"}
STATIC_SCENARIOS = {"FogScenario", "HybridScenario", "MobileScenario"}
ALL_SCENARIOS = {*DYNAMIC_SCENARIOS, *STATIC_SCENARIOS}

APP_NAME = DEFAULT_APP
SIMULATION_MODE = "dynamic"  # dynamic | static

# Task1 profile (ECG classification)
CURRENT_EXECUTION_MODE = cfg.TASK_EXECUTION_MODE


def get_execution_mode() -> str:
    return str(CURRENT_EXECUTION_MODE).upper()


def set_execution_mode(mode: str) -> None:
    global CURRENT_EXECUTION_MODE
    CURRENT_EXECUTION_MODE = str(mode).upper()


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


def infer_message_transport(message_name: str) -> str:
    tokens = str(message_name).upper().replace("-", ".").replace("_", ".").split(".")
    if "TCP" in tokens:
        return "TCP"
    if "UDP" in tokens:
        return "UDP"
    return "UDP"


def build_simulation_tag() -> str:
    if SIMULATION_MODE == "static":
        return f"{APP_NAME}_static"
    alpha_str = str(cfg.ALPHA_LATENCY).replace(".", "p")
    beta_str = str(cfg.BETA_ENERGY).replace(".", "p")
    return f"{APP_NAME}_{cfg.OPTIMIZATION_METHOD}_a{alpha_str}_b{beta_str}"


def rssi_from_distance_dbm(
    distance_m: float,
    d0_m: float = cfg.RSSI_REFERENCE_DISTANCE_M,
    rssi0_dbm: float = cfg.RSSI_AT_REFERENCE_DBM,
    n: float = cfg.RSSI_ENVIRONMENT_COEFF,
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
        return ips_to_ipt(cfg.WIFI_MAX_BW_MBPS)
    if rssi_dbm >= -85.0:
        return ips_to_ipt(cfg.WIFI_MEDIUM_BW_MBPS)
    if rssi_dbm <= -90.0:
        return ips_to_ipt(cfg.WIFI_MIN_BW_MBPS)

    # Linear interpolation on (-90, -85) for intermediate weak-signal values.
    # rssi=-90 -> 1 Mbps, rssi=-85 -> 11 Mbps
    x = np.array([-90.0, -85.0], dtype=float)
    y = np.array([cfg.WIFI_MIN_BW_MBPS, cfg.WIFI_MEDIUM_BW_MBPS], dtype=float)
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


def energy_wh_per_mb(throughput_mb_s: float, tx_power_w: float, rx_power_w: float) -> float:
    """
    One-way transfer energy (TX+RX) per transferred MB.
    """
    tx_wh_per_mb, rx_wh_per_mb = transfer_energy_components_wh_per_mb(
        throughput_mb_s=throughput_mb_s,
        tx_power_w=tx_power_w,
        rx_power_w=rx_power_w,
    )
    return tx_wh_per_mb + rx_wh_per_mb


def transfer_energy_components_wh_per_mb(
    throughput_mb_s: float, tx_power_w: float, rx_power_w: float
) -> Tuple[float, float]:
    """
    One-way transfer energy components per transferred MB: (TX, RX).
    """
    safe_bw = max(throughput_mb_s, 1e-12)
    transfer_time_tu = 1.0 / safe_bw
    tx_wh_per_mb = watt_to_wpt(tx_power_w) * transfer_time_tu
    rx_wh_per_mb = watt_to_wpt(rx_power_w) * transfer_time_tu
    return tx_wh_per_mb, rx_wh_per_mb


def fixed_overhead_energy_wh(power_w: float, duration_s: float) -> float:
    return watt_to_wpt(power_w) * seconds_to_tu(duration_s)


def tail_energy_wh_per_transfer() -> float:
    """
    Fixed WiFi tail energy after data exchange completion.
    """
    return watt_to_wpt(cfg.WIFI_TAIL_POWER_W) * seconds_to_tu(cfg.WIFI_TAIL_TIME_S)


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
    # if rssi_dbm < -70.0:
    #     return np.nan, np.nan, False

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
    promotion_wh = fixed_overhead_energy_wh(cfg.SENSOR_WIFI_PROMOTION_POWER_W, cfg.SENSOR_WIFI_PROMOTION_TIME_S)
    tail_wh = fixed_overhead_energy_wh(cfg.SENSOR_WIFI_TAIL_POWER_W, cfg.SENSOR_WIFI_TAIL_TIME_S)
    total_wh_per_mb = data_wh_per_mb + promotion_wh + tail_wh
    return total_wh_per_mb, data_wh_per_mb, promotion_wh, tail_wh, tx_w, rx_w, True


def sensor_ble_energy_wh_per_mb() -> Tuple[float, float, float]:
    """
    Sensor BLE energy model (sensor-side TX data + tail).
    """
    # Characterizing Smartwatch Usage in the Wild - DOI: 10.1145/3081333.3081351
    # Sensors could be connected via BLE with Edge device, with typical distance around 0.5m
    # and Tx/Rx power around 110mW. BLE tail is long (4.77s) but low power (34mW).
    data_wh_per_mb = energy_wh_per_mb(
        throughput_mb_s=ips_to_ipt(cfg.SENSOR_BLE_DATA_BW_MBPS),
        tx_power_w=cfg.SENSOR_BLE_TX_POWER_W,
        rx_power_w=0.0,
    )
    tail_wh = fixed_overhead_energy_wh(cfg.SENSOR_BLE_TAIL_POWER_W, cfg.SENSOR_BLE_TAIL_TIME_S)
    total_wh_per_mb = data_wh_per_mb + tail_wh
    return total_wh_per_mb, data_wh_per_mb, tail_wh


def edge_key(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u <= v else (v, u)


class LinkUpdateRegistry:
    """
    Stores per-edge update callbacks. One callback is invoked per registered edge on each tick.
    """

    def __init__(self):
        self._callbacks = {}

    def register(self, u: int, v: int, callback) -> None:
        self._callbacks[edge_key(u, v)] = callback

    def update_all(self, sim, context: dict) -> None:
        for (u, v), callback in self._callbacks.items():
            if not sim.topology.G.has_edge(u, v):
                continue
            edge_data = sim.topology.G[u][v]
            callback(sim, u, v, edge_data, context)


def _topology_node_attr(sim, node_id: int, attr_name: str, default: str = "") -> str:
    # Topology.load() stores full node attrs in topology.nodeAttributes, while G.nodes
    # may contain only mandatory attrs (e.g. IPT). Keep both paths for compatibility.
    val = sim.topology.G.nodes[node_id].get(attr_name, None)
    if val is None:
        node_info = sim.topology.get_info().get(node_id)
        if node_info is None:
            node_info = sim.topology.get_info().get(str(node_id), {})
        val = node_info.get(attr_name, default)
    return str(val)


def _node_model(sim, node_id: int) -> str:
    return _topology_node_attr(sim, node_id, "model", "").lower()


def _node_type(sim, node_id: int) -> str:
    return _topology_node_attr(sim, node_id, "type", "").upper()


def _directional_watts_or_default(edge_data: dict, src: int, dst: int) -> Tuple[float, float]:
    tx = float(edge_data.get(f"WATT_TRANS_{src}-{dst}", edge_data.get("WATT_TRANS", 0.0)))
    rx = float(edge_data.get(f"WATT_RECV_{src}-{dst}", edge_data.get("WATT_RECV", tx)))
    return tx, rx


def _update_mobile_fog_link(sim, u: int, v: int, edge_data: dict, context: dict) -> None:
    mobile_id = u if _node_model(sim, u) == "mobile-device" else v
    fog_id = v if mobile_id == u else u
    state = context["wireless_by_fog"].get(fog_id)
    if state is None:
        return

    edge_data["BW"] = state["wifi_bw_mb_s"]
    edge_data["RTR"] = state["tcp_retx_rate"]

    fog_tx_wpt, _ = _directional_watts_or_default(edge_data, fog_id, mobile_id)
    _, fog_rx_wpt = _directional_watts_or_default(edge_data, mobile_id, fog_id)
    edge_data[f"WATT_TRANS_{mobile_id}-{fog_id}"] = watt_to_wpt(state["mobile_tx_w"])
    edge_data[f"WATT_RECV_{mobile_id}-{fog_id}"] = fog_rx_wpt
    edge_data[f"WATT_TRANS_{fog_id}-{mobile_id}"] = fog_tx_wpt
    edge_data[f"WATT_RECV_{fog_id}-{mobile_id}"] = watt_to_wpt(state["mobile_rx_w"])


def _update_sensor_fog_link(sim, u: int, v: int, edge_data: dict, context: dict) -> None:
    sensor_models = {"ecg-device", "smartwatch-device"}
    sensor_id = u if _node_model(sim, u) in sensor_models else v
    fog_id = v if sensor_id == u else u
    state = context["wireless_by_fog"].get(fog_id)
    if state is None:
        return

    if not state["sensor_wifi_available"]:
        # 100 Kbps in MBps, to avoid zero division and allow some minimal connectivity for energy calculations.
        edge_data["BW"] = 0.1 / 8
    else:
        edge_data["BW"] = state["wifi_bw_mb_s"]
    edge_data["RTR"] = state["tcp_retx_rate"]

    fog_tx_wpt, _ = _directional_watts_or_default(edge_data, fog_id, sensor_id)
    _, fog_rx_wpt = _directional_watts_or_default(edge_data, sensor_id, fog_id)
    edge_data[f"WATT_TRANS_{sensor_id}-{fog_id}"] = watt_to_wpt(state["sensor_tx_w"])
    edge_data[f"WATT_RECV_{sensor_id}-{fog_id}"] = fog_rx_wpt
    edge_data[f"WATT_TRANS_{fog_id}-{sensor_id}"] = fog_tx_wpt
    edge_data[f"WATT_RECV_{fog_id}-{sensor_id}"] = watt_to_wpt(state["sensor_rx_w"])


def _noop_link_update(sim, u: int, v: int, edge_data: dict, context: dict) -> None:
    return


def assign_tail_fields_by_models(link: dict, model_by_id: dict) -> None:
    s = int(link["s"])
    d = int(link["d"])
    model_s = str(model_by_id.get(s, "")).lower()
    model_d = str(model_by_id.get(d, "")).lower()
    models = {model_s, model_d}

    wifi_tail_wh = fixed_overhead_energy_wh(cfg.WIFI_TAIL_POWER_W, cfg.WIFI_TAIL_TIME_S)
    sensor_wifi_tail_wh = fixed_overhead_energy_wh(cfg.SENSOR_WIFI_TAIL_POWER_W, cfg.SENSOR_WIFI_TAIL_TIME_S)
    ble_tail_wh = fixed_overhead_energy_wh(cfg.SENSOR_BLE_TAIL_POWER_W, cfg.SENSOR_BLE_TAIL_TIME_S)

    if {"mobile-device", "city-fog-device"} == models or {"mobile-device", "fog-device"} == models:
        mobile_id = s if model_s == "mobile-device" else d
        fog_id = d if mobile_id == s else s
        link[f"TAIL_TRANS_{mobile_id}-{fog_id}"] = wifi_tail_wh
        link[f"TAIL_RECV_{mobile_id}-{fog_id}"] = 0.0
        link[f"TAIL_TRANS_{fog_id}-{mobile_id}"] = 0.0
        link[f"TAIL_RECV_{fog_id}-{mobile_id}"] = wifi_tail_wh
        return

    if {"ecg-device", "city-fog-device"} == models or {"ecg-device", "fog-device"} == models:
        sensor_id = s if model_s == "ecg-device" else d
        fog_id = d if sensor_id == s else s
        link[f"TAIL_TRANS_{sensor_id}-{fog_id}"] = sensor_wifi_tail_wh
        link[f"TAIL_RECV_{sensor_id}-{fog_id}"] = 0.0
        link[f"TAIL_TRANS_{fog_id}-{sensor_id}"] = 0.0
        link[f"TAIL_RECV_{fog_id}-{sensor_id}"] = 0.0
        return

    if {"smartwatch-device", "city-fog-device"} == models or {"smartwatch-device", "fog-device"} == models:
        sw_id = s if model_s == "smartwatch-device" else d
        fog_id = d if sw_id == s else s
        link[f"TAIL_TRANS_{sw_id}-{fog_id}"] = sensor_wifi_tail_wh
        link[f"TAIL_RECV_{sw_id}-{fog_id}"] = 0.0
        link[f"TAIL_TRANS_{fog_id}-{sw_id}"] = 0.0
        link[f"TAIL_RECV_{fog_id}-{sw_id}"] = 0.0
        return

    if (model_s == "mobile-device" and model_d in {"ecg-device", "smartwatch-device"}) or (
        model_d == "mobile-device" and model_s in {"ecg-device", "smartwatch-device"}
    ):
        mobile_id = s if model_s == "mobile-device" else d
        iot_id = d if mobile_id == s else s
        link[f"TAIL_TRANS_{iot_id}-{mobile_id}"] = ble_tail_wh
        link[f"TAIL_RECV_{iot_id}-{mobile_id}"] = ble_tail_wh
        link[f"TAIL_TRANS_{mobile_id}-{iot_id}"] = ble_tail_wh
        link[f"TAIL_RECV_{mobile_id}-{iot_id}"] = ble_tail_wh


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
        self.link_registry = LinkUpdateRegistry()
        self.link_registry_initialized = False

    def _init_link_registry(self, sim) -> None:
        if self.link_registry_initialized:
            return
        for u, v in sim.topology.get_edges():
            models = {_node_model(sim, u), _node_model(sim, v)}
            types = {_node_type(sim, u), _node_type(sim, v)}
            if "cloud-device" in models or "CLOUD" in types:
                self.link_registry.register(u, v, _noop_link_update)
            elif {"mobile-device", "city-fog-device"} == models:
                self.link_registry.register(u, v, _update_mobile_fog_link)
            elif {"ecg-device", "city-fog-device"} == models:
                self.link_registry.register(u, v, _update_sensor_fog_link)
            elif {"smartwatch-device", "city-fog-device"} == models:
                self.link_registry.register(u, v, _update_sensor_fog_link)
            elif {"mobile-device", "ecg-device"} == models:
                self.link_registry.register(u, v, _noop_link_update)
            elif {"mobile-device", "smartwatch-device"} == models:
                self.link_registry.register(u, v, _noop_link_update)
            else:
                self.link_registry.register(u, v, _noop_link_update)
        self.link_registry_initialized = True

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
        self._init_link_registry(sim)
        step = int(sim.env.now)
        user_lat, user_lon = self.mobility_model.user_position(step)
        distances = self.mobility_model.distances_for_step(step)
        if not distances:
            return

        distances_sorted = sorted(distances, key=lambda item: item[1])
        nearest_topo_id = distances_sorted[0][0].topo_id

        if cfg.PRINT_NEAREST_DISTANCES:
            nearest_n = distances_sorted[:max(1, cfg.NEAREST_NODES_TO_PRINT)]
            nearest_info = ", ".join(
                f"{fog.topo_id}:{dist_m:.2f}m" for fog, dist_m in nearest_n
            )
            print(f"[t={step:4d}s] nearest_fog_nodes -> {nearest_info}")

        execution_mode = self._current_execution_mode(sim)
        current_fog_topology_id = self._current_fog_topology_id(sim)
        current_mobile_topology_id = self._current_mobile_topology_id(sim)
        task_processing_topology_id = self._current_task_processing_topology_id(sim)
        sensor_ble_total_wh_per_mb, sensor_ble_data_wh_per_mb, sensor_ble_tail_wh = sensor_ble_energy_wh_per_mb()

        wireless_by_fog = {}
        for fog, distance_m in distances:
            rssi_dbm = rssi_from_distance_dbm(distance_m)
            wifi_bw_mb_s = wifi_throughput_mbps_from_rssi(rssi_dbm)
            tcp_retx_rate = tcp_retransmission_rate_from_rssi(rssi_dbm)

            udp_effective_bw_mb_s = wifi_bw_mb_s
            tcp_effective_bw_mb_s = wifi_bw_mb_s * (1.0 - tcp_retx_rate)

            tx_power_w, rx_power_w = galaxy_s4_wifi_powers_w_from_rssi(rssi_dbm)
            tail_energy_wh = tail_energy_wh_per_transfer()

            udp_tx_energy_wh_per_mb, udp_rx_energy_wh_per_mb = transfer_energy_components_wh_per_mb(
                throughput_mb_s=udp_effective_bw_mb_s,
                tx_power_w=tx_power_w,
                rx_power_w=rx_power_w,
            )
            tcp_tx_energy_wh_per_mb, tcp_rx_energy_wh_per_mb = transfer_energy_components_wh_per_mb(
                throughput_mb_s=max(tcp_effective_bw_mb_s, 1e-12),
                tx_power_w=tx_power_w,
                rx_power_w=rx_power_w,
            )

            sensor_wifi_total_wh_per_mb, sensor_wifi_data_wh_per_mb, sensor_wifi_promotion_wh, sensor_wifi_tail_wh, sensor_wifi_tx_w, sensor_wifi_rx_w, sensor_wifi_available = sensor_wifi_energy_wh_per_mb(
                rssi_dbm=rssi_dbm,
                throughput_mb_s=wifi_bw_mb_s,
            )
            wireless_by_fog[fog.topo_id] = {
                "wifi_bw_mb_s": wifi_bw_mb_s,
                "tcp_retx_rate": tcp_retx_rate,
                "mobile_tx_w": tx_power_w,
                "mobile_rx_w": rx_power_w,
                "sensor_tx_w": sensor_wifi_tx_w,
                "sensor_rx_w": sensor_wifi_rx_w,
                "sensor_wifi_available": sensor_wifi_available,
            }

            # Active sensor link depends on current execution mode.
            if execution_mode == "FOG" and sensor_wifi_available:
                sensor_active_link = "WIFI"
                sensor_active_energy_wh_per_mb = sensor_wifi_total_wh_per_mb
            else:
                sensor_active_link = "BLE"
                sensor_active_energy_wh_per_mb = sensor_ble_total_wh_per_mb

            self.buffer.append(
                {
                    "sim_time_s": step,
                    "execution_mode": execution_mode,
                    "task_processing_topology_id": task_processing_topology_id,
                    # Optional debug fields (disabled):
                    # "user_latitude": user_lat,
                    # "user_longitude": user_lon,
                    # "fog_dataset_id": fog.dataset_id,
                    # "fog_topology_id": fog.topo_id,
                    # "fog_latitude": fog.latitude,
                    # "fog_longitude": fog.longitude,
                    # "distance_m": distance_m,
                    # "rssi_dbm": rssi_dbm,
                    # "wifi_bw_mb_s": wifi_bw_mb_s,
                    # "udp_effective_bw_mb_s": udp_effective_bw_mb_s,
                    # "tcp_retransmission_rate": tcp_retx_rate,
                    # "tcp_effective_bw_mb_s": tcp_effective_bw_mb_s,
                    # "wifi_tx_power_w": tx_power_w,
                    # "wifi_rx_power_w": rx_power_w,
                    # "wifi_tail_energy_wh_per_transfer": tail_energy_wh,
                    # "udp_tx_energy_wh_per_mb": udp_tx_energy_wh_per_mb,
                    # "udp_rx_energy_wh_per_mb": udp_rx_energy_wh_per_mb,
                    # "tcp_tx_energy_wh_per_mb": tcp_tx_energy_wh_per_mb,
                    # "tcp_rx_energy_wh_per_mb": tcp_rx_energy_wh_per_mb,
                    # "sensor_active_link": sensor_active_link,
                    # "sensor_active_energy_wh_per_mb": sensor_active_energy_wh_per_mb,
                    # "sensor_wifi_available": int(sensor_wifi_available),
                    # "sensor_wifi_tx_power_w": sensor_wifi_tx_w,
                    # "sensor_wifi_rx_power_w": sensor_wifi_rx_w,
                    # "sensor_wifi_energy_wh_per_mb": sensor_wifi_total_wh_per_mb,
                    # "sensor_wifi_data_energy_wh_per_mb": sensor_wifi_data_wh_per_mb,
                    # "sensor_wifi_promotion_energy_wh_per_transfer": sensor_wifi_promotion_wh,
                    # "sensor_wifi_tail_energy_wh_per_transfer": sensor_wifi_tail_wh,
                    # "sensor_ble_energy_wh_per_mb": sensor_ble_total_wh_per_mb,
                    # "sensor_ble_data_energy_wh_per_mb": sensor_ble_data_wh_per_mb,
                    # "sensor_ble_tail_energy_wh_per_transfer": sensor_ble_tail_wh,
                    # "current_fog_topology_id": current_fog_topology_id,
                    # "current_mobile_topology_id": current_mobile_topology_id,
                    # "is_nearest": int(fog.topo_id == nearest_topo_id),
                }
            )

        self.link_registry.update_all(
            sim,
            {
                "step": step,
                "wireless_by_fog": wireless_by_fog,
            },
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


class StaticLinkMonitor:
    """
    Updates static-scenario wireless links from a fixed RSSI profile.
    """
    def __init__(self, fixed_rssi_dbm: float):
        self.fixed_rssi_dbm = float(fixed_rssi_dbm)
        self.link_registry = LinkUpdateRegistry()
        self.link_registry_initialized = False

    def _init_link_registry(self, sim) -> None:
        if self.link_registry_initialized:
            return
        for u, v in sim.topology.get_edges():
            models = {_node_model(sim, u), _node_model(sim, v)}
            types = {_node_type(sim, u), _node_type(sim, v)}
            if "cloud-device" in models or "CLOUD" in types:
                self.link_registry.register(u, v, _noop_link_update)
            elif {"mobile-device", "city-fog-device"} == models or {"mobile-device", "fog-device"} == models:
                self.link_registry.register(u, v, _update_mobile_fog_link)
            elif {"ecg-device", "city-fog-device"} == models or {"ecg-device", "fog-device"} == models:
                self.link_registry.register(u, v, _update_sensor_fog_link)
            elif {"smartwatch-device", "city-fog-device"} == models or {"smartwatch-device", "fog-device"} == models:
                self.link_registry.register(u, v, _update_sensor_fog_link)
            else:
                self.link_registry.register(u, v, _noop_link_update)
        self.link_registry_initialized = True

    def run(self, sim):
        self._init_link_registry(sim)
        rssi_dbm = self.fixed_rssi_dbm
        wifi_bw_mb_s = wifi_throughput_mbps_from_rssi(rssi_dbm)
        tcp_retx_rate = tcp_retransmission_rate_from_rssi(rssi_dbm)
        mobile_tx_w, mobile_rx_w = galaxy_s4_wifi_powers_w_from_rssi(rssi_dbm)
        sensor_tx_w, sensor_rx_w, sensor_wifi_available = sensor_wifi_data_powers_w_from_rssi(rssi_dbm)

        fog_ids = [
            node_id
            for node_id in sim.topology.G.nodes()
            if _node_model(sim, node_id) in {"city-fog-device", "fog-device"}
        ]
        wireless_state = {
            "wifi_bw_mb_s": wifi_bw_mb_s,
            "tcp_retx_rate": tcp_retx_rate,
            "mobile_tx_w": mobile_tx_w,
            "mobile_rx_w": mobile_rx_w,
            "sensor_tx_w": sensor_tx_w,
            "sensor_rx_w": sensor_rx_w,
            "sensor_wifi_available": sensor_wifi_available,
        }
        wireless_by_fog = {fog_id: wireless_state for fog_id in fog_ids}

        self.link_registry.update_all(
            sim,
            {
                "step": int(sim.env.now),
                "wireless_by_fog": wireless_by_fog,
            },
        )


def create_topology_dynamic(city_fog_devices: List[CityFogDevice]) -> Topology:
    """
    TOPOLOGY
    """
    topology_json = {}
    topology_json["entity"] = []
    topology_json["link"] = []
    model_by_id = {}

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
    cloud_dev    = {"id": 0, "model": "cloud-device", "type": "CLOUD", "IPT": 5000 * 10**6, "RAM": 40000, "WATT": 0.0}
    mobile_dev = {"id": 2, "model": "mobile-device", "type": "EDGE",
                  "IPT": ips_to_ipt(1.9 * 10**9),
                  "RAM": 2000,
                  "WATT": watt_to_wpt(1.3)}
    smartwatch_dev = {"id": 3, "model": "smartwatch-device", "type": "IOT",
                      "IPT": ips_to_ipt(768 * 10**6),
                      "RAM": 256,
                      "WATT": watt_to_wpt(0.361)}
    ecg_dev = {"id": 4, "model": "ecg-device", "type": "IOT",
               "IPT": ips_to_ipt(768 * 10**6),
               "RAM": 256,
               "WATT": watt_to_wpt(0.361)}

    topology_json["entity"].append(cloud_dev)
    topology_json["entity"].append(mobile_dev)
    topology_json["entity"].append(smartwatch_dev)
    topology_json["entity"].append(ecg_dev)
    model_by_id[cloud_dev["id"]] = cloud_dev["model"]
    model_by_id[mobile_dev["id"]] = mobile_dev["model"]
    model_by_id[smartwatch_dev["id"]] = smartwatch_dev["model"]
    model_by_id[ecg_dev["id"]] = ecg_dev["model"]

    for fog in city_fog_devices:
        fog_entity = {
            "id": fog.topo_id,
            "model": "city-fog-device",
            "type": "FOG",
            "IPT": ips_to_ipt(500 * 10**6),
            "RAM": 256,
            "WATT": watt_to_wpt(0.9),
            "LATITUDE": fog.latitude,
            "LONGITUDE": fog.longitude,
            "DETAILS": fog.details,
        }
        topology_json["entity"].append(fog_entity)
        model_by_id[fog_entity["id"]] = fog_entity["model"]
    
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
    ble_link = {
        "PR": 0,
        "s": 4,
        "BW": ips_to_ipt(0.305),
        "d": 2,
        "RTR": 0.0,
        "WATT_TRANS": watt_to_wpt(0.1807),
        "WATT_RECV": watt_to_wpt(0.1749),
    }
    assign_tail_fields_by_models(ble_link, model_by_id)
    topology_json["link"].append(ble_link)

    # SmartWatch -> Mobile (BLE).
    sw_ble_link = {
        "PR": 0,
        "s": 3,
        "BW": ips_to_ipt(0.305),
        "d": 2,
        "RTR": 0.0,
        "WATT_TRANS": watt_to_wpt(0.1807),
        "WATT_RECV": watt_to_wpt(0.1749),
    }
    assign_tail_fields_by_models(sw_ble_link, model_by_id)
    topology_json["link"].append(sw_ble_link)

    # Mobile -> each city fog (wireless uplink).
    for fog in city_fog_devices:
        mobile_fog_link = {
            "PR": 0,
            "s": 2,
            "BW": ips_to_ipt(12.5),
            "d": fog.topo_id,
            "RTR": 0.0,
            "WATT_TRANS": watt_to_wpt(0.654),
            "WATT_RECV": watt_to_wpt(3.7),
        }
        assign_tail_fields_by_models(mobile_fog_link, model_by_id)
        topology_json["link"].append(mobile_fog_link)

    # ECG sensor -> each city fog (WiFi uplink, used in FOG execution mode).
    for fog in city_fog_devices:
        sensor_fog_link = {
            "PR": 0,
            "s": 4,
            "BW": ips_to_ipt(12.5),
            "d": fog.topo_id,
            "RTR": 0.0,
            "WATT_TRANS": watt_to_wpt(0.6728),
            "WATT_RECV": watt_to_wpt(3.7),
        }
        assign_tail_fields_by_models(sensor_fog_link, model_by_id)
        topology_json["link"].append(sensor_fog_link)

    # SmartWatch -> each city fog (WiFi uplink, used in FOG execution mode).
    for fog in city_fog_devices:
        sw_fog_link = {
            "PR": 0,
            "s": 3,
            "BW": ips_to_ipt(12.5),
            "d": fog.topo_id,
            "RTR": 0.0,
            "WATT_TRANS": watt_to_wpt(0.7399),
            "WATT_RECV": watt_to_wpt(3.7),
        }
        assign_tail_fields_by_models(sw_fog_link, model_by_id)
        topology_json["link"].append(sw_fog_link)

    # Every city fog device -> cloud.
    for fog in city_fog_devices:
        fog_cloud_link = {
            "PR": 0,
            "s": fog.topo_id,
            "BW": ips_to_ipt(12.5),
            "d": 0,
            "RTR": 0.0,
            "WATT_TRANS": watt_to_wpt(4.9),
            "WATT_RECV": watt_to_wpt(3.7),
        }
        assign_tail_fields_by_models(fog_cloud_link, model_by_id)
        topology_json["link"].append(fog_cloud_link)

    t = Topology()
    t.load(topology_json)
    validate_topology_constraints(t)

    return t


def create_topology_static() -> Topology:
    topology_json = {"entity": [], "link": []}

    cloud_dev = {
        "id": 0,
        "model": "cloud-device",
        "type": "CLOUD",
        "IPT": 5000 * 10**6,
        "RAM": 40000,
        "WATT": 0.0,
    }
    fog_dev = {
        "id": 1,
        "model": "fog-device",
        "type": "FOG",
        "IPT": ips_to_ipt(500 * 10**6),
        "RAM": 256,
        "WATT": watt_to_wpt(0.9),
    }
    mobile_dev = {
        "id": 2,
        "model": "mobile-device",
        "type": "EDGE",
        "IPT": ips_to_ipt(1.9 * 10**9),
        "RAM": 2000,
        "WATT": watt_to_wpt(1.3),
    }
    smartwatch_dev = {
        "id": 3,
        "model": "smartwatch-device",
        "type": "IOT",
        "IPT": ips_to_ipt(768 * 10**6),
        "RAM": 256,
        "WATT": watt_to_wpt(0.361),
    }
    ecg_dev = {
        "id": 4,
        "model": "ecg-device",
        "type": "IOT",
        "IPT": ips_to_ipt(768 * 10**6),
        "RAM": 256,
        "WATT": watt_to_wpt(0.361),
    }

    topology_json["entity"].extend([cloud_dev, smartwatch_dev, mobile_dev, fog_dev, ecg_dev])
    model_by_id = {int(e["id"]): str(e["model"]) for e in topology_json["entity"]}

    with open(Path(__file__).parent / APP_NAME / "networkDefinition.json", "r") as f:
        data = json.load(f)
    for link in data["link"]:
        link["WATT_TRANS"] = watt_to_wpt(link["WATT_TRANS"])
        link["WATT_RECV"] = watt_to_wpt(link["WATT_RECV"])
        link["BW"] = ips_to_ipt(link["BW"])
        link["RTR"] = float(link.get("RTR", 0.0))
        assign_tail_fields_by_models(link, model_by_id)
        topology_json["link"].append(link)

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

    modules = [
        {"Cloud": {"Type": Application.TYPE_SINK}},
        {"Fog": {"RAM": 1024, "Type": Application.TYPE_MODULE}},
        {"Mobile": {"RAM": 1024, "Type": Application.TYPE_MODULE}},
        {"SmartWatch": {"Type": Application.TYPE_MODULE}},
        {"EcgSensor": {"Type": Application.TYPE_MODULE}},
        {"Virtual-SW-Gen": {"Type": Application.TYPE_SOURCE}},
        {"Virtual-ECG-Gen": {"Type": Application.TYPE_SOURCE}},
    ]
    a.set_modules(modules)

    with open(Path(__file__).parent / APP_NAME / "appDefinition.json", "r") as f:
        data = json.load(f)

    messages = {}
    for message in data["message"]:
        transport = str(message.get("transport", infer_message_transport(message["name"]))).upper()
        m = Message(
            message["name"],
            message["s"],
            message["d"],
            instructions=message["instructions"],
            bytes=message["bytes"],
            transport=transport,
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


def build_task_latency_report(stats: Stats) -> pd.DataFrame:
    latency_by_message = pd.concat(
        [stats.times("time_latency"), stats.times("time_service"), stats.times("time_total_response")],
        axis=1,
    )
    task_labels = latency_by_message.index.to_series().str.extract(r"^(M\.TASK\d+)")[0].fillna("UNCLASSIFIED")
    latency_by_task = latency_by_message.groupby(task_labels).sum(numeric_only=True)
    if not latency_by_task.empty:
        latency_by_task.loc["mean"] = latency_by_task.mean()
    return latency_by_task


def main_dynamic(stop_time, it, folder_results):
    global CURRENT_EXECUTION_MODE
    CURRENT_EXECUTION_MODE = str(cfg.TASK_EXECUTION_MODE).upper()
    sim_tag = build_simulation_tag()
    trace_basename = f"sim_trace_{sim_tag}"

    dataset_dir = Path(__file__).parent / f"{APP_NAME}/dataset"
    city_fog_devices = load_city_fog_devices(dataset_dir)
    user_trace = load_user_trace(dataset_dir)
    mobility_model = MobilityModel(user_trace=user_trace, fog_devices=city_fog_devices)

    """
    TOPOLOGY
    """
    t = create_topology_dynamic(city_fog_devices)

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
            {"app": APP_NAME, "module_name": "SmartWatch", "id_resource": 3},
            {"app": APP_NAME, "module_name": "EcgSensor", "id_resource": 4}
        ]
    }
    placement_dist = deterministic_distribution(name="MobilityPlacementTick", time=cfg.SIM_STEP_SECONDS)
    decision_output = Path(folder_results) / f"offloading_decisions_{sim_tag}.csv"
    placement = LPOptimizationPlacement(
        name=f"MobilityPlacement_{cfg.OPTIMIZATION_METHOD}",
        json=placementJson,
        activation_dist=placement_dist,
        mobility_model=mobility_model,
        app_name=APP_NAME,
        output_path=decision_output,
        mode_getter=get_execution_mode,
        mode_setter=set_execution_mode,
        rssi_from_distance_dbm=rssi_from_distance_dbm,
        wifi_throughput_mbps_from_rssi=wifi_throughput_mbps_from_rssi,
        tcp_retransmission_rate_from_rssi=tcp_retransmission_rate_from_rssi,
        sensor_ble_energy_wh_per_mb=sensor_ble_energy_wh_per_mb,
        sensor_wifi_data_powers_w_from_rssi=sensor_wifi_data_powers_w_from_rssi,
        galaxy_s4_wifi_powers_w_from_rssi=galaxy_s4_wifi_powers_w_from_rssi,
        fixed_overhead_energy_wh=fixed_overhead_energy_wh,
        tail_energy_wh_per_transfer=tail_energy_wh_per_transfer,
        kb_to_mb=kb_to_mb,
        mi_to_instructions=mi_to_instructions,
        solve_lp_two_mode=solve_lp_two_mode,
        ips_to_ipt=ips_to_ipt,
        watt_to_wpt=watt_to_wpt,
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
    pop.set_sink_control({"model": "cloud-device", "number":1, "module":app.get_sink_modules()})

    #In addition, a source includes a distribution function:
    dDistribution1 = deterministic_distribution(name="Deterministic", time=cfg.TASK1_PERIOD_S)
    pop.set_src_control({"model": "ecg-device", "number":1, "message": app.get_message("M.TASK1.TCP.Generation"), "distribution": dDistribution1})
    dDistribution2 = deterministic_distribution(name="Deterministic", time=cfg.TASK2_PERIOD_S)
    pop.set_src_control({"model": "smartwatch-device", "number":1, "message": app.get_message("M.TASK2.UDP.Generation"), "distribution": dDistribution2})
    dDistribution3 = deterministic_distribution(name="Deterministic", time=cfg.TASK3_PERIOD_S)
    pop.set_src_control({"model": "smartwatch-device", "number":1, "message": app.get_message("M.TASK3.TCP.Generation"), "distribution": dDistribution3})

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
        time=cfg.SIM_STEP_SECONDS,
    )
    s.deploy_monitor(
        f"MobilityDistanceMonitor_{cfg.OPTIMIZATION_METHOD}",
        distance_monitor.run,
        monitor_dist,
        sim=s,
    )

    """
    RUNNING
    """
    logging.info(" Performing simulation: %i " % it)
    s.run(stop_time)  # To test deployments put test_initial_deploy a TRUE
    distance_monitor.flush()
    placement.flush()
    s.print_debug_assignaments()

    if cfg.ENABLE_ANIMATION:
        generate_mobility_animation(
            mobility_model=mobility_model,
            output_path=Path(folder_results) / f"mobility_placement_{sim_tag}.gif",
            app_name=APP_NAME,
            optimization_method=cfg.OPTIMIZATION_METHOD,
            current_execution_mode=CURRENT_EXECUTION_MODE,
            animation_format=cfg.ANIMATION_FORMAT,
            animation_step_stride=cfg.ANIMATION_STEP_STRIDE,
            placement_history_path=distance_output,
            simulated_until_step=max(0, int(stop_time) - 1),
        )

    generate_energy_decision_plot(
        decision_history_path=decision_output,
        output_path=Path(folder_results) / f"energy_decision_{sim_tag}.png",
        app_name=APP_NAME,
        optimization_method=cfg.OPTIMIZATION_METHOD,
        decision_period_s=cfg.OFFLOADING_DECISION_PERIOD_S,
    )

    s1 = Stats(defaultPath=os.path.join(os.getcwd(), folder_results, trace_basename))

    # Consumption and number of bytes report
    s1.showResults(total_time=stop_time, topology=t, multiplier=1)

    # Latency report
    print("\nLatency Report (in time unit):")
    latency = pd.concat([s1.times("time_latency"), s1.times("time_service"), s1.times("time_total_response")], axis=1)
    latency.loc["Total"] = latency.sum()
    print(latency)
    print("\nLatency Report by Task (in time unit):")
    print(build_task_latency_report(s1))


def main_static(stop_time, it, folder_results):
    sim_tag = build_simulation_tag()
    trace_basename = f"sim_trace_{sim_tag}"

    t = create_topology_static()

    print(t.G.nodes())

    pos = nx.spring_layout(t.G)
    nx.draw_networkx(t.G, pos, with_labels=True)
    nx.draw_networkx_edge_labels(t.G, pos, alpha=0.5, font_size=5, verticalalignment="top")

    app = create_application()

    module_to_node = {
        "Fog": 1,
        "Mobile": 2,
        "SmartWatch": 3,
        "EcgSensor": 4,
    }
    initial_allocation = [
        {"app": APP_NAME, "module_name": module_name, "id_resource": node_id}
        for module_name, node_id in module_to_node.items()
        if module_name in app.services
    ]
    placement_json = {"initialAllocation": initial_allocation}
    placement = JSONPlacement(name="Placement", json=placement_json)

    pop = Statical("Statical")
    pop.set_sink_control({"model": "cloud-device", "number": 1, "module": app.get_sink_modules()})

    d_ecg = deterministic_distribution(name="Deterministic", time=cfg.TASK1_PERIOD_S)
    pop.set_src_control({
        "model": "ecg-device",
        "number": 1,
        "message": app.get_message("M.TASK1.TCP.Generation"),
        "distribution": d_ecg,
    })
    d_sw = deterministic_distribution(name="Deterministic", time=cfg.TASK2_PERIOD_S)
    pop.set_src_control({
        "model": "smartwatch-device",
        "number": 1,
        "message": app.get_message("M.TASK2.UDP.Generation"),
        "distribution": d_sw,
    })
    d_sw_t3 = deterministic_distribution(name="Deterministic", time=cfg.TASK3_PERIOD_S)
    pop.set_src_control({
        "model": "smartwatch-device",
        "number": 1,
        "message": app.get_message("M.TASK3.TCP.Generation"),
        "distribution": d_sw_t3,
    })
    d_sw_t4 = deterministic_distribution(name="Deterministic", time=cfg.TASK4_PERIOD_S)
    pop.set_src_control({
        "model": "smartwatch-device",
        "number": 1,
        "message": app.get_message("M.TASK4.TCP.Generation"),
        "distribution": d_sw_t4,
    })
    d_ecg_t5 = deterministic_distribution(name="Deterministic", time=cfg.TASK5_PERIOD_S)
    pop.set_src_control({
        "model": "ecg-device",
        "number": 1,
        "message": app.get_message("M.TASK5.TCP.Generation"),
        "distribution": d_ecg_t5,
    })

    selector_path = First_ShortestPath()
    s = Sim(t, default_results_path=folder_results + trace_basename)
    s.deploy_app2(app, placement, pop, selector_path)

    static_link_monitor = StaticLinkMonitor(fixed_rssi_dbm=cfg.STATIC_LINK_RSSI_DBM)
    static_monitor_dist = deterministicDistributionStartPoint(
        name="StaticLinkTick",
        start=0,
        time=cfg.SIM_STEP_SECONDS,
    )
    s.deploy_monitor(
        "StaticLinkMonitor",
        static_link_monitor.run,
        static_monitor_dist,
        sim=s,
    )

    logging.info(" Performing simulation: %i ", it)
    s.run(stop_time)
    s.print_debug_assignaments()

    s1 = Stats(defaultPath=os.path.join(os.getcwd(), folder_results, trace_basename))
    s1.showResults(total_time=stop_time, topology=t, multiplier=1)

    print("\nLatency Report (in time unit):")
    latency = pd.concat([s1.times("time_latency"), s1.times("time_service"), s1.times("time_total_response")], axis=1)
    latency.loc["Total"] = latency.sum()
    print(latency)
    print("\nLatency Report by Task (in time unit):")
    print(build_task_latency_report(s1))


def parse_runtime_args():
    parser = argparse.ArgumentParser(description="YAFS offloading simulation runner")
    parser.add_argument(
        "--scenario",
        choices=sorted(ALL_SCENARIOS),
        default=DEFAULT_APP,
        help="Scenario name. CityScenario runs dynamic mode; Fog/Hybrid/Mobile run static mode.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Optional simulation duration override (seconds).",
    )
    return parser.parse_args()


if __name__ == '__main__':
    LOGGING_CONFIG = Path(__file__).parent / 'logging.ini'
    logging.config.fileConfig(LOGGING_CONFIG)

    folder_results = Path("results/")
    folder_results.mkdir(parents=True, exist_ok=True)
    folder_results = str(folder_results)+"/"

    args = parse_runtime_args()
    APP_NAME = args.scenario
    SIMULATION_MODE = "dynamic" if APP_NAME in DYNAMIC_SCENARIOS else "static"
    nIterations = 1  # iteration for each experiment
    simulationDuration = 3600
    if args.duration is not None:
        simulationDuration = args.duration
    elif APP_NAME == "CityScenario":
        dataset_dir = Path(__file__).parent / f"{APP_NAME}/dataset"
        user_trace = load_user_trace(dataset_dir)
        simulationDuration = len(user_trace)

    logging.info("Runtime configuration: scenario=%s duration=%s", APP_NAME, simulationDuration)

    # Iteration for each experiment changing the seed of randoms
    for iteration in range(nIterations):
        random.seed(iteration)
        logging.info("Running experiment it: - %i" % iteration)

        start_time = time.time()
        if SIMULATION_MODE == "dynamic":
            main_dynamic(stop_time=simulationDuration, it=iteration, folder_results=folder_results)
        else:
            main_static(stop_time=simulationDuration, it=iteration, folder_results=folder_results)

        print("\n--- %s seconds ---" % (time.time() - start_time))

    print("Simulation Done!")
    sim_tag = build_simulation_tag()
    trace_basename = f"sim_trace_{sim_tag}"
  
    # Analysing the results. 
    dfl = pd.read_csv(folder_results + trace_basename + "_link.csv")
    print("Number of total messages between nodes: %i"%len(dfl))

    df = pd.read_csv(folder_results + trace_basename + ".csv")
    print("Number of requests handled by deployed services: %i"%len(df))
