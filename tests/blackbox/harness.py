from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import sys
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
SIMULATION_PATH = PROJECT_ROOT / "simulation"

for import_path in (str(SRC_PATH), str(SIMULATION_PATH)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

import config as cfg
from device_profiles import ECG_SENSOR_DEVICE, SAMSUNG_S4_MOBILE_DEVICE
from offloading import (
    assign_tail_fields_by_models,
    galaxy_s4_wifi_powers_w_from_rssi,
    ips_to_ipt,
    sensor_ble_data_bw_mb_s,
    sensor_ble_rx_power_w,
    sensor_ble_tx_power_w,
    sensor_wifi_data_powers_w_from_rssi,
    tcp_retransmission_rate_from_rssi,
    watt_to_wpt,
    wifi_throughput_mbps_from_rssi,
)
from yafs.application import Application, Message
from yafs.core import Sim
from yafs.distribution import deterministicDistributionStartPoint
from yafs.metrics import Metrics
from yafs.placement import JSONPlacement
from yafs.population import Statical
from yafs.selection import First_ShortestPath
from yafs.stats import Stats
from yafs.topology import Topology


APP_NAME = "BlackBoxSimulationApp"
RESULT_BASENAME = "sim_trace_blackbox"

RSSI_ANCHORS = (-90, -85, -80, -70, -65, -60, -55, -50, -42)
FAST_RSSI_ANCHORS = (-90, -70, -42)

TOPOLOGY_BLE_MOBILE_FOG = "sensor_ble_mobile_wifi_fog_eth_cloud"
TOPOLOGY_WIFI_FOG = "sensor_wifi_fog_wifi_mobile_eth_cloud"
TOPOLOGIES = (TOPOLOGY_BLE_MOBILE_FOG, TOPOLOGY_WIFI_FOG)
PROCESSING_MODES = ("EDGE", "FOG")


@dataclass(frozen=True)
class BlackBoxCase:
    topology_name: str
    processing_mode: str
    rssi_dbm: int


def case_key(case: BlackBoxCase) -> str:
    return (
        f"topology={case.topology_name}|"
        f"mode={case.processing_mode}|"
        f"rssi={case.rssi_dbm}"
    )


def case_id(case: BlackBoxCase) -> str:
    return f"{case.topology_name}:{case.processing_mode}:rssi{case.rssi_dbm}"


ALL_CASES = tuple(
    BlackBoxCase(topology_name=topology, processing_mode=mode, rssi_dbm=rssi)
    for topology in TOPOLOGIES
    for mode in PROCESSING_MODES
    for rssi in RSSI_ANCHORS
)

FAST_CASES = tuple(
    case for case in ALL_CASES if case.rssi_dbm in FAST_RSSI_ANCHORS
)


UDP_EXACT_FLOAT_KEYS = {
    "udp_rtr_max",
    "udp_latency_sum",
    "udp_tx_wh_sum",
    "udp_rx_wh_sum",
    "udp_tail_tx_wh_sum",
    "udp_tail_rx_wh_sum",
}


def _always_true(threshold: float) -> bool:
    return True


def _mode_selectivity(threshold: float, mode: str, active_mode: str) -> bool:
    return str(mode).upper() == str(active_mode).upper()


def _to_int(value: float | int) -> int:
    return int(value)


def _to_float(value: float | int, digits: int = 12) -> float:
    return round(float(value), digits)


def _sum_int(df: pd.DataFrame, col: str) -> int:
    if df.empty:
        return 0
    return _to_int(df[col].sum())


def _sum_float(df: pd.DataFrame, col: str) -> float:
    if df.empty:
        return 0.0
    return _to_float(df[col].sum())


def _min_int(df: pd.DataFrame, col: str) -> int:
    if df.empty:
        return 0
    return _to_int(df[col].min())


def _max_int(df: pd.DataFrame, col: str) -> int:
    if df.empty:
        return 0
    return _to_int(df[col].max())


def _max_float(df: pd.DataFrame, col: str) -> float:
    if df.empty:
        return 0.0
    return _to_float(df[col].max())


def _mean_float(df: pd.DataFrame, col: str) -> float:
    if df.empty:
        return 0.0
    return _to_float(df[col].mean())


def _build_topology(
    topology_name: str,
    rssi_dbm: float,
) -> Topology:
    model_by_id = {
        0: "cloud-device",
        1: "fog-device",
        2: SAMSUNG_S4_MOBILE_DEVICE.get_model(2),
        3: ECG_SENSOR_DEVICE.get_model(3),
    }
    type_by_id = {
        0: "CLOUD",
        1: "FOG",
        2: "EDGE",
        3: "IOT",
    }

    entities = [
        {
            "id": 0,
            "model": "cloud-device",
            "type": "CLOUD",
            "IPT": 5000 * 10**6,
            "RAM": 40000,
            "WATT": 0.0,
        },
        {
            "id": 1,
            "model": "fog-device",
            "type": "FOG",
            "IPT": ips_to_ipt(500 * 10**6),
            "RAM": 256,
            "WATT": watt_to_wpt(0.9),
        },
        {
            "id": 2,
            "model": SAMSUNG_S4_MOBILE_DEVICE.get_model(2),
            "type": "EDGE",
            "IPT": ips_to_ipt(1.9 * 10**9),
            "RAM": 2000,
            "WATT": watt_to_wpt(1.3),
        },
        {
            "id": 3,
            "model": ECG_SENSOR_DEVICE.get_model(3),
            "type": "IOT",
            "IPT": ips_to_ipt(768 * 10**6),
            "RAM": 256,
            "WATT": watt_to_wpt(0.361),
        },
    ]

    wifi_bw = wifi_throughput_mbps_from_rssi(rssi_dbm)
    tcp_rtr = tcp_retransmission_rate_from_rssi(rssi_dbm)
    mobile_tx_w, mobile_rx_w = galaxy_s4_wifi_powers_w_from_rssi(rssi_dbm)
    sensor_tx_w, sensor_rx_w, _ = sensor_wifi_data_powers_w_from_rssi(rssi_dbm)

    links = []

    def add_link(link: Dict[str, float]) -> None:
        assign_tail_fields_by_models(link, model_by_id, type_by_id)
        links.append(link)

    add_link(
        {
            "PR": 0.0,
            "s": 1,
            "d": 0,
            "BW": ips_to_ipt(12.5),
            "RTR": 0.0,
            "WATT_TRANS": watt_to_wpt(4.9),
            "WATT_RECV": watt_to_wpt(5.0),
        }
    )

    if topology_name == TOPOLOGY_BLE_MOBILE_FOG:
        add_link(
            {
                "PR": 0.0,
                "s": 3,
                "d": 2,
                "BW": sensor_ble_data_bw_mb_s(),
                "RTR": 0.0,
                "WATT_TRANS": watt_to_wpt(sensor_ble_tx_power_w()),
                "WATT_RECV": watt_to_wpt(sensor_ble_rx_power_w()),
                "WATT_TRANS_3-2": watt_to_wpt(sensor_ble_tx_power_w()),
                "WATT_RECV_3-2": watt_to_wpt(sensor_ble_rx_power_w()),
                "WATT_TRANS_2-3": watt_to_wpt(sensor_ble_tx_power_w()),
                "WATT_RECV_2-3": watt_to_wpt(sensor_ble_rx_power_w()),
            }
        )
        add_link(
            {
                "PR": 0.0,
                "s": 2,
                "d": 1,
                "BW": wifi_bw,
                "RTR": tcp_rtr,
                "WATT_TRANS": watt_to_wpt(4.9),
                "WATT_RECV": watt_to_wpt(3.7),
                "WATT_TRANS_2-1": watt_to_wpt(mobile_tx_w),
                "WATT_RECV_2-1": watt_to_wpt(3.7),
                "WATT_TRANS_1-2": watt_to_wpt(4.9),
                "WATT_RECV_1-2": watt_to_wpt(mobile_rx_w),
            }
        )
    elif topology_name == TOPOLOGY_WIFI_FOG:
        add_link(
            {
                "PR": 0.0,
                "s": 3,
                "d": 1,
                "BW": wifi_bw,
                "RTR": tcp_rtr,
                "WATT_TRANS": watt_to_wpt(4.9),
                "WATT_RECV": watt_to_wpt(3.7),
                "WATT_TRANS_3-1": watt_to_wpt(sensor_tx_w),
                "WATT_RECV_3-1": watt_to_wpt(3.7),
                "WATT_TRANS_1-3": watt_to_wpt(4.9),
                "WATT_RECV_1-3": watt_to_wpt(sensor_rx_w),
            }
        )
        add_link(
            {
                "PR": 0.0,
                "s": 1,
                "d": 2,
                "BW": wifi_bw,
                "RTR": tcp_rtr,
                "WATT_TRANS": watt_to_wpt(4.9),
                "WATT_RECV": watt_to_wpt(3.7),
                "WATT_TRANS_2-1": watt_to_wpt(mobile_tx_w),
                "WATT_RECV_2-1": watt_to_wpt(3.7),
                "WATT_TRANS_1-2": watt_to_wpt(4.9),
                "WATT_RECV_1-2": watt_to_wpt(mobile_rx_w),
            }
        )
    else:
        raise ValueError(f"Unknown topology: {topology_name}")

    topology_json = {"entity": entities, "link": links}
    topology = Topology()
    topology.load(topology_json)
    return topology


def _build_application(active_mode: str) -> Application:
    app = Application(name=APP_NAME)
    app.set_modules(
        [
            {"Cloud": {"Type": Application.TYPE_SINK}},
            {"Fog": {"Type": Application.TYPE_MODULE}},
            {"Mobile": {"Type": Application.TYPE_MODULE}},
            {"Sensor": {"Type": Application.TYPE_MODULE}},
        ]
    )

    msg_tcp_gen = Message(
        "M.TCP.Generation",
        "None",
        "Sensor",
        instructions=0,
        bytes=36864,
        transport="TCP",
    )
    msg_tcp_edge_req = Message(
        "M.TCP.Sensor-EDGE",
        "Sensor",
        "Mobile",
        instructions=500000000,
        bytes=36864,
        transport="TCP",
    )
    msg_tcp_fog_req = Message(
        "M.TCP.Sensor-FOG",
        "Sensor",
        "Fog",
        instructions=500000000,
        bytes=36864,
        transport="TCP",
    )
    msg_tcp_edge_result = Message(
        "M.TCP.EDGE-CLOUD",
        "Mobile",
        "Cloud",
        instructions=0,
        bytes=740,
        transport="TCP",
    )
    msg_tcp_fog_result = Message(
        "M.TCP.FOG-CLOUD",
        "Fog",
        "Cloud",
        instructions=0,
        bytes=740,
        transport="TCP",
    )

    msg_udp_gen = Message(
        "M.UDP.Generation",
        "None",
        "Sensor",
        instructions=0,
        bytes=512000,
        transport="UDP",
    )
    msg_udp_edge_req = Message(
        "M.UDP.Sensor-EDGE",
        "Sensor",
        "Mobile",
        instructions=1000000000,
        bytes=512000,
        transport="UDP",
    )
    msg_udp_fog_req = Message(
        "M.UDP.Sensor-FOG",
        "Sensor",
        "Fog",
        instructions=1000000000,
        bytes=512000,
        transport="UDP",
    )
    msg_udp_edge_result = Message(
        "M.UDP.EDGE-CLOUD",
        "Mobile",
        "Cloud",
        instructions=0,
        bytes=4096,
        transport="UDP",
    )
    msg_udp_fog_result = Message(
        "M.UDP.FOG-CLOUD",
        "Fog",
        "Cloud",
        instructions=0,
        bytes=4096,
        transport="UDP",
    )

    app.add_source_messages(msg_tcp_gen)
    app.add_source_messages(msg_udp_gen)

    app.add_service_module(
        "Sensor",
        msg_tcp_gen,
        msg_tcp_edge_req,
        _mode_selectivity,
        threshold=1.0,
        mode="EDGE",
        active_mode=active_mode,
    )
    app.add_service_module(
        "Sensor",
        msg_tcp_gen,
        msg_tcp_fog_req,
        _mode_selectivity,
        threshold=1.0,
        mode="FOG",
        active_mode=active_mode,
    )
    app.add_service_module(
        "Sensor",
        msg_udp_gen,
        msg_udp_edge_req,
        _mode_selectivity,
        threshold=1.0,
        mode="EDGE",
        active_mode=active_mode,
    )
    app.add_service_module(
        "Sensor",
        msg_udp_gen,
        msg_udp_fog_req,
        _mode_selectivity,
        threshold=1.0,
        mode="FOG",
        active_mode=active_mode,
    )

    app.add_service_module(
        "Mobile",
        msg_tcp_edge_req,
        msg_tcp_edge_result,
        _always_true,
        threshold=1.0,
    )
    app.add_service_module(
        "Mobile",
        msg_udp_edge_req,
        msg_udp_edge_result,
        _always_true,
        threshold=1.0,
    )

    app.add_service_module(
        "Fog",
        msg_tcp_fog_req,
        msg_tcp_fog_result,
        _always_true,
        threshold=1.0,
    )
    app.add_service_module(
        "Fog",
        msg_udp_fog_req,
        msg_udp_fog_result,
        _always_true,
        threshold=1.0,
    )

    app.add_service_module("Cloud", msg_tcp_edge_result)
    app.add_service_module("Cloud", msg_tcp_fog_result)
    app.add_service_module("Cloud", msg_udp_edge_result)
    app.add_service_module("Cloud", msg_udp_fog_result)

    return app


def _build_population(app: Application) -> Statical:
    population = Statical("StaticPopulation")
    population.set_sink_control(
        {"model": "cloud-device", "number": 1, "module": app.get_sink_modules()}
    )

    dist_tcp = deterministicDistributionStartPoint(
        name="TCPStart",
        start=1,
        time=10,
    )
    dist_udp = deterministicDistributionStartPoint(
        name="UDPStart",
        start=2,
        time=10,
    )

    population.set_src_control(
        {
            "model": ECG_SENSOR_DEVICE.get_model(3),
            "number": 1,
            "message": app.get_message("M.TCP.Generation"),
            "distribution": dist_tcp,
        }
    )
    population.set_src_control(
        {
            "model": ECG_SENSOR_DEVICE.get_model(3),
            "number": 1,
            "message": app.get_message("M.UDP.Generation"),
            "distribution": dist_udp,
        }
    )
    return population


def _build_placement() -> JSONPlacement:
    placement_json = {
        "initialAllocation": [
            {"app": APP_NAME, "module_name": "Fog", "id_resource": 1},
            {"app": APP_NAME, "module_name": "Mobile", "id_resource": 2},
            {"app": APP_NAME, "module_name": "Sensor", "id_resource": 3},
        ]
    }
    return JSONPlacement(name="StaticPlacement", json=placement_json)


def _snapshot_from_results(
    default_results_path: Path,
    total_time: int,
    topology: Topology,
) -> Dict[str, float | int]:
    events = pd.read_csv(f"{default_results_path}.csv")
    links = pd.read_csv(f"{default_results_path}_link.csv")

    links = links.copy()
    links["transport"] = np.where(
        links["message"].str.contains(".UDP.", regex=False),
        "UDP",
        "TCP",
    )
    tcp_links = links[links["transport"] == "TCP"]
    udp_links = links[links["transport"] == "UDP"]

    stats = Stats(defaultPath=str(default_results_path))
    service_watt = stats.get_watt(total_time, topology, Metrics.WATT_SERVICE)
    link_watt = stats.get_watt(total_time, topology, Metrics.WATT_LINK)

    def service_node(node_id: int) -> float:
        return _to_float(service_watt.get(node_id, {}).get("watt", 0.0))

    def link_tx_node(node_id: int) -> float:
        return _to_float(link_watt.get(node_id, {}).get("watt_trans", 0.0))

    def link_rx_node(node_id: int) -> float:
        return _to_float(link_watt.get(node_id, {}).get("watt_recv", 0.0))

    return {
        "rows_events": _to_int(len(events)),
        "rows_links": _to_int(len(links)),
        "bytes_total": _to_int(links["size"].sum()),
        "tcp_rows": _to_int(len(tcp_links)),
        "udp_rows": _to_int(len(udp_links)),
        "tcp_attempts_sum": _sum_int(tcp_links, "attempts"),
        "tcp_attempts_max": _max_int(tcp_links, "attempts"),
        "tcp_rtr_mean": _mean_float(tcp_links, "rtr"),
        "tcp_size_sum": _sum_int(tcp_links, "size"),
        "tcp_latency_sum": _sum_float(tcp_links, "latency"),
        "tcp_tx_wh_sum": _sum_float(tcp_links, "tx_wh"),
        "tcp_rx_wh_sum": _sum_float(tcp_links, "rx_wh"),
        "tcp_tail_tx_wh_sum": _sum_float(tcp_links, "tail_tx_wh"),
        "tcp_tail_rx_wh_sum": _sum_float(tcp_links, "tail_rx_wh"),
        "udp_attempts_sum": _sum_int(udp_links, "attempts"),
        "udp_attempts_min": _min_int(udp_links, "attempts"),
        "udp_attempts_max": _max_int(udp_links, "attempts"),
        "udp_rtr_max": _max_float(udp_links, "rtr"),
        "udp_size_sum": _sum_int(udp_links, "size"),
        "udp_latency_sum": _sum_float(udp_links, "latency"),
        "udp_tx_wh_sum": _sum_float(udp_links, "tx_wh"),
        "udp_rx_wh_sum": _sum_float(udp_links, "rx_wh"),
        "udp_tail_tx_wh_sum": _sum_float(udp_links, "tail_tx_wh"),
        "udp_tail_rx_wh_sum": _sum_float(udp_links, "tail_rx_wh"),
        "mobile_service_wh": service_node(2),
        "fog_service_wh": service_node(1),
        "mobile_link_tx_wh": link_tx_node(2),
        "mobile_link_rx_wh": link_rx_node(2),
        "fog_link_tx_wh": link_tx_node(1),
        "fog_link_rx_wh": link_rx_node(1),
        "sensor_link_tx_wh": link_tx_node(3),
        "sensor_link_rx_wh": link_rx_node(3),
        "total_link_tx_wh": _sum_float(links, "tx_wh"),
        "total_link_rx_wh": _sum_float(links, "rx_wh"),
        "total_link_tail_tx_wh": _sum_float(links, "tail_tx_wh"),
        "total_link_tail_rx_wh": _sum_float(links, "tail_rx_wh"),
    }


def snapshot_to_golden_payload(
    snapshot: Dict[str, float | int],
) -> Dict[str, Dict[str, float | int]]:
    exact: Dict[str, float | int] = {}
    approx: Dict[str, float | int] = {}

    for key, value in snapshot.items():
        if isinstance(value, int):
            exact[key] = value
            continue

        if key.startswith("udp_") or key in UDP_EXACT_FLOAT_KEYS:
            exact[key] = value
        else:
            approx[key] = value

    return {"exact": exact, "approx": approx}


def run_case(
    case: BlackBoxCase,
    output_dir: Path,
    duration_s: int = 35,
) -> Dict[str, float | int]:
    random.seed(0)
    np.random.seed(0)

    rssi_dbm = float(cfg.STATIC_LINK_RSSI_DBM)
    if int(round(rssi_dbm)) != int(case.rssi_dbm):
        raise ValueError(
            "cfg.STATIC_LINK_RSSI_DBM must match case.rssi_dbm. "
            f"cfg={rssi_dbm}, case={case.rssi_dbm}"
        )

    topology = _build_topology(case.topology_name, rssi_dbm)
    app = _build_application(case.processing_mode)
    placement = _build_placement()
    population = _build_population(app)
    selector = First_ShortestPath()

    output_dir.mkdir(parents=True, exist_ok=True)
    default_results_path = output_dir / RESULT_BASENAME

    sim = Sim(topology, default_results_path=str(default_results_path))
    sim.deploy_app2(app, placement, population, selector)
    sim.run(duration_s)

    return _snapshot_from_results(default_results_path, duration_s, topology)


def iter_case_pairs_for_processing_checks() -> Iterable[Tuple[BlackBoxCase, BlackBoxCase]]:
    for topology_name in TOPOLOGIES:
        edge_case = BlackBoxCase(
            topology_name=topology_name,
            processing_mode="EDGE",
            rssi_dbm=-70,
        )
        fog_case = BlackBoxCase(
            topology_name=topology_name,
            processing_mode="FOG",
            rssi_dbm=-70,
        )
        yield edge_case, fog_case


def iter_case_pairs_for_retransmission_checks() -> Iterable[Tuple[BlackBoxCase, BlackBoxCase]]:
    for topology_name in TOPOLOGIES:
        for mode in PROCESSING_MODES:
            strong_case = BlackBoxCase(
                topology_name=topology_name,
                processing_mode=mode,
                rssi_dbm=-42,
            )
            weak_case = BlackBoxCase(
                topology_name=topology_name,
                processing_mode=mode,
                rssi_dbm=-90,
            )
            yield strong_case, weak_case
