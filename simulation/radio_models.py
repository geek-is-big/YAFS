from typing import Tuple

import numpy as np

import config as cfg
from device_profiles import (
    ECG_SENSOR_DEVICE,
    LG_URBANE_SMARTWATCH_DEVICE,
    SAMSUNG_S4_MOBILE_DEVICE,
    resolve_device_profile,
)
from utils import seconds_to_tu, watt_to_wpt


DEFAULT_MOBILE_MODEL = SAMSUNG_S4_MOBILE_DEVICE.get_model(2)
DEFAULT_SENSOR_MODEL = ECG_SENSOR_DEVICE.get_model(4)
DEFAULT_FOG_MODEL = "fog-device"
DEFAULT_EDGE_TYPE = "EDGE"
DEFAULT_SENSOR_TYPE = "IOT"
DEFAULT_FOG_TYPE = "FOG"

FOG_WIRELESS_TX_POWER_W = 4.9
FOG_WIRELESS_RX_POWER_W = 3.7


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
    if d0_m <= 0:
        raise ValueError("Reference distance d0_m must be > 0")
    effective_d = max(distance_m, d0_m)
    return rssi0_dbm - (10.0 * n * np.log10(effective_d / d0_m))


def _wifi_cap_mb_s(model: str, node_type: str, rssi_dbm: float) -> float:
    normalized_model = str(model).lower()
    normalized_type = str(node_type).upper()
    if normalized_model in {"fog-device", "city-fog-device"} or normalized_type == "FOG":
        return float("inf")
    profile = resolve_device_profile(model, node_type)
    return float(profile.wifi_throughput_mbps_from_rssi(rssi_dbm))


def _tcp_rtr(model: str, node_type: str, rssi_dbm: float) -> float:
    normalized_model = str(model).lower()
    normalized_type = str(node_type).upper()
    if normalized_model in {"fog-device", "city-fog-device"} or normalized_type == "FOG":
        return 0.0
    profile = resolve_device_profile(model, node_type)
    return float(profile.tcp_retransmission_rate_from_rssi(rssi_dbm))


def _wifi_tx_rx_w(model: str, node_type: str, rssi_dbm: float) -> Tuple[float, float]:
    normalized_model = str(model).lower()
    normalized_type = str(node_type).upper()
    if normalized_model in {"fog-device", "city-fog-device"} or normalized_type == "FOG":
        return FOG_WIRELESS_TX_POWER_W, FOG_WIRELESS_RX_POWER_W
    profile = resolve_device_profile(model, node_type)
    tx_w, rx_w = profile.wifi_tx_rx_powers_w_from_rssi(rssi_dbm)
    return float(tx_w), float(rx_w)


def wireless_wifi_bw_mbps_from_models(
    src_model: str,
    src_type: str,
    dst_model: str,
    dst_type: str,
    rssi_dbm: float,
) -> float:
    src_cap = _wifi_cap_mb_s(src_model, src_type, rssi_dbm)
    dst_cap = _wifi_cap_mb_s(dst_model, dst_type, rssi_dbm)
    return min(src_cap, dst_cap)


def wireless_tcp_retransmission_rate_from_models(
    src_model: str,
    src_type: str,
    dst_model: str,
    dst_type: str,
    rssi_dbm: float,
) -> float:
    src_rtr = _tcp_rtr(src_model, src_type, rssi_dbm)
    dst_rtr = _tcp_rtr(dst_model, dst_type, rssi_dbm)
    return max(src_rtr, dst_rtr)


def wireless_directional_powers_w_from_models(
    src_model: str,
    src_type: str,
    dst_model: str,
    dst_type: str,
    rssi_dbm: float,
) -> Tuple[float, float]:
    src_tx_w, _ = _wifi_tx_rx_w(src_model, src_type, rssi_dbm)
    _, dst_rx_w = _wifi_tx_rx_w(dst_model, dst_type, rssi_dbm)
    return src_tx_w, dst_rx_w


def mobile_wifi_throughput_mbps_from_rssi(rssi_dbm: float) -> float:
    return wireless_wifi_bw_mbps_from_models(
        src_model=DEFAULT_MOBILE_MODEL,
        src_type=DEFAULT_EDGE_TYPE,
        dst_model=DEFAULT_FOG_MODEL,
        dst_type=DEFAULT_FOG_TYPE,
        rssi_dbm=rssi_dbm,
    )


def sensor_wifi_throughput_mbps_from_rssi(rssi_dbm: float) -> float:
    return wireless_wifi_bw_mbps_from_models(
        src_model=DEFAULT_SENSOR_MODEL,
        src_type=DEFAULT_SENSOR_TYPE,
        dst_model=DEFAULT_FOG_MODEL,
        dst_type=DEFAULT_FOG_TYPE,
        rssi_dbm=rssi_dbm,
    )


def mobile_to_fog_wifi_powers_w_from_rssi(rssi_dbm: float) -> Tuple[float, float]:
    return wireless_directional_powers_w_from_models(
        src_model=DEFAULT_MOBILE_MODEL,
        src_type=DEFAULT_EDGE_TYPE,
        dst_model=DEFAULT_FOG_MODEL,
        dst_type=DEFAULT_FOG_TYPE,
        rssi_dbm=rssi_dbm,
    )


def fog_to_mobile_wifi_powers_w_from_rssi(rssi_dbm: float) -> Tuple[float, float]:
    return wireless_directional_powers_w_from_models(
        src_model=DEFAULT_FOG_MODEL,
        src_type=DEFAULT_FOG_TYPE,
        dst_model=DEFAULT_MOBILE_MODEL,
        dst_type=DEFAULT_EDGE_TYPE,
        rssi_dbm=rssi_dbm,
    )


def sensor_to_fog_wifi_powers_w_from_rssi(rssi_dbm: float) -> Tuple[float, float]:
    return wireless_directional_powers_w_from_models(
        src_model=DEFAULT_SENSOR_MODEL,
        src_type=DEFAULT_SENSOR_TYPE,
        dst_model=DEFAULT_FOG_MODEL,
        dst_type=DEFAULT_FOG_TYPE,
        rssi_dbm=rssi_dbm,
    )


def fog_to_sensor_wifi_powers_w_from_rssi(rssi_dbm: float) -> Tuple[float, float]:
    return wireless_directional_powers_w_from_models(
        src_model=DEFAULT_FOG_MODEL,
        src_type=DEFAULT_FOG_TYPE,
        dst_model=DEFAULT_SENSOR_MODEL,
        dst_type=DEFAULT_SENSOR_TYPE,
        rssi_dbm=rssi_dbm,
    )


def wifi_throughput_mbps_from_rssi(rssi_dbm: float) -> float:
    """
    Compatibility wrapper for existing callers.
    """
    return mobile_wifi_throughput_mbps_from_rssi(rssi_dbm)


def tcp_retransmission_rate_from_rssi(rssi_dbm: float) -> float:
    """
    Compatibility wrapper for existing callers.
    """
    return wireless_tcp_retransmission_rate_from_models(
        src_model=DEFAULT_MOBILE_MODEL,
        src_type=DEFAULT_EDGE_TYPE,
        dst_model=DEFAULT_FOG_MODEL,
        dst_type=DEFAULT_FOG_TYPE,
        rssi_dbm=rssi_dbm,
    )


def galaxy_s4_wifi_powers_w_from_rssi(rssi_dbm: float) -> Tuple[float, float]:
    """
    Compatibility wrapper for existing callers.
    """
    return _wifi_tx_rx_w(DEFAULT_MOBILE_MODEL, DEFAULT_EDGE_TYPE, rssi_dbm)


def energy_wh_per_mb(throughput_mb_s: float, tx_power_w: float, rx_power_w: float) -> float:
    tx_wh_per_mb, rx_wh_per_mb = transfer_energy_components_wh_per_mb(
        throughput_mb_s=throughput_mb_s,
        tx_power_w=tx_power_w,
        rx_power_w=rx_power_w,
    )
    return tx_wh_per_mb + rx_wh_per_mb


def transfer_energy_components_wh_per_mb(
    throughput_mb_s: float,
    tx_power_w: float,
    rx_power_w: float,
) -> Tuple[float, float]:
    safe_bw = max(throughput_mb_s, 1e-12)
    transfer_time_tu = 1.0 / safe_bw
    tx_wh_per_mb = watt_to_wpt(tx_power_w) * transfer_time_tu
    rx_wh_per_mb = watt_to_wpt(rx_power_w) * transfer_time_tu
    return tx_wh_per_mb, rx_wh_per_mb


def fixed_overhead_energy_wh(power_w: float, duration_s: float) -> float:
    return watt_to_wpt(power_w) * seconds_to_tu(duration_s)


def tail_energy_wh_per_transfer() -> float:
    return SAMSUNG_S4_MOBILE_DEVICE.wifi_tail_energy_wh()


def sensor_wifi_data_powers_w_from_rssi(rssi_dbm: float) -> Tuple[float, float, bool]:
    return ECG_SENSOR_DEVICE.sensor_wifi_data_powers_w_from_rssi(rssi_dbm)


def sensor_wifi_promotion_energy_wh_per_transfer() -> float:
    return ECG_SENSOR_DEVICE.sensor_wifi_promotion_energy_wh()


def sensor_wifi_tail_energy_wh_per_transfer() -> float:
    return ECG_SENSOR_DEVICE.sensor_wifi_tail_energy_wh()


def sensor_ble_data_bw_mb_s() -> float:
    return ECG_SENSOR_DEVICE.sensor_ble_data_bw_mb_s()


def sensor_ble_tx_power_w() -> float:
    return ECG_SENSOR_DEVICE.sensor_ble_tx_power_w()


def sensor_ble_rx_power_w() -> float:
    return ECG_SENSOR_DEVICE.sensor_ble_rx_power_w()


def sensor_ble_tail_energy_wh_per_transfer() -> float:
    return ECG_SENSOR_DEVICE.sensor_ble_tail_energy_wh()


def smartwatch_ble_tx_power_w() -> float:
    return LG_URBANE_SMARTWATCH_DEVICE.sensor_ble_tx_power_w()


def smartwatch_ble_rx_power_w() -> float:
    return LG_URBANE_SMARTWATCH_DEVICE.sensor_ble_rx_power_w()


def sensor_wifi_energy_wh_per_mb(
    rssi_dbm: float,
    throughput_mb_s: float,
) -> Tuple[float, float, float, float, float, float, bool]:
    tx_w, rx_w, available = sensor_wifi_data_powers_w_from_rssi(rssi_dbm)
    if not available:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False

    data_wh_per_mb = energy_wh_per_mb(throughput_mb_s, tx_w, rx_w)
    promotion_wh = sensor_wifi_promotion_energy_wh_per_transfer()
    tail_wh = sensor_wifi_tail_energy_wh_per_transfer()
    total_wh_per_mb = data_wh_per_mb + promotion_wh + tail_wh
    return total_wh_per_mb, data_wh_per_mb, promotion_wh, tail_wh, tx_w, rx_w, True


def sensor_ble_energy_wh_per_mb() -> Tuple[float, float, float]:
    data_wh_per_mb = energy_wh_per_mb(
        throughput_mb_s=sensor_ble_data_bw_mb_s(),
        tx_power_w=sensor_ble_tx_power_w(),
        rx_power_w=0.0,
    )
    tail_wh = sensor_ble_tail_energy_wh_per_transfer()
    total_wh_per_mb = data_wh_per_mb + tail_wh
    return total_wh_per_mb, data_wh_per_mb, tail_wh
