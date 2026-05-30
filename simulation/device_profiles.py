from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from utils import ips_to_ipt, seconds_to_tu, watt_to_wpt


ROLE_CLOUD = "cloud"
ROLE_FOG = "fog"
ROLE_MOBILE = "mobile"
ROLE_SENSOR = "sensor"


@dataclass(frozen=True)
class BaseDeviceProfile:
    name: str

    def get_model(self, node_id: int) -> str:
        raise NotImplementedError

    def get_entity(self, node_id: int, **extra: object) -> Dict[str, object]:
        raise NotImplementedError


@dataclass(frozen=True)
class BaseSamsungS4WirelessDevice(BaseDeviceProfile):
    WIFI_MAX_BW_MBPS: float = 54.0 / 8.0
    WIFI_MEDIUM_BW_MBPS: float = 11.0 / 8.0
    WIFI_MIN_BW_MBPS: float = 1.0 / 8.0

    def model_prefix(self) -> str:
        raise NotImplementedError

    def default_node_type(self) -> str:
        raise NotImplementedError

    def default_ipt(self) -> float:
        raise NotImplementedError

    def default_ram(self) -> int:
        raise NotImplementedError

    def default_watt(self) -> float:
        raise NotImplementedError

    def get_model(self, node_id: int) -> str:
        return f"{self.model_prefix()}{int(node_id)}"

    def get_entity(self, node_id: int, **extra: object) -> Dict[str, object]:
        entity = {
            "id": int(node_id),
            "model": self.get_model(node_id),
            "type": self.default_node_type(),
            "IPT": self.default_ipt(),
            "RAM": self.default_ram(),
            "WATT": self.default_watt(),
        }
        entity.update(extra)
        return entity

    def wifi_throughput_mbps_from_rssi(self, rssi_dbm: float) -> float:
        if rssi_dbm >= -70.0:
            return ips_to_ipt(self.WIFI_MAX_BW_MBPS)
        if rssi_dbm >= -85.0:
            return ips_to_ipt(self.WIFI_MEDIUM_BW_MBPS)
        if rssi_dbm <= -90.0:
            return ips_to_ipt(self.WIFI_MIN_BW_MBPS)

        x = np.array([-90.0, -85.0], dtype=float)
        y = np.array([self.WIFI_MIN_BW_MBPS, self.WIFI_MEDIUM_BW_MBPS], dtype=float)
        return ips_to_ipt(float(np.interp(rssi_dbm, x, y)))

    def tcp_retransmission_rate_from_rssi(self, rssi_dbm: float) -> float:
        if rssi_dbm >= -60.0:
            return 0.0
        if rssi_dbm <= -90.0:
            return 0.9

        x = np.array([-90.0, -85.0, -80.0, -70.0, -60.0], dtype=float)
        y = np.array([0.9, 0.55, 0.3, 0.2, 0.0], dtype=float)
        return float(np.interp(rssi_dbm, x, y))

    def wifi_tx_rx_powers_w_from_rssi(self, rssi_dbm: float) -> Tuple[float, float]:
        raise NotImplementedError

    def sensor_wifi_data_powers_w_from_rssi(
        self, rssi_dbm: float
    ) -> Tuple[float, float, bool]:
        raise NotImplementedError

    def sensor_wifi_tail_energy_wh(self) -> float:
        raise NotImplementedError

    def sensor_wifi_promotion_energy_wh(self) -> float:
        raise NotImplementedError

    def sensor_ble_tail_energy_wh(self) -> float:
        raise NotImplementedError

    def sensor_ble_data_bw_mb_s(self) -> float:
        raise NotImplementedError

    def sensor_ble_tx_power_w(self) -> float:
        raise NotImplementedError

    def sensor_ble_rx_power_w(self) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class SamsungS4MobileDevice(BaseSamsungS4WirelessDevice):
    WIFI_TAIL_TIME_S: float = 0.210
    WIFI_TAIL_POWER_W: float = 0.289

    def model_prefix(self) -> str:
        return "samsung-s4-"

    def default_node_type(self) -> str:
        return "EDGE"

    def default_ipt(self) -> float:
        return ips_to_ipt(1.9 * 10**9)

    def default_ram(self) -> int:
        return 2000

    def default_watt(self) -> float:
        return watt_to_wpt(1.3)

    def wifi_tx_rx_powers_w_from_rssi(self, rssi_dbm: float) -> Tuple[float, float]:
        x = np.array([-90.0, -85.0, -80.0, -70.0, -60.0, -50.0], dtype=float)

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

    def wifi_tail_energy_wh(self) -> float:
        return watt_to_wpt(self.WIFI_TAIL_POWER_W) * seconds_to_tu(self.WIFI_TAIL_TIME_S)

    def sensor_wifi_data_powers_w_from_rssi(
        self, rssi_dbm: float
    ) -> Tuple[float, float, bool]:
        raise ValueError("Mobile profile does not provide sensor WiFi powers")

    def sensor_wifi_tail_energy_wh(self) -> float:
        raise ValueError("Mobile profile does not provide sensor WiFi tail energy")

    def sensor_wifi_promotion_energy_wh(self) -> float:
        raise ValueError("Mobile profile does not provide sensor WiFi promotion energy")

    def sensor_ble_tail_energy_wh(self) -> float:
        raise ValueError("Mobile profile does not provide sensor BLE tail energy")

    def sensor_ble_data_bw_mb_s(self) -> float:
        raise ValueError("Mobile profile does not provide sensor BLE data bandwidth")

    def sensor_ble_tx_power_w(self) -> float:
        raise ValueError("Mobile profile does not provide sensor BLE TX power")

    def sensor_ble_rx_power_w(self) -> float:
        raise ValueError("Mobile profile does not provide sensor BLE RX power")


@dataclass(frozen=True)
class LgUrbaneSmartwatchDevice(BaseSamsungS4WirelessDevice):
    SENSOR_WIFI_TAIL_TIME_S: float = 0.18
    SENSOR_WIFI_TAIL_POWER_W: float = 0.1212
    SENSOR_WIFI_PROMOTION_TIME_S: float = 0.30
    SENSOR_WIFI_PROMOTION_POWER_W: float = 0.2425

    SENSOR_BLE_TAIL_TIME_S: float = 4.77
    SENSOR_BLE_TAIL_POWER_W: float = 0.0341
    SENSOR_BLE_TX_POWER_W: float = 0.1807
    SENSOR_BLE_RX_POWER_W: float = 0.1749
    SENSOR_BLE_DATA_BW_MBPS: float = 0.305

    def model_prefix(self) -> str:
        return "lg-urbane-"

    def default_node_type(self) -> str:
        return "IOT"

    def default_ipt(self) -> float:
        return ips_to_ipt(768 * 10**6)

    def default_ram(self) -> int:
        return 512

    def default_watt(self) -> float:
        return watt_to_wpt(0.361)

    def wifi_tx_rx_powers_w_from_rssi(self, rssi_dbm: float) -> Tuple[float, float]:
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

        return tx / 1000.0, rx / 1000.0

    def sensor_wifi_data_powers_w_from_rssi(
        self, rssi_dbm: float
    ) -> Tuple[float, float, bool]:
        tx_w, rx_w = self.wifi_tx_rx_powers_w_from_rssi(rssi_dbm)
        return tx_w, rx_w, True

    def sensor_wifi_tail_energy_wh(self) -> float:
        return watt_to_wpt(self.SENSOR_WIFI_TAIL_POWER_W) * seconds_to_tu(
            self.SENSOR_WIFI_TAIL_TIME_S
        )

    def sensor_wifi_promotion_energy_wh(self) -> float:
        return watt_to_wpt(self.SENSOR_WIFI_PROMOTION_POWER_W) * seconds_to_tu(
            self.SENSOR_WIFI_PROMOTION_TIME_S
        )

    def sensor_ble_tail_energy_wh(self) -> float:
        return watt_to_wpt(self.SENSOR_BLE_TAIL_POWER_W) * seconds_to_tu(
            self.SENSOR_BLE_TAIL_TIME_S
        )

    def sensor_ble_data_bw_mb_s(self) -> float:
        return ips_to_ipt(self.SENSOR_BLE_DATA_BW_MBPS)

    def sensor_ble_tx_power_w(self) -> float:
        return float(self.SENSOR_BLE_TX_POWER_W)

    def sensor_ble_rx_power_w(self) -> float:
        return float(self.SENSOR_BLE_RX_POWER_W)


@dataclass(frozen=True)
class EcgSensorDevice(LgUrbaneSmartwatchDevice):
    SENSOR_BLE_TX_POWER_W: float = 0.1115
    SENSOR_BLE_RX_POWER_W: float = 0.1172

    def model_prefix(self) -> str:
        return "ecg-sensor-"

    def default_ram(self) -> int:
        return 256


SAMSUNG_S4_MOBILE_DEVICE = SamsungS4MobileDevice(name="samsung-s4-mobile")
LG_URBANE_SMARTWATCH_DEVICE = LgUrbaneSmartwatchDevice(name="lg-urbane")
ECG_SENSOR_DEVICE = EcgSensorDevice(name="ecg-sensor")


def resolve_device_profile(model: str, node_type: str) -> BaseSamsungS4WirelessDevice:
    normalized_model = str(model).lower()
    normalized_type = str(node_type).upper()

    if normalized_model.startswith(SAMSUNG_S4_MOBILE_DEVICE.model_prefix()):
        if normalized_type != "EDGE":
            raise ValueError(
                "Samsung S4 model must have EDGE type. "
                f"Got model={model!r}, type={node_type!r}"
            )
        return SAMSUNG_S4_MOBILE_DEVICE

    if normalized_model.startswith(LG_URBANE_SMARTWATCH_DEVICE.model_prefix()):
        if normalized_type != "IOT":
            raise ValueError(
                "LG Urbane model must have IOT type. "
                f"Got model={model!r}, type={node_type!r}"
            )
        return LG_URBANE_SMARTWATCH_DEVICE

    if normalized_model.startswith(ECG_SENSOR_DEVICE.model_prefix()):
        if normalized_type != "IOT":
            raise ValueError(
                "ECG sensor model must have IOT type. "
                f"Got model={model!r}, type={node_type!r}"
            )
        return ECG_SENSOR_DEVICE

    raise ValueError(f"Unsupported device model for profile resolution: {model}")


def infer_node_role(model: str, node_type: str) -> str:
    normalized_model = str(model).lower()
    normalized_type = str(node_type).upper()

    if normalized_model == "cloud-device" or normalized_type == "CLOUD":
        return ROLE_CLOUD
    if normalized_model in {"fog-device", "city-fog-device"} or normalized_type == "FOG":
        return ROLE_FOG
    if normalized_model.startswith(SAMSUNG_S4_MOBILE_DEVICE.model_prefix()):
        return ROLE_MOBILE
    if normalized_model.startswith(LG_URBANE_SMARTWATCH_DEVICE.model_prefix()):
        return ROLE_SENSOR
    if normalized_model.startswith(ECG_SENSOR_DEVICE.model_prefix()):
        return ROLE_SENSOR

    if normalized_model == "mobile-device":
        return ROLE_MOBILE
    if normalized_model in {"ecg-device", "smartwatch-device"}:
        return ROLE_SENSOR

    raise ValueError(f"Unsupported model/type combination: model={model!r}, type={node_type!r}")
