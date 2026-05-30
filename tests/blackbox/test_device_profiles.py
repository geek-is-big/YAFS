from __future__ import annotations

from math import isclose

import pytest

from device_profiles import (
    ECG_SENSOR_DEVICE,
    LG_URBANE_SMARTWATCH_DEVICE,
    ROLE_MOBILE,
    ROLE_SENSOR,
    SAMSUNG_S4_MOBILE_DEVICE,
    infer_node_role,
    resolve_device_profile,
)
from radio_models import (
    wireless_directional_powers_w_from_models,
    wireless_wifi_bw_mbps_from_models,
)


@pytest.mark.fast_matrix
def test_resolver_accepts_samsung_s4_models():
    assert (
        resolve_device_profile("samsung-s4-2", "EDGE")
        is SAMSUNG_S4_MOBILE_DEVICE
    )
    assert (
        resolve_device_profile("lg-urbane-3", "IOT")
        is LG_URBANE_SMARTWATCH_DEVICE
    )
    assert (
        resolve_device_profile("ecg-sensor-4", "IOT")
        is ECG_SENSOR_DEVICE
    )


@pytest.mark.fast_matrix
def test_resolver_raises_for_unknown_model():
    with pytest.raises(ValueError):
        resolve_device_profile("unknown-model-1", "EDGE")


@pytest.mark.fast_matrix
def test_get_entity_schema_for_samsung_s4_mobile():
    entity = SAMSUNG_S4_MOBILE_DEVICE.get_entity(
        node_id=9,
        DETAILS="x",
    )
    assert entity["id"] == 9
    assert entity["model"] == "samsung-s4-9"
    assert entity["type"] == "EDGE"
    assert isclose(entity["IPT"], SAMSUNG_S4_MOBILE_DEVICE.default_ipt(), abs_tol=1e-12)
    assert entity["RAM"] == SAMSUNG_S4_MOBILE_DEVICE.default_ram()
    assert isclose(entity["WATT"], SAMSUNG_S4_MOBILE_DEVICE.default_watt(), abs_tol=1e-12)
    assert entity["DETAILS"] == "x"


@pytest.mark.fast_matrix
def test_get_entity_schema_for_lg_urbane_sensor():
    entity = LG_URBANE_SMARTWATCH_DEVICE.get_entity(node_id=10)
    assert entity["id"] == 10
    assert entity["model"] == "lg-urbane-10"
    assert entity["type"] == "IOT"
    assert isclose(entity["IPT"], LG_URBANE_SMARTWATCH_DEVICE.default_ipt(), abs_tol=1e-12)
    assert entity["RAM"] == LG_URBANE_SMARTWATCH_DEVICE.default_ram()
    assert isclose(entity["WATT"], LG_URBANE_SMARTWATCH_DEVICE.default_watt(), abs_tol=1e-12)


@pytest.mark.fast_matrix
def test_get_entity_schema_for_ecg_sensor():
    entity = ECG_SENSOR_DEVICE.get_entity(node_id=11)
    assert entity["id"] == 11
    assert entity["model"] == "ecg-sensor-11"
    assert entity["type"] == "IOT"
    assert entity["RAM"] == 256


@pytest.mark.fast_matrix
def test_wireless_bw_is_min_of_endpoint_caps():
    bw = wireless_wifi_bw_mbps_from_models(
        src_model="samsung-s4-2",
        src_type="EDGE",
        dst_model="fog-device",
        dst_type="FOG",
        rssi_dbm=-85.0,
    )
    expected = SAMSUNG_S4_MOBILE_DEVICE.wifi_throughput_mbps_from_rssi(-85.0)
    assert isclose(bw, expected, abs_tol=1e-12)


@pytest.mark.fast_matrix
def test_directional_power_mapping_mobile_fog():
    mobile_tx_w, fog_rx_w = wireless_directional_powers_w_from_models(
        src_model="samsung-s4-2",
        src_type="EDGE",
        dst_model="fog-device",
        dst_type="FOG",
        rssi_dbm=-80.0,
    )
    fog_tx_w, mobile_rx_w = wireless_directional_powers_w_from_models(
        src_model="fog-device",
        src_type="FOG",
        dst_model="samsung-s4-2",
        dst_type="EDGE",
        rssi_dbm=-80.0,
    )
    assert isclose(mobile_tx_w, 1.113, abs_tol=1e-12)
    assert isclose(fog_rx_w, 3.7, abs_tol=1e-12)
    assert isclose(fog_tx_w, 4.9, abs_tol=1e-12)
    assert isclose(mobile_rx_w, 0.633, abs_tol=1e-12)


@pytest.mark.fast_matrix
def test_directional_power_mapping_sensor_fog():
    sensor_tx_w, fog_rx_w = wireless_directional_powers_w_from_models(
        src_model="lg-urbane-3",
        src_type="IOT",
        dst_model="fog-device",
        dst_type="FOG",
        rssi_dbm=-60.0,
    )
    fog_tx_w, sensor_rx_w = wireless_directional_powers_w_from_models(
        src_model="fog-device",
        src_type="FOG",
        dst_model="lg-urbane-3",
        dst_type="IOT",
        rssi_dbm=-60.0,
    )
    assert isclose(sensor_tx_w, 0.75675, abs_tol=1e-12)
    assert isclose(fog_rx_w, 3.7, abs_tol=1e-12)
    assert isclose(fog_tx_w, 4.9, abs_tol=1e-12)
    assert isclose(sensor_rx_w, 0.29765, abs_tol=1e-12)


@pytest.mark.fast_matrix
def test_infer_node_role_for_samsung_models():
    assert infer_node_role("samsung-s4-2", "EDGE") == ROLE_MOBILE
    assert infer_node_role("lg-urbane-3", "IOT") == ROLE_SENSOR
    assert infer_node_role("ecg-sensor-4", "IOT") == ROLE_SENSOR
