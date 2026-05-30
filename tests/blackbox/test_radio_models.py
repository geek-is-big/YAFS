from __future__ import annotations

from math import isclose

import pytest

from radio_models import (
    galaxy_s4_wifi_powers_w_from_rssi,
    sensor_wifi_data_powers_w_from_rssi,
    tcp_retransmission_rate_from_rssi,
    wifi_throughput_mbps_from_rssi,
)


@pytest.mark.fast_matrix
@pytest.mark.parametrize(
    "rssi_dbm,expected",
    [
        (-70.0, 54.0 / 8.0),
        (-85.0, 11.0 / 8.0),
        (-90.0, 1.0 / 8.0),
        (-87.5, 0.75),
    ],
)
def test_wifi_throughput_rssi_anchors_and_interpolation(rssi_dbm, expected):
    assert isclose(wifi_throughput_mbps_from_rssi(rssi_dbm), expected, abs_tol=1e-12)


@pytest.mark.fast_matrix
@pytest.mark.parametrize(
    "rssi_dbm,expected",
    [
        (-95.0, 0.9),
        (-90.0, 0.9),
        (-85.0, 0.55),
        (-80.0, 0.3),
        (-75.0, 0.25),
        (-70.0, 0.2),
        (-60.0, 0.0),
        (-55.0, 0.0),
    ],
)
def test_tcp_retransmission_rssi_anchors_and_interpolation(rssi_dbm, expected):
    assert isclose(tcp_retransmission_rate_from_rssi(rssi_dbm), expected, abs_tol=1e-12)


@pytest.mark.fast_matrix
@pytest.mark.parametrize(
    "rssi_dbm,expected_tx_w,expected_rx_w",
    [
        (-100.0, 0.671, 0.395),
        (-90.0, 0.671, 0.395),
        (-85.0, 0.892, 0.514),
        (-75.0, 1.066, 0.6125),
        (-50.0, 0.654, 0.451),
        (-40.0, 0.654, 0.451),
    ],
)
def test_mobile_wifi_power_rssi_clamp_and_interpolation(
    rssi_dbm, expected_tx_w, expected_rx_w
):
    tx_w, rx_w = galaxy_s4_wifi_powers_w_from_rssi(rssi_dbm)
    assert isclose(tx_w, expected_tx_w, abs_tol=1e-12)
    assert isclose(rx_w, expected_rx_w, abs_tol=1e-12)


@pytest.mark.fast_matrix
@pytest.mark.parametrize(
    "rssi_dbm,expected_tx_w,expected_rx_w",
    [
        (-100.0, 0.8407, 0.2523),
        (-65.0, 0.8407, 0.2523),
        (-60.0, 0.75675, 0.29765),
        (-55.0, 0.6728, 0.3430),
        (-42.0, 0.6691, 0.3785),
        (-30.0, 0.6691, 0.3785),
    ],
)
def test_sensor_wifi_power_is_clamped_and_available(
    rssi_dbm, expected_tx_w, expected_rx_w
):
    tx_w, rx_w, available = sensor_wifi_data_powers_w_from_rssi(rssi_dbm)
    assert available is True
    assert isclose(tx_w, expected_tx_w, abs_tol=1e-12)
    assert isclose(rx_w, expected_rx_w, abs_tol=1e-12)
