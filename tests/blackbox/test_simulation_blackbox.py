from __future__ import annotations

import json
from math import isclose
from pathlib import Path
from typing import Dict

import pytest

import config as cfg

from tests.blackbox.harness import (
    ALL_CASES,
    FAST_CASES,
    BlackBoxCase,
    case_id,
    case_key,
    iter_case_pairs_for_processing_checks,
    iter_case_pairs_for_retransmission_checks,
    run_case,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = PROJECT_ROOT / "tests" / "blackbox" / "goldens" / "blackbox_v1.json"
ABS_TOL = 1e-9
PROCESSING_CASE_PAIRS = list(iter_case_pairs_for_processing_checks())
PROCESSING_CASE_IDS = [
    f"{edge_case.topology_name}:rssi{edge_case.rssi_dbm}"
    for edge_case, _ in PROCESSING_CASE_PAIRS
]
RETRANSMISSION_CASE_PAIRS = list(iter_case_pairs_for_retransmission_checks())
RETRANSMISSION_CASE_IDS = [
    f"{strong_case.topology_name}:{strong_case.processing_mode}"
    for strong_case, _ in RETRANSMISSION_CASE_PAIRS
]


@pytest.fixture(scope="session")
def golden_data() -> Dict[str, Dict[str, Dict[str, float | int]]]:
    if not GOLDEN_PATH.exists():
        raise AssertionError(
            "Golden file is missing. Generate it with:\n"
            "  /opt/anaconda3/envs/fog-simulation/bin/python "
            "tests/blackbox/generate_goldens.py"
        )
    return json.loads(GOLDEN_PATH.read_text())


def _assert_matches_golden(
    case: BlackBoxCase,
    actual_snapshot: Dict[str, float | int],
    golden_data: Dict[str, Dict[str, Dict[str, float | int]]],
) -> None:
    case_payload = golden_data["cases"][case_key(case)]
    exact = case_payload["exact"]
    approx = case_payload["approx"]

    for key, expected in exact.items():
        assert actual_snapshot[key] == expected, (
            f"Exact mismatch for {case_key(case)}:{key} "
            f"(actual={actual_snapshot[key]!r}, expected={expected!r})"
        )

    for key, expected in approx.items():
        actual = actual_snapshot[key]
        assert isclose(float(actual), float(expected), abs_tol=ABS_TOL), (
            f"Approx mismatch for {case_key(case)}:{key} "
            f"(actual={actual!r}, expected={expected!r}, tol={ABS_TOL})"
        )


def _assert_transport_behavior(
    case: BlackBoxCase,
    snapshot: Dict[str, float | int],
) -> None:
    assert snapshot["udp_attempts_min"] == 1
    assert snapshot["udp_attempts_max"] == 1
    assert snapshot["udp_rtr_max"] == 0.0
    assert snapshot["udp_size_sum"] > 0

    if case.rssi_dbm == -90:
        assert snapshot["tcp_attempts_max"] > 1
        assert snapshot["tcp_attempts_sum"] > snapshot["tcp_rows"]

    assert snapshot["total_link_tx_wh"] > 0.0
    assert snapshot["total_link_rx_wh"] > 0.0
    assert snapshot["total_link_tail_tx_wh"] >= 0.0
    assert snapshot["total_link_tail_rx_wh"] >= 0.0


@pytest.mark.fast_matrix
@pytest.mark.parametrize("case", FAST_CASES, ids=case_id)
def test_blackbox_fast_matrix(case, golden_data, tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "STATIC_LINK_RSSI_DBM", float(case.rssi_dbm))
    case_dir = tmp_path / "fast" / case_key(case).replace("|", "__")
    snapshot = run_case(case, output_dir=case_dir)
    _assert_matches_golden(case, snapshot, golden_data)
    _assert_transport_behavior(case, snapshot)


@pytest.mark.full_matrix
@pytest.mark.parametrize("case", ALL_CASES, ids=case_id)
def test_blackbox_full_matrix(case, golden_data, tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "STATIC_LINK_RSSI_DBM", float(case.rssi_dbm))
    case_dir = tmp_path / "full" / case_key(case).replace("|", "__")
    snapshot = run_case(case, output_dir=case_dir)
    _assert_matches_golden(case, snapshot, golden_data)
    _assert_transport_behavior(case, snapshot)


@pytest.mark.fast_matrix
@pytest.mark.parametrize(
    "edge_case,fog_case",
    PROCESSING_CASE_PAIRS,
    ids=PROCESSING_CASE_IDS,
)
def test_processing_energy_shifts_by_mode(edge_case, fog_case, tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "STATIC_LINK_RSSI_DBM", float(edge_case.rssi_dbm))
    edge_snapshot = run_case(
        edge_case,
        output_dir=tmp_path / "processing" / case_key(edge_case).replace("|", "__"),
    )

    monkeypatch.setattr(cfg, "STATIC_LINK_RSSI_DBM", float(fog_case.rssi_dbm))
    fog_snapshot = run_case(
        fog_case,
        output_dir=tmp_path / "processing" / case_key(fog_case).replace("|", "__"),
    )

    assert edge_snapshot["mobile_service_wh"] > edge_snapshot["fog_service_wh"]
    assert fog_snapshot["fog_service_wh"] > fog_snapshot["mobile_service_wh"]


@pytest.mark.fast_matrix
@pytest.mark.parametrize(
    "strong_case,weak_case",
    RETRANSMISSION_CASE_PAIRS,
    ids=RETRANSMISSION_CASE_IDS,
)
def test_tcp_retransmission_trends_with_rssi(
    strong_case,
    weak_case,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(cfg, "STATIC_LINK_RSSI_DBM", float(strong_case.rssi_dbm))
    strong_snapshot = run_case(
        strong_case,
        output_dir=tmp_path / "rssi" / case_key(strong_case).replace("|", "__"),
    )

    monkeypatch.setattr(cfg, "STATIC_LINK_RSSI_DBM", float(weak_case.rssi_dbm))
    weak_snapshot = run_case(
        weak_case,
        output_dir=tmp_path / "rssi" / case_key(weak_case).replace("|", "__"),
    )

    assert weak_snapshot["tcp_attempts_sum"] >= strong_snapshot["tcp_attempts_sum"]
    assert weak_snapshot["tcp_latency_sum"] >= strong_snapshot["tcp_latency_sum"]
    assert weak_snapshot["tcp_tx_wh_sum"] >= strong_snapshot["tcp_tx_wh_sum"]
