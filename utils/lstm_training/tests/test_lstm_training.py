from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from utils.lstm_training.data import (
    RouteWindowDataset,
    build_fixed_route_rssi_frame,
    build_lstm_input_from_route_window,
    sorted_rssi_columns,
)
from utils.lstm_training.inference import RSSIPredictor, predict_future_metrics
from utils.lstm_training.model import RSSILSTM
from utils.lstm_training.scaler import StandardScaler
from utils.lstm_training.train import _configure_torch_threads_for_device


REPO_ROOT = Path(__file__).resolve().parents[3]
FIXED_ROUTE = REPO_ROOT / "simulation" / "CityScenario" / "dataset" / "usersLocation-melbCBD_1.csv"
FOG_CSV = REPO_ROOT / "simulation" / "CityScenario" / "dataset" / "edgeResources-melbCBD.csv"


def _frame() -> pd.DataFrame:
    rows = []
    for route_id, offset in (("route_a", 0.0), ("route_b", 100.0)):
        for t in range(6):
            rows.append(
                {
                    "route_id": route_id,
                    "time": t,
                    "RSSI_FOG_10": offset + t,
                    "RSSI_FOG_2": offset + t + 10.0,
                }
            )
    return pd.DataFrame(rows)


def test_standard_scaler_round_trip(tmp_path: Path):
    values = np.array([[-80.0, -70.0], [-60.0, -50.0]], dtype=np.float32)
    scaler = StandardScaler.fit(values)
    restored = scaler.inverse_transform(scaler.transform(values))
    assert np.allclose(restored, values)

    path = tmp_path / "scaler.json"
    scaler.save(path)
    loaded = StandardScaler.load(path)
    assert loaded == scaler


def test_sorted_rssi_columns_use_numeric_fog_ids():
    assert sorted_rssi_columns(_frame()) == ["RSSI_FOG_2", "RSSI_FOG_10"]


def test_build_lstm_input_delta_channel():
    window = np.array([[1.0, 10.0], [3.0, 13.0], [2.0, 15.0]], dtype=np.float32)
    features = build_lstm_input_from_route_window(window, use_delta=True)
    assert features.shape == (3, 4)
    assert np.allclose(features[:, :2], window)
    assert np.allclose(features[:, 2:], [[0.0, 0.0], [2.0, 3.0], [-1.0, 2.0]])


def test_route_window_dataset_shapes_and_route_isolation():
    frame = _frame()
    columns = sorted_rssi_columns(frame)
    dataset = RouteWindowDataset(frame, columns, history_len=3, pred_horizon=2, use_delta=True)

    assert len(dataset) == 4
    x0, y0 = dataset[0]
    assert tuple(x0.shape) == (3, 4)
    assert tuple(y0.shape) == (2, 2)
    assert np.allclose(y0.numpy(), [[13.0, 3.0], [14.0, 4.0]])

    x_last, y_last = dataset[2]
    assert np.all(x_last.numpy()[:, 0] >= 110.0)
    assert np.all(y_last.numpy()[:, 0] >= 113.0)


def test_rssi_lstm_output_shape():
    model = RSSILSTM(
        num_fog_nodes=2,
        history_len=3,
        pred_horizon=4,
        use_delta=True,
        hidden_size_1=5,
        hidden_size_2=3,
        dropout=0.0,
    )
    output = model(torch.zeros(7, 3, 4))
    assert tuple(output.shape) == (7, 4, 2)


def test_configure_torch_threads_for_cpu(monkeypatch):
    calls = []
    monkeypatch.setattr("utils.lstm_training.train._physical_cpu_count", lambda: 4)
    monkeypatch.setattr("utils.lstm_training.train.torch.set_num_threads", calls.append)

    configured = _configure_torch_threads_for_device(torch.device("cpu"))

    assert configured == 4
    assert calls == [4]


def test_fixed_route_rssi_frame_has_finite_values():
    frame = build_fixed_route_rssi_frame(FIXED_ROUTE, FOG_CSV)
    columns = sorted_rssi_columns(frame)
    assert len(frame) == len(pd.read_csv(FIXED_ROUTE))
    assert len(columns) == len(pd.read_csv(FOG_CSV))
    assert np.isfinite(frame.loc[:, columns].to_numpy(dtype=float)).all()


def test_predictor_checkpoint_round_trip(tmp_path: Path):
    model = RSSILSTM(
        num_fog_nodes=2,
        history_len=3,
        pred_horizon=2,
        use_delta=True,
        hidden_size_1=4,
        hidden_size_2=3,
        dropout=0.0,
    )
    model_path = tmp_path / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "num_fog_nodes": 2,
                "history_len": 3,
                "pred_horizon": 2,
                "use_delta": True,
                "hidden_size_1": 4,
                "hidden_size_2": 3,
                "dropout": 0.0,
            },
            "rssi_columns": ["RSSI_FOG_0", "RSSI_FOG_1"],
        },
        model_path,
    )
    scaler_path = tmp_path / "scaler.json"
    StandardScaler.fit(np.array([[-90.0, -80.0], [-70.0, -60.0]], dtype=np.float32)).save(scaler_path)

    predictor = RSSIPredictor(str(model_path), str(scaler_path), {"device": "cpu"})
    prediction = predictor.predict(np.array([[-80.0, -70.0], [-79.0, -69.0], [-78.0, -68.0]], dtype=np.float32))
    assert prediction.shape == (2, 2)
    assert np.isfinite(prediction).all()


def test_predict_future_metrics_with_radio_model():
    class RadioModel:
        wifi_retransmission_rtt_s = 0.05

        def wifi_throughput_mbps_from_rssi(self, rssi_dbm):
            return 10.0 + rssi_dbm / 100.0

        def tcp_retransmission_rate_from_rssi(self, rssi_dbm):
            return 0.2 if rssi_dbm < -70.0 else 0.0

        def galaxy_s4_wifi_powers_w_from_rssi(self, rssi_dbm):
            return 0.7, 0.4

    metrics = predict_future_metrics(np.array([[-80.0, -60.0]], dtype=np.float32), RadioModel())
    assert set(metrics) == {"rtt_s", "throughput_mb_s", "tcp_retransmission_rate", "P_tx_w", "P_rx_w"}
    assert metrics["rtt_s"].shape == (1, 2)
    assert metrics["rtt_s"][0, 0] == pytest.approx(0.01)
    assert metrics["P_tx_w"][0, 1] == pytest.approx(0.7)
