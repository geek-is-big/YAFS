from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from .data import build_lstm_input_from_route_window
from .model import RSSILSTM
from .scaler import StandardScaler


class RSSIPredictor:
    def __init__(self, model_path: str, scaler_path: str, config: dict) -> None:
        self.model_path = Path(model_path)
        self.scaler = StandardScaler.load(Path(scaler_path))
        self.config = dict(config)
        self.device = torch.device(self.config.get("device", "cpu"))
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        model_config = dict(checkpoint.get("model_config", {}))
        if not model_config:
            model_config = {
                "num_fog_nodes": int(self.config["num_fog_nodes"]),
                "history_len": int(self.config["history_len"]),
                "pred_horizon": int(self.config["pred_horizon"]),
                "use_delta": bool(self.config.get("use_delta", True)),
                "hidden_size_1": int(self.config.get("hidden_size_1", 64)),
                "hidden_size_2": int(self.config.get("hidden_size_2", 32)),
                "dropout": float(self.config.get("dropout", 0.1)),
            }
        self.model = RSSILSTM(**model_config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.history_len = int(model_config["history_len"])
        self.num_fog_nodes = int(model_config["num_fog_nodes"])
        self.use_delta = bool(model_config.get("use_delta", True))

    def predict(self, rssi_history: np.ndarray) -> np.ndarray:
        history = np.asarray(rssi_history, dtype=np.float32)
        expected_shape = (self.history_len, self.num_fog_nodes)
        if history.shape != expected_shape:
            raise ValueError(f"rssi_history must have shape {expected_shape}, got {history.shape}.")
        normalized = self.scaler.transform(history)
        model_input = build_lstm_input_from_route_window(normalized, self.use_delta)
        tensor = torch.from_numpy(model_input).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prediction = self.model(tensor).cpu().numpy()[0]
        return self.scaler.inverse_transform(prediction)


def predict_future_metrics(predicted_rssi: np.ndarray, radio_model: Optional[Any] = None) -> Dict[str, np.ndarray]:
    rssi = np.asarray(predicted_rssi, dtype=np.float32)
    if rssi.ndim != 2:
        raise ValueError("predicted_rssi must have shape (pred_horizon, num_fog_nodes).")
    radio = radio_model or _DefaultRadioModel()

    throughput = np.empty_like(rssi, dtype=np.float32)
    tcp_retx = np.empty_like(rssi, dtype=np.float32)
    rtt_s = np.empty_like(rssi, dtype=np.float32)
    tx_power = np.empty_like(rssi, dtype=np.float32)
    rx_power = np.empty_like(rssi, dtype=np.float32)

    for index, value in np.ndenumerate(rssi):
        rssi_dbm = float(value)
        throughput[index] = float(radio.wifi_throughput_mbps_from_rssi(rssi_dbm))
        tcp_retx[index] = float(radio.tcp_retransmission_rate_from_rssi(rssi_dbm))
        tx_w, rx_w = radio.galaxy_s4_wifi_powers_w_from_rssi(rssi_dbm)
        tx_power[index] = float(tx_w)
        rx_power[index] = float(rx_w)
        rtt_s[index] = float(getattr(radio, "wifi_retransmission_rtt_s", 0.03)) * tcp_retx[index]

    return {
        "rtt_s": rtt_s,
        "throughput_mb_s": throughput,
        "tcp_retransmission_rate": tcp_retx,
        "P_tx_w": tx_power,
        "P_rx_w": rx_power,
    }


class _DefaultRadioModel:
    def __init__(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        for path in (repo_root / "simulation", repo_root / "src", repo_root):
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
        import config as cfg
        from offloading import (
            galaxy_s4_wifi_powers_w_from_rssi,
            tcp_retransmission_rate_from_rssi,
            wifi_throughput_mbps_from_rssi,
        )

        self.wifi_retransmission_rtt_s = float(cfg.WIFI_RETRANSMISSION_RTT_S)
        self.wifi_throughput_mbps_from_rssi = wifi_throughput_mbps_from_rssi
        self.tcp_retransmission_rate_from_rssi = tcp_retransmission_rate_from_rssi
        self.galaxy_s4_wifi_powers_w_from_rssi = galaxy_s4_wifi_powers_w_from_rssi
