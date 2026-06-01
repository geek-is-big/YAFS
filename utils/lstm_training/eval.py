from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "utils.lstm_training"

from .config import RSSILSTMConfig, load_lstm_config
from .data import RouteWindowDataset, build_fixed_route_rssi_frame, sorted_rssi_columns
from .model import RSSILSTM
from .scaler import StandardScaler
from .train import _resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate LSTM RSSI forecasting on a fixed route.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("lstm.yaml"),
        help="Path to utils/lstm_training/lstm.yaml.",
    )
    parser.add_argument("--route", type=Path, required=True, help="Fixed evaluation route CSV.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_lstm_config(args.config)
    evaluate_lstm(config, args.route)


def evaluate_lstm(config: RSSILSTMConfig, route_csv_path: Path) -> Dict[str, object]:
    device = _resolve_device(config.training.device)
    scaler = StandardScaler.load(config.artifacts.scaler_path)
    checkpoint = torch.load(config.artifacts.model_path, map_location=device, weights_only=True)
    model_config = checkpoint["model_config"]
    model = RSSILSTM(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    fixed_frame = build_fixed_route_rssi_frame(route_csv_path, config.data.fog_csv)
    rssi_columns = list(checkpoint.get("rssi_columns") or sorted_rssi_columns(fixed_frame))
    dataset = RouteWindowDataset(
        fixed_frame,
        rssi_columns,
        history_len=config.lstm.history_len,
        pred_horizon=config.lstm.pred_horizon,
        use_delta=config.lstm.use_delta,
        scaler=scaler,
    )
    if len(dataset) == 0:
        raise ValueError("Fixed route is too short for the configured history_len and pred_horizon.")

    loader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=0)
    predictions = []
    targets = []
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).cpu().numpy()
            predictions.append(scaler.inverse_transform(pred))
            targets.append(scaler.inverse_transform(y.numpy()))
    pred_dbm = np.concatenate(predictions, axis=0)
    true_dbm = np.concatenate(targets, axis=0)
    metrics = _evaluation_metrics(pred_dbm, true_dbm, rssi_columns)

    _save_prediction_frame(config.results.eval_predictions_path, pred_dbm, true_dbm, rssi_columns)
    config.results.eval_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    config.results.eval_metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def _evaluation_metrics(pred_dbm: np.ndarray, true_dbm: np.ndarray, rssi_columns) -> Dict[str, object]:
    diff = pred_dbm - true_dbm
    mae_per_horizon = np.mean(np.abs(diff), axis=(0, 2))
    rmse_per_horizon = np.sqrt(np.mean(np.square(diff), axis=(0, 2)))
    mae_per_fog = np.mean(np.abs(diff), axis=(0, 1))
    return {
        "MAE_RSSI": float(np.mean(np.abs(diff))),
        "RMSE_RSSI": float(np.sqrt(np.mean(np.square(diff)))),
        "MAE_per_horizon_step": [float(value) for value in mae_per_horizon],
        "RMSE_per_horizon_step": [float(value) for value in rmse_per_horizon],
        "MAE_per_fog_node": {
            str(column).replace("RSSI_FOG_", ""): float(value)
            for column, value in zip(rssi_columns, mae_per_fog)
        },
    }


def _save_prediction_frame(path: Path, pred_dbm: np.ndarray, true_dbm: np.ndarray, rssi_columns) -> None:
    rows = []
    for window_idx in range(pred_dbm.shape[0]):
        for horizon_idx in range(pred_dbm.shape[1]):
            row = {"window_index": window_idx, "horizon_step": horizon_idx + 1}
            for fog_idx, column in enumerate(rssi_columns):
                fog_id = str(column).replace("RSSI_FOG_", "")
                row[f"pred_RSSI_FOG_{fog_id}"] = float(pred_dbm[window_idx, horizon_idx, fog_idx])
                row[f"true_RSSI_FOG_{fog_id}"] = float(true_dbm[window_idx, horizon_idx, fog_idx])
            rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    main()
