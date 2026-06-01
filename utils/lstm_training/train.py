from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "utils.lstm_training"

from .config import RSSILSTMConfig, load_lstm_config
from .data import RouteWindowDataset, fog_ids_from_columns, load_rssi_frame
from .model import RSSILSTM
from .scaler import StandardScaler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an LSTM model to forecast future RSSI.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("lstm.yaml"),
        help="Path to utils/lstm_training/lstm.yaml.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_lstm_config(args.config)
    train_lstm(config)


def train_lstm(config: RSSILSTMConfig) -> Dict[str, Dict[str, float]]:
    _seed_everything(config.training.seed)
    device = _resolve_device(config.training.device)

    train_frame, train_columns = load_rssi_frame(config.data.train_rssi_csv)
    val_frame, val_columns = load_rssi_frame(config.data.val_rssi_csv)
    if train_columns != val_columns:
        raise ValueError("Train and validation RSSI columns differ.")

    scaler = StandardScaler.fit(train_frame.loc[:, train_columns].to_numpy(dtype=np.float32))
    scaler.save(config.artifacts.scaler_path)

    train_dataset = RouteWindowDataset(
        train_frame,
        train_columns,
        history_len=config.lstm.history_len,
        pred_horizon=config.lstm.pred_horizon,
        use_delta=config.lstm.use_delta,
        scaler=scaler,
    )
    val_dataset = RouteWindowDataset(
        val_frame,
        train_columns,
        history_len=config.lstm.history_len,
        pred_horizon=config.lstm.pred_horizon,
        use_delta=config.lstm.use_delta,
        scaler=scaler,
    )
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Train and validation datasets must each contain at least one sliding window.")

    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=0)

    model = RSSILSTM(
        num_fog_nodes=len(train_columns),
        history_len=config.lstm.history_len,
        pred_horizon=config.lstm.pred_horizon,
        use_delta=config.lstm.use_delta,
        hidden_size_1=config.lstm.hidden_size_1,
        hidden_size_2=config.lstm.hidden_size_2,
        dropout=config.lstm.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_train_metrics: Dict[str, float] = {}
    best_val_metrics: Dict[str, float] = {}
    patience_left = config.training.early_stopping_patience

    for epoch in range(1, config.training.epochs + 1):
        epoch_started_at = time.perf_counter()
        train_started_at = time.perf_counter()
        train_metrics = _run_epoch(model, train_loader, criterion, optimizer, scaler, device)
        train_time_s = time.perf_counter() - train_started_at

        val_started_at = time.perf_counter()
        val_metrics = _run_epoch(model, val_loader, criterion, None, scaler, device)
        val_time_s = time.perf_counter() - val_started_at
        epoch_time_s = time.perf_counter() - epoch_started_at
        train_metrics = {**train_metrics, "train_time_s": train_time_s, "epoch_time_s": epoch_time_s}
        val_metrics = {**val_metrics, "val_time_s": val_time_s, "epoch_time_s": epoch_time_s}
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['mse_norm']:.6f} "
            f"val_loss={val_metrics['mse_norm']:.6f} "
            f"val_mae_dbm={val_metrics['mae_dbm']:.4f} "
            f"train_time_s={train_time_s:.2f} "
            f"epoch_time_s={epoch_time_s:.2f}"
        )

        if val_metrics["mse_norm"] < best_val_loss:
            best_val_loss = val_metrics["mse_norm"]
            best_train_metrics = {**train_metrics, "epoch": epoch}
            best_val_metrics = {**val_metrics, "epoch": epoch}
            patience_left = config.training.early_stopping_patience
            _save_checkpoint(config, model, train_columns)
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    _write_json(config.artifacts.train_metrics_path, best_train_metrics)
    _write_json(config.artifacts.val_metrics_path, best_val_metrics)
    return {"train": best_train_metrics, "val": best_val_metrics}


def _run_epoch(model, loader, criterion, optimizer, scaler: StandardScaler, device: torch.device) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    abs_sum = 0.0
    sq_sum = 0.0
    count = 0

    with torch.set_grad_enabled(training):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            prediction = model(x)
            loss = criterion(prediction, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_count = int(y.numel())
            loss_sum += float(loss.item()) * batch_count
            pred_dbm = scaler.inverse_transform(prediction.detach().cpu().numpy())
            true_dbm = scaler.inverse_transform(y.detach().cpu().numpy())
            diff = pred_dbm - true_dbm
            abs_sum += float(np.abs(diff).sum())
            sq_sum += float(np.square(diff).sum())
            count += batch_count

    return {
        "mse_norm": loss_sum / max(count, 1),
        "mae_dbm": abs_sum / max(count, 1),
        "rmse_dbm": float(np.sqrt(sq_sum / max(count, 1))),
        "num_values": count,
    }


def _save_checkpoint(config: RSSILSTMConfig, model: RSSILSTM, rssi_columns) -> None:
    config.artifacts.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "num_fog_nodes": model.num_fog_nodes,
                "history_len": model.history_len,
                "pred_horizon": model.pred_horizon,
                "use_delta": model.use_delta,
                "hidden_size_1": config.lstm.hidden_size_1,
                "hidden_size_2": config.lstm.hidden_size_2,
                "dropout": config.lstm.dropout,
            },
            "rssi_columns": list(rssi_columns),
            "fog_ids": fog_ids_from_columns(rssi_columns),
        },
        config.artifacts.model_path,
    )


def _write_json(path: Path, payload: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _resolve_device(device_name: str) -> torch.device:
    if device_name != "auto":
        device = torch.device(device_name)
        _configure_torch_threads_for_device(device)
        print(f"Using specified device: {device}")
        return device
    if torch.cuda.is_available():
        print("Auto-detected CUDA device. Using GPU for training.")
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Auto-detected Apple Silicon device. Using GPU for training.")
        return torch.device("mps")
    
    device = torch.device("cpu")
    _configure_torch_threads_for_device(device)
    print("No GPU detected. Using CPU for training.")
    return device


def _configure_torch_threads_for_device(device: torch.device) -> Optional[int]:
    if device.type != "cpu":
        return None
    thread_count = _physical_cpu_count()
    if thread_count is None:
        return None
    torch.set_num_threads(thread_count)
    print(f"Using {thread_count} CPU threads for PyTorch.")
    return thread_count


def _physical_cpu_count() -> Optional[int]:
    # if sys.platform == "darwin":
    #     return _physical_cpu_count_from_sysctl()

    count = _physical_cpu_count_from_psutil()
    if count is not None:
        return count

    if sys.platform.startswith("linux"):
        count = _physical_cpu_count_from_proc_cpuinfo()
        if count is not None:
            return count

    return _valid_cpu_count(os.cpu_count())


def _physical_cpu_count_from_psutil() -> Optional[int]:
    try:
        import psutil
    except Exception:
        return None
    try:
        count = psutil.cpu_count(logical=False)
    except Exception:
        return None
    return _valid_cpu_count(count)


def _physical_cpu_count_from_sysctl() -> Optional[int]:
    try:
        output = subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    return _valid_cpu_count(output.strip())


def _physical_cpu_count_from_proc_cpuinfo() -> Optional[int]:
    cpuinfo_path = Path("/proc/cpuinfo")
    try:
        text = cpuinfo_path.read_text(encoding="utf-8")
    except Exception:
        return None

    cores = set()
    for block in text.strip().split("\n\n"):
        physical_id = None
        core_id = None
        for line in block.splitlines():
            if ":" not in line:
                continue
            key, value = [part.strip() for part in line.split(":", 1)]
            if key == "physical id":
                physical_id = value
            elif key == "core id":
                core_id = value
        if physical_id is not None and core_id is not None:
            cores.add((physical_id, core_id))

    return _valid_cpu_count(len(cores))


def _valid_cpu_count(value) -> Optional[int]:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    if count < 1:
        return None
    return count


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
