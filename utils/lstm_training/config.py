from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class LSTMConfig:
    history_len: int
    pred_horizon: int
    use_delta: bool
    hidden_size_1: int
    hidden_size_2: int
    dropout: float


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    early_stopping_patience: int
    seed: int
    device: str


@dataclass(frozen=True)
class DataConfig:
    train_rssi_csv: Path
    val_rssi_csv: Path
    fog_csv: Path


@dataclass(frozen=True)
class ArtifactConfig:
    model_path: Path
    scaler_path: Path
    train_metrics_path: Path
    val_metrics_path: Path


@dataclass(frozen=True)
class ResultConfig:
    eval_predictions_path: Path
    eval_metrics_path: Path


@dataclass(frozen=True)
class RSSILSTMConfig:
    lstm: LSTMConfig
    training: TrainingConfig
    data: DataConfig
    artifacts: ArtifactConfig
    results: ResultConfig


def _section(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"Config section '{name}' must be a mapping.")
    return value


def _path(repo_root: Path, value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return repo_root / path


def load_lstm_config(config_path: Path) -> RSSILSTMConfig:
    config_path = Path(config_path)
    repo_root = _find_repo_root(config_path.resolve())
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("LSTM config root must be a mapping.")

    lstm = _section(payload, "lstm")
    training = _section(payload, "training")
    data = _section(payload, "data")
    artifacts = _section(payload, "artifacts")
    results = _section(payload, "results")

    config = RSSILSTMConfig(
        lstm=LSTMConfig(
            history_len=int(lstm["history_len"]),
            pred_horizon=int(lstm["pred_horizon"]),
            use_delta=bool(lstm.get("use_delta", True)),
            hidden_size_1=int(lstm.get("hidden_size_1", 64)),
            hidden_size_2=int(lstm.get("hidden_size_2", 32)),
            dropout=float(lstm.get("dropout", 0.1)),
        ),
        training=TrainingConfig(
            batch_size=int(training.get("batch_size", 64)),
            epochs=int(training.get("epochs", 100)),
            learning_rate=float(training.get("learning_rate", 0.001)),
            early_stopping_patience=int(training.get("early_stopping_patience", 10)),
            seed=int(training.get("seed", 12345)),
            device=str(training.get("device", "auto")),
        ),
        data=DataConfig(
            train_rssi_csv=_path(repo_root, data["train_rssi_csv"]),
            val_rssi_csv=_path(repo_root, data["val_rssi_csv"]),
            fog_csv=_path(repo_root, data["fog_csv"]),
        ),
        artifacts=ArtifactConfig(
            model_path=_path(repo_root, artifacts["model_path"]),
            scaler_path=_path(repo_root, artifacts["scaler_path"]),
            train_metrics_path=_path(repo_root, artifacts["train_metrics_path"]),
            val_metrics_path=_path(repo_root, artifacts["val_metrics_path"]),
        ),
        results=ResultConfig(
            eval_predictions_path=_path(repo_root, results["eval_predictions_path"]),
            eval_metrics_path=_path(repo_root, results["eval_metrics_path"]),
        ),
    )
    _validate_config(config)
    return config


def _find_repo_root(config_path: Path) -> Path:
    for parent in (config_path.parent, *config_path.parents):
        if (parent / "simulation").is_dir() and (parent / "utils").is_dir():
            return parent
    return Path.cwd()


def _validate_config(config: RSSILSTMConfig) -> None:
    if config.lstm.history_len < 1:
        raise ValueError("lstm.history_len must be >= 1.")
    if config.lstm.pred_horizon < 1:
        raise ValueError("lstm.pred_horizon must be >= 1.")
    if config.lstm.hidden_size_1 < 1 or config.lstm.hidden_size_2 < 1:
        raise ValueError("LSTM hidden sizes must be >= 1.")
    if not 0.0 <= config.lstm.dropout < 1.0:
        raise ValueError("lstm.dropout must be in [0, 1).")
    if config.training.batch_size < 1:
        raise ValueError("training.batch_size must be >= 1.")
    if config.training.epochs < 1:
        raise ValueError("training.epochs must be >= 1.")
    if config.training.learning_rate <= 0.0:
        raise ValueError("training.learning_rate must be > 0.")
    if config.training.early_stopping_patience < 1:
        raise ValueError("training.early_stopping_patience must be >= 1.")
