from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import yaml


ROUTE_TYPES = (
    "static",
    "random_shortest_path",
    "multi_stop",
    "stop_and_go",
    "passing_by_fog",
    "handover",
    "u_turn",
)

DEFAULT_ROUTE_TYPE_WEIGHTS = {route_type: 1.0 for route_type in ROUTE_TYPES}


@dataclass(frozen=True)
class RouteGenerationConfig:
    num_train_routes: int
    num_val_routes: int
    route_duration_steps_min: int
    route_duration_steps_max: int
    step_seconds: float
    speed_mps_min: float
    speed_mps_max: float
    stop_probability: float
    stop_duration_min: int
    stop_duration_max: int
    route_type_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_ROUTE_TYPE_WEIGHTS))
    waypoint_count_min: int = 1
    waypoint_count_max: int = 3


@dataclass(frozen=True)
class RadioConfig:
    add_rssi_noise: bool
    rssi_noise_std_db: float
    random_seed: int = 12345


@dataclass(frozen=True)
class DatasetConfig:
    route_generation: RouteGenerationConfig
    radio: RadioConfig


def _require_mapping(payload: object, section_name: str) -> MutableMapping[str, object]:
    if not isinstance(payload, MutableMapping):
        raise ValueError(f"Config section '{section_name}' must be a mapping.")
    return payload


def _validate_route_type_weights(weights: Mapping[str, object]) -> Dict[str, float]:
    missing = [route_type for route_type in ROUTE_TYPES if route_type not in weights]
    extra = sorted(set(weights.keys()) - set(ROUTE_TYPES))
    if missing or extra:
        raise ValueError(
            "route_type_weights must contain exactly these keys: "
            f"{', '.join(ROUTE_TYPES)}. Missing={missing}, extra={extra}"
        )

    normalized: Dict[str, float] = {}
    total = 0.0
    for route_type in ROUTE_TYPES:
        value = float(weights[route_type])
        if value < 0.0:
            raise ValueError(f"route_type_weights['{route_type}'] must be >= 0.")
        normalized[route_type] = value
        total += value

    if total <= 0.0:
        raise ValueError("At least one route type weight must be > 0.")

    for route_type in ROUTE_TYPES:
        normalized[route_type] /= total
    return normalized


def _validate_route_generation(section: MutableMapping[str, object]) -> RouteGenerationConfig:
    weights = section.get("route_type_weights", DEFAULT_ROUTE_TYPE_WEIGHTS)
    weight_map = _require_mapping(weights, "route_generation.route_type_weights")

    config = RouteGenerationConfig(
        num_train_routes=int(section["num_train_routes"]),
        num_val_routes=int(section["num_val_routes"]),
        route_duration_steps_min=int(section["route_duration_steps_min"]),
        route_duration_steps_max=int(section["route_duration_steps_max"]),
        step_seconds=float(section["step_seconds"]),
        speed_mps_min=float(section["speed_mps_min"]),
        speed_mps_max=float(section["speed_mps_max"]),
        stop_probability=float(section["stop_probability"]),
        stop_duration_min=int(section["stop_duration_min"]),
        stop_duration_max=int(section["stop_duration_max"]),
        route_type_weights=_validate_route_type_weights(weight_map),
        waypoint_count_min=int(section.get("waypoint_count_min", 1)),
        waypoint_count_max=int(section.get("waypoint_count_max", 3)),
    )

    if config.num_train_routes < 0 or config.num_val_routes < 0:
        raise ValueError("num_train_routes and num_val_routes must be >= 0.")
    if config.route_duration_steps_min < 2:
        raise ValueError("route_duration_steps_min must be >= 2.")
    if config.route_duration_steps_max < config.route_duration_steps_min:
        raise ValueError("route_duration_steps_max must be >= route_duration_steps_min.")
    if config.step_seconds <= 0.0:
        raise ValueError("step_seconds must be > 0.")
    if config.speed_mps_min <= 0.0 or config.speed_mps_max < config.speed_mps_min:
        raise ValueError("speed_mps range must be positive and ordered.")
    if not 0.0 <= config.stop_probability <= 1.0:
        raise ValueError("stop_probability must be in [0, 1].")
    if config.stop_duration_min < 1 or config.stop_duration_max < config.stop_duration_min:
        raise ValueError("Stop duration bounds must be >= 1 and ordered.")
    if config.waypoint_count_min < 1 or config.waypoint_count_max < config.waypoint_count_min:
        raise ValueError("Waypoint count bounds must be >= 1 and ordered.")

    return config


def _validate_radio(section: MutableMapping[str, object]) -> RadioConfig:
    config = RadioConfig(
        add_rssi_noise=bool(section.get("add_rssi_noise", False)),
        rssi_noise_std_db=float(section.get("rssi_noise_std_db", 2.0)),
        random_seed=int(section.get("random_seed", 12345)),
    )
    if config.rssi_noise_std_db < 0.0:
        raise ValueError("rssi_noise_std_db must be >= 0.")
    return config


def load_dataset_config(config_path: Path) -> DatasetConfig:
    payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    root = _require_mapping(payload, "root")
    route_generation = _require_mapping(root.get("route_generation"), "route_generation")
    radio = _require_mapping(root.get("radio", {}), "radio")
    return DatasetConfig(
        route_generation=_validate_route_generation(route_generation),
        radio=_validate_radio(radio),
    )
