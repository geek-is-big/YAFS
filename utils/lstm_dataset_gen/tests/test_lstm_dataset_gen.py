from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from utils.lstm_dataset_gen.compat import haversine_distance_m, rssi_from_distance_dbm
from utils.lstm_dataset_gen.config import load_dataset_config
from utils.lstm_dataset_gen.generator import LSTMDatasetGenerator
from utils.lstm_dataset_gen.graph import load_fog_nodes, load_walkway_graph, parse_linestring_wkt


REPO_ROOT = Path(__file__).resolve().parents[3]
WALKWAY_CSV = REPO_ROOT / "utils" / "load_map" / "parallelogram_walkways.csv"
FOG_CSV = REPO_ROOT / "simulation" / "CityScenario" / "dataset" / "edgeResources-melbCBD.csv"


def _write_config(tmp_path: Path, add_noise: bool = False) -> Path:
    payload = {
        "route_generation": {
            "num_train_routes": 7,
            "num_val_routes": 7,
            "route_duration_steps_min": 120,
            "route_duration_steps_max": 120,
            "step_seconds": 1.0,
            "speed_mps_min": 1.0,
            "speed_mps_max": 1.0,
            "stop_probability": 0.10,
            "stop_duration_min": 3,
            "stop_duration_max": 8,
            "waypoint_count_min": 1,
            "waypoint_count_max": 2,
            "route_type_weights": {
                "static": 1.0,
                "random_shortest_path": 1.0,
                "multi_stop": 1.0,
                "stop_and_go": 1.0,
                "passing_by_fog": 1.0,
                "handover": 1.0,
                "u_turn": 1.0,
            },
        },
        "radio": {
            "add_rssi_noise": add_noise,
            "rssi_noise_std_db": 2.0,
            "random_seed": 20260601,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def test_parse_linestring_wkt_and_graph_loading():
    geometry = parse_linestring_wkt("LINESTRING (144.0 -37.0, 144.1 -37.1)")
    assert geometry == [(-37.0, 144.0), (-37.1, 144.1)]

    walkway = load_walkway_graph(WALKWAY_CSV)
    assert walkway.graph.number_of_nodes() > 0
    assert walkway.graph.number_of_edges() > 0


def test_load_fog_nodes_uses_all_levels():
    fog_nodes = load_fog_nodes(FOG_CSV)
    assert len(fog_nodes) == len(pd.read_csv(FOG_CSV))


def test_rssi_model_matches_reference_distance_behavior():
    near_rssi = rssi_from_distance_dbm(1.0)
    farther_rssi = rssi_from_distance_dbm(10.0)
    much_farther_rssi = rssi_from_distance_dbm(100.0)

    assert near_rssi > farther_rssi > much_farther_rssi


def test_noise_is_reproducible_with_fixed_seed(tmp_path: Path):
    config = load_dataset_config(_write_config(tmp_path, add_noise=True))
    generator_a = LSTMDatasetGenerator(config, seed=123)
    generator_b = LSTMDatasetGenerator(config, seed=123)

    datasets_a = generator_a.generate_datasets()
    datasets_b = generator_b.generate_datasets()

    assert np.allclose(
        datasets_a["routes_train_rssi"].filter(like="RSSI_FOG_").to_numpy(),
        datasets_b["routes_train_rssi"].filter(like="RSSI_FOG_").to_numpy(),
    )


def test_generate_all_route_types_and_rssi_outputs(tmp_path: Path):
    config = load_dataset_config(_write_config(tmp_path))
    generator = LSTMDatasetGenerator(config)
    datasets = generator.generate_datasets()

    routes_train = datasets["routes_train"]
    routes_train_rssi = datasets["routes_train_rssi"]

    assert set(routes_train["route_type"].unique()) == {
        "static",
        "random_shortest_path",
        "multi_stop",
        "stop_and_go",
        "passing_by_fog",
        "handover",
        "u_turn",
    }
    assert len(routes_train) == len(routes_train_rssi)
    assert not routes_train.isna().any().any()
    assert not routes_train_rssi.isna().any().any()

    for route_id, frame in routes_train.groupby("route_id"):
        positions = list(zip(frame["latitude"], frame["longitude"]))
        route_type = frame["route_type"].iloc[0]
        if route_type == "static":
            assert len(set(positions)) == 1
        if route_type == "stop_and_go":
            assert any(a == b for a, b in zip(positions, positions[1:]))
        if route_type == "passing_by_fog":
            target_fog_id = int(frame["target_fog_id"].iloc[0])
            rssi = routes_train_rssi[routes_train_rssi["route_id"] == route_id][f"RSSI_FOG_{target_fog_id}"].to_numpy()
            peak = int(np.argmax(rssi))
            assert 0 < peak < len(rssi) - 1
        if route_type == "handover":
            target_fog_id = int(frame["target_fog_id"].iloc[0])
            route_rssi = routes_train_rssi[routes_train_rssi["route_id"] == route_id]
            target_series = route_rssi[f"RSSI_FOG_{target_fog_id}"].to_numpy()
            dominant = route_rssi.filter(like="RSSI_FOG_").idxmax(axis=1)
            assert dominant.iloc[0] != dominant.iloc[-1]
            assert target_series[-1] > target_series[0]
        if route_type == "u_turn":
            target_fog_id = int(frame["target_fog_id"].iloc[0])
            rssi = routes_train_rssi[routes_train_rssi["route_id"] == route_id][f"RSSI_FOG_{target_fog_id}"].to_numpy()
            trough = int(np.argmin(rssi))
            assert 0 < trough < len(rssi) - 1


def test_smoke_cli_outputs_match_expected_shapes(tmp_path: Path):
    config = load_dataset_config(_write_config(tmp_path))
    generator = LSTMDatasetGenerator(config)
    output_dir = tmp_path / "generated"
    datasets = generator.generate_and_save(output_dir)

    assert (output_dir / "routes_train.csv").exists()
    assert (output_dir / "routes_val.csv").exists()
    assert (output_dir / "routes_train_rssi.csv").exists()
    assert (output_dir / "routes_val_rssi.csv").exists()
    assert len(datasets["routes_train"]) == len(datasets["routes_train_rssi"])


def test_generated_positions_do_not_make_large_jumps(tmp_path: Path):
    config = load_dataset_config(_write_config(tmp_path))
    generator = LSTMDatasetGenerator(config)
    routes = generator.generate_datasets()["routes_train"]
    max_speed = config.route_generation.speed_mps_max
    max_step_distance = max_speed * config.route_generation.step_seconds

    for _, frame in routes.groupby("route_id"):
        points = list(zip(frame["latitude"], frame["longitude"]))
        for start, end in zip(points, points[1:]):
            distance = haversine_distance_m(start[0], start[1], end[0], end[1])
            assert distance <= max_step_distance + 1.0
