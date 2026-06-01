from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from .compat import haversine_distance_m, rssi_from_distance_dbm, rssi_model_parameters
from .config import DatasetConfig, ROUTE_TYPES
from .graph import FogNode, LatLon, WalkwayGraph, load_fog_nodes, load_walkway_graph


MAX_ROUTE_ATTEMPTS = 250
TARGET_LENGTH_FACTOR = 1.20
FOG_TARGET_LENGTH_FACTOR = 1.10
_PRECOMPUTED_CACHE: Dict[Tuple[str, str], Dict[str, object]] = {}


@dataclass
class RouteSample:
    route_type: str
    positions: List[LatLon]
    target_fog_id: Optional[int]


class LSTMDatasetGenerator:
    def __init__(
        self,
        config: DatasetConfig,
        walkway_csv_path: Optional[Path] = None,
        fog_csv_path: Optional[Path] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.config = config
        repo_root = Path(__file__).resolve().parents[2]
        default_walkway = repo_root / "utils" / "load_map" / "parallelogram_walkways.csv"
        default_fog = repo_root / "simulation" / "CityScenario" / "dataset" / "edgeResources-melbCBD.csv"

        walkway_path = str((walkway_csv_path or default_walkway).resolve())
        fog_path = str((fog_csv_path or default_fog).resolve())
        cached = self._get_precomputed_assets(walkway_path, fog_path)
        self.walkway = cached["walkway"]
        self.fog_nodes = cached["fog_nodes"]
        random_seed = self.config.radio.random_seed if seed is None else int(seed)
        self.rng = random.Random(random_seed)
        self.noise_rng = np.random.default_rng(random_seed)
        self.node_ids = cached["node_ids"]
        self.node_index_by_id = cached["node_index_by_id"]
        self.node_latitudes = cached["node_latitudes"]
        self.node_longitudes = cached["node_longitudes"]
        self.fog_ids = cached["fog_ids"]
        self.fog_latitudes = cached["fog_latitudes"]
        self.fog_longitudes = cached["fog_longitudes"]
        self.fogs_by_id = cached["fogs_by_id"]
        self._fog_node_distances = cached["_fog_node_distances"]
        self._fog_sorted_node_ids = cached["_fog_sorted_node_ids"]
        self.fog_nearest_node = cached["fog_nearest_node"]
        self.fog_pairs = cached["fog_pairs"]
        self._rssi_reference_distance_m, self._rssi_reference_dbm, self._rssi_environment_coeff = (
            rssi_model_parameters()
        )

    @classmethod
    def _get_precomputed_assets(cls, walkway_path: str, fog_path: str) -> Dict[str, object]:
        cache_key = (walkway_path, fog_path)
        if cache_key in _PRECOMPUTED_CACHE:
            return _PRECOMPUTED_CACHE[cache_key]

        walkway = load_walkway_graph(Path(walkway_path))
        fog_nodes = load_fog_nodes(Path(fog_path))
        node_ids = list(walkway.graph.nodes())
        node_index_by_id = {node_id: index for index, node_id in enumerate(node_ids)}
        node_latitudes = np.asarray([walkway.node_positions[node_id][0] for node_id in node_ids])
        node_longitudes = np.asarray([walkway.node_positions[node_id][1] for node_id in node_ids])
        fogs_by_id = {fog.fog_id: fog for fog in fog_nodes}
        fog_ids = np.asarray([fog.fog_id for fog in fog_nodes], dtype=int)
        fog_latitudes = np.asarray([fog.latitude for fog in fog_nodes], dtype=float)
        fog_longitudes = np.asarray([fog.longitude for fog in fog_nodes], dtype=float)

        temp = cls.__new__(cls)
        temp.walkway = walkway
        temp.fog_nodes = fog_nodes
        temp.node_ids = node_ids
        temp.node_latitudes = node_latitudes
        temp.node_longitudes = node_longitudes
        temp.fog_ids = fog_ids
        temp.fog_latitudes = fog_latitudes
        temp.fog_longitudes = fog_longitudes

        fog_node_distances = {fog.fog_id: temp._vectorized_distances_to_fog(fog) for fog in fog_nodes}
        fog_sorted_node_ids = {
            fog.fog_id: [node_ids[index] for index in np.argsort(fog_node_distances[fog.fog_id])]
            for fog in fog_nodes
        }

        temp.fogs_by_id = fogs_by_id
        fog_nearest_node = {fog.fog_id: fog_sorted_node_ids[fog.fog_id][0] for fog in fog_nodes}
        fog_pairs = temp._build_fog_pairs()

        cached = {
            "walkway": walkway,
            "fog_nodes": fog_nodes,
            "node_ids": node_ids,
            "node_index_by_id": node_index_by_id,
            "node_latitudes": node_latitudes,
            "node_longitudes": node_longitudes,
            "fog_ids": fog_ids,
            "fog_latitudes": fog_latitudes,
            "fog_longitudes": fog_longitudes,
            "fogs_by_id": fogs_by_id,
            "_fog_node_distances": fog_node_distances,
            "_fog_sorted_node_ids": fog_sorted_node_ids,
            "fog_nearest_node": fog_nearest_node,
            "fog_pairs": fog_pairs,
        }
        _PRECOMPUTED_CACHE[cache_key] = cached
        return cached

    def generate_and_save(self, output_dir: Path) -> Dict[str, pd.DataFrame]:
        output_dir.mkdir(parents=True, exist_ok=True)
        datasets = self.generate_datasets()
        datasets["routes_train"].to_csv(output_dir / "routes_train.csv", index=False)
        datasets["routes_val"].to_csv(output_dir / "routes_val.csv", index=False)
        datasets["routes_train_rssi"].to_csv(output_dir / "routes_train_rssi.csv", index=False)
        datasets["routes_val_rssi"].to_csv(output_dir / "routes_val_rssi.csv", index=False)
        return datasets

    def generate_datasets(self) -> Dict[str, pd.DataFrame]:
        train_routes = self._generate_split("train", self.config.route_generation.num_train_routes)
        val_routes = self._generate_split("val", self.config.route_generation.num_val_routes)
        return {
            "routes_train": self._routes_to_frame(train_routes),
            "routes_val": self._routes_to_frame(val_routes),
            "routes_train_rssi": self._routes_to_rssi_frame(train_routes),
            "routes_val_rssi": self._routes_to_rssi_frame(val_routes),
        }

    def _generate_split(self, split_name: str, route_count: int) -> List[Tuple[str, RouteSample]]:
        route_types = self._allocate_route_types(route_count)
        self.rng.shuffle(route_types)
        samples: List[Tuple[str, RouteSample]] = []
        for index, route_type in enumerate(route_types, start=1):
            route_id = f"{split_name}_{index:04d}"
            duration_steps = self.rng.randint(
                self.config.route_generation.route_duration_steps_min,
                self.config.route_generation.route_duration_steps_max,
            )
            speed_mps = self.rng.uniform(
                self.config.route_generation.speed_mps_min,
                self.config.route_generation.speed_mps_max,
            )
            sample = self._generate_route(route_type, duration_steps, speed_mps)
            samples.append((route_id, sample))
        return samples

    def _allocate_route_types(self, route_count: int) -> List[str]:
        if route_count <= 0:
            return []
        weights = self.config.route_generation.route_type_weights
        raw = [(route_type, weights[route_type] * route_count) for route_type in ROUTE_TYPES]
        base_counts = {route_type: int(math.floor(value)) for route_type, value in raw}
        remainder = route_count - sum(base_counts.values())
        fractions = sorted(
            ((value - base_counts[route_type], route_type) for route_type, value in raw),
            reverse=True,
        )
        for _, route_type in fractions[:remainder]:
            base_counts[route_type] += 1

        allocated: List[str] = []
        for route_type in ROUTE_TYPES:
            allocated.extend([route_type] * base_counts[route_type])
        return allocated

    def _generate_route(self, route_type: str, duration_steps: int, speed_mps: float) -> RouteSample:
        builders = {
            "static": self._build_static_route,
            "random_shortest_path": self._build_random_shortest_path_route,
            "multi_stop": self._build_multi_stop_route,
            "stop_and_go": self._build_stop_and_go_route,
            "passing_by_fog": self._build_passing_by_fog_route,
            "handover": self._build_handover_route,
            "u_turn": self._build_u_turn_route,
        }
        try:
            return builders[route_type](duration_steps, speed_mps)
        except KeyError as exc:
            raise ValueError(f"Unsupported route type: {route_type}") from exc

    def _build_static_route(self, duration_steps: int, speed_mps: float) -> RouteSample:
        del speed_mps
        u, v = self.rng.choice(self.walkway.edge_list)
        point = self._sample_point_on_edge(u, v)
        return RouteSample(route_type="static", positions=[point] * duration_steps, target_fog_id=None)

    def _build_random_shortest_path_route(self, duration_steps: int, speed_mps: float) -> RouteSample:
        required_distance = self._required_distance_m(duration_steps, speed_mps)
        for _ in range(MAX_ROUTE_ATTEMPTS):
            start = self.rng.choice(self.node_ids)
            path = self._pick_path_from_source(start, required_distance, TARGET_LENGTH_FACTOR)
            if path is None:
                continue
            route = self._sample_route_from_path(
                route_type="random_shortest_path",
                path=path,
                duration_steps=duration_steps,
                speed_mps=speed_mps,
                max_factor=TARGET_LENGTH_FACTOR,
            )
            if route is not None and self.walkway.path_length_m(path) >= required_distance:
                return route
        raise RuntimeError("Unable to generate a random_shortest_path route after repeated attempts.")

    def _build_multi_stop_route(self, duration_steps: int, speed_mps: float) -> RouteSample:
        waypoint_count = self.rng.randint(
            self.config.route_generation.waypoint_count_min,
            self.config.route_generation.waypoint_count_max,
        )
        segment_count = waypoint_count + 1
        required_distance = self._required_distance_m(duration_steps, speed_mps)
        segment_target = max(required_distance / segment_count, 10.0)
        for _ in range(MAX_ROUTE_ATTEMPTS):
            current = self.rng.choice(self.node_ids)
            node_paths: List[List[int]] = []
            for _segment_index in range(segment_count):
                segment_path = self._pick_path_from_source(current, segment_target, 1.45)
                if segment_path is None:
                    node_paths = []
                    break
                node_paths.append(segment_path)
                current = segment_path[-1]
            if not node_paths:
                continue
            combined_path = self._combine_node_paths(node_paths)
            route = self._sample_route_from_path(
                route_type="multi_stop",
                path=combined_path,
                duration_steps=duration_steps,
                speed_mps=speed_mps,
                max_factor=TARGET_LENGTH_FACTOR,
            )
            if route is not None:
                return route
        raise RuntimeError("Unable to generate a multi_stop route after repeated attempts.")

    def _build_stop_and_go_route(self, duration_steps: int, speed_mps: float) -> RouteSample:
        minimum_stop_budget = min(duration_steps - 2, self.config.route_generation.stop_duration_min)
        heuristic_budget = max(
            minimum_stop_budget,
            int(round(duration_steps * max(self.config.route_generation.stop_probability, 0.08))),
        )
        stop_budget = min(duration_steps - 2, heuristic_budget + self.rng.randint(0, max(1, heuristic_budget)))
        moving_steps = max(2, duration_steps - stop_budget)

        base_route = self._build_random_shortest_path_route(moving_steps, speed_mps)
        positions = self._insert_stops(base_route.positions, duration_steps)
        return RouteSample(route_type="stop_and_go", positions=positions, target_fog_id=None)

    def _build_passing_by_fog_route(self, duration_steps: int, speed_mps: float) -> RouteSample:
        required_distance = self._required_distance_m(duration_steps, speed_mps)
        minimum_distance_m = max(20.0, required_distance * 0.2)
        for _ in range(MAX_ROUTE_ATTEMPTS):
            fog = self.rng.choice(self.fog_nodes)
            pivot = self.fog_nearest_node[fog.fog_id]
            lengths, paths = nx.single_source_dijkstra(
                self.walkway.graph,
                pivot,
                cutoff=required_distance * FOG_TARGET_LENGTH_FACTOR,
                weight="length_m",
            )
            branch_candidates: Dict[int, List[Tuple[float, int, List[int]]]] = {}
            for node_id, distance in lengths.items():
                if node_id == pivot or distance < max(required_distance * 0.25, minimum_distance_m):
                    continue
                if distance > required_distance * 0.75:
                    continue
                path = paths[node_id]
                if len(path) < 2:
                    continue
                branch = path[1]
                branch_candidates.setdefault(branch, []).append((distance, node_id, path))

            if len(branch_candidates) < 2:
                continue
            branch_keys = list(branch_candidates.keys())
            start_branch, end_branch = self.rng.sample(branch_keys, 2)
            start_distance, _, start_path = self.rng.choice(branch_candidates[start_branch])
            end_distance, _, end_path = self.rng.choice(branch_candidates[end_branch])
            total_length = start_distance + end_distance
            if not required_distance <= total_length <= required_distance * FOG_TARGET_LENGTH_FACTOR:
                continue
            combined_path = list(reversed(start_path)) + end_path[1:]
            route = self._sample_route_from_path(
                route_type="passing_by_fog",
                path=combined_path,
                duration_steps=duration_steps,
                speed_mps=speed_mps,
                max_factor=FOG_TARGET_LENGTH_FACTOR,
                target_fog_id=fog.fog_id,
            )
            if route is not None and self._validate_passing_by(route.positions, fog):
                return route
        raise RuntimeError("Unable to generate a passing_by_fog route after repeated attempts.")

    def _build_handover_route(self, duration_steps: int, speed_mps: float) -> RouteSample:
        required_distance = self._required_distance_m(duration_steps, speed_mps)
        candidate_pairs = [
            pair
            for pair in self.fog_pairs
            if required_distance * 0.15 <= pair[0] <= required_distance * 0.9 + 120.0
        ]
        pairs = candidate_pairs or self.fog_pairs
        for _ in range(MAX_ROUTE_ATTEMPTS):
            _, fog_a_id, fog_b_id = self.rng.choice(pairs)
            fog_a = self.fogs_by_id[fog_a_id]
            fog_b = self.fogs_by_id[fog_b_id]
            start = self.fog_nearest_node[fog_a_id]
            end = self.fog_nearest_node[fog_b_id]
            if start == end:
                continue
            path = nx.shortest_path(self.walkway.graph, start, end, weight="length_m")
            route = self._sample_route_from_path(
                route_type="handover",
                path=path,
                duration_steps=duration_steps,
                speed_mps=speed_mps,
                max_factor=FOG_TARGET_LENGTH_FACTOR,
                target_fog_id=fog_b_id,
            )
            if route is None:
                continue
            if self._validate_handover(route.positions, fog_a, fog_b):
                return route
        raise RuntimeError("Unable to generate a handover route after repeated attempts.")

    def _build_u_turn_route(self, duration_steps: int, speed_mps: float) -> RouteSample:
        required_distance = self._required_distance_m(duration_steps, speed_mps)
        lower_outbound = required_distance / 2.0
        upper_outbound = required_distance / 2.0 * FOG_TARGET_LENGTH_FACTOR
        minimum_distance_m = max(15.0, lower_outbound * 0.6)
        for _ in range(MAX_ROUTE_ATTEMPTS):
            fog = self.rng.choice(self.fog_nodes)
            start = self.fog_nearest_node[fog.fog_id]
            lengths, paths = nx.single_source_dijkstra(
                self.walkway.graph,
                start,
                cutoff=upper_outbound,
                weight="length_m",
            )
            candidates = [
                node_id
                for node_id, distance in lengths.items()
                if node_id != start and lower_outbound <= distance <= upper_outbound
                and self._fog_node_distances[fog.fog_id][self.node_index_by_id[node_id]] >= minimum_distance_m
            ]
            if not candidates:
                continue
            far = self.rng.choice(candidates)
            outbound = paths[far]
            full_path = outbound + list(reversed(outbound[:-1]))
            route = self._sample_route_from_path(
                route_type="u_turn",
                path=full_path,
                duration_steps=duration_steps,
                speed_mps=speed_mps,
                max_factor=FOG_TARGET_LENGTH_FACTOR,
                target_fog_id=fog.fog_id,
            )
            if route is not None and self._validate_u_turn(route.positions, fog):
                return route
        raise RuntimeError("Unable to generate a u_turn route after repeated attempts.")

    def _sample_route_from_path(
        self,
        route_type: str,
        path: Sequence[int],
        duration_steps: int,
        speed_mps: float,
        max_factor: float,
        target_fog_id: Optional[int] = None,
    ) -> Optional[RouteSample]:
        polyline = self.walkway.polyline_for_path(path)
        total_length = self._polyline_length_m(polyline)
        required_distance = self._required_distance_m(duration_steps, speed_mps)
        if total_length < required_distance:
            return None
        if total_length > required_distance * max_factor:
            return None
        positions = self._positions_from_polyline(polyline, duration_steps, speed_mps)
        return RouteSample(route_type=route_type, positions=positions, target_fog_id=target_fog_id)

    def _combine_shortest_paths(self, node_sequence: Sequence[int]) -> List[int]:
        combined: List[int] = []
        for start, end in zip(node_sequence, node_sequence[1:]):
            segment = nx.shortest_path(self.walkway.graph, start, end, weight="length_m")
            if not combined:
                combined.extend(segment)
            else:
                combined.extend(segment[1:])
        return combined

    def _combine_node_paths(self, node_paths: Sequence[Sequence[int]]) -> List[int]:
        combined: List[int] = []
        for path in node_paths:
            if not combined:
                combined.extend(path)
            else:
                combined.extend(path[1:])
        return combined

    def _positions_from_polyline(self, polyline: Sequence[LatLon], duration_steps: int, speed_mps: float) -> List[LatLon]:
        if duration_steps == 1:
            return [polyline[0]]

        step_distance_m = speed_mps * self.config.route_generation.step_seconds
        segment_lengths = [0.0]
        for start, end in zip(polyline, polyline[1:]):
            segment_lengths.append(segment_lengths[-1] + haversine_distance_m(start[0], start[1], end[0], end[1]))

        positions: List[LatLon] = []
        for step_index in range(duration_steps):
            distance_mark = step_index * step_distance_m
            positions.append(self._point_at_distance(polyline, segment_lengths, distance_mark))
        return positions

    def _point_at_distance(
        self, polyline: Sequence[LatLon], cumulative_lengths: Sequence[float], distance_mark: float
    ) -> LatLon:
        if distance_mark <= 0.0:
            return polyline[0]
        if distance_mark >= cumulative_lengths[-1]:
            return polyline[-1]

        for index in range(1, len(polyline)):
            if cumulative_lengths[index] >= distance_mark:
                start = polyline[index - 1]
                end = polyline[index]
                segment_start = cumulative_lengths[index - 1]
                segment_length = cumulative_lengths[index] - segment_start
                ratio = 0.0 if segment_length <= 0.0 else (distance_mark - segment_start) / segment_length
                latitude = start[0] + (end[0] - start[0]) * ratio
                longitude = start[1] + (end[1] - start[1]) * ratio
                return (float(latitude), float(longitude))
        return polyline[-1]

    def _insert_stops(self, base_positions: Sequence[LatLon], target_duration_steps: int) -> List[LatLon]:
        positions = list(base_positions)
        extra_steps = target_duration_steps - len(positions)
        if extra_steps <= 0:
            return positions[:target_duration_steps]

        while extra_steps > 0:
            insertion_index = self.rng.randint(0, len(positions) - 1)
            max_block = min(self.config.route_generation.stop_duration_max, extra_steps)
            min_block = min(self.config.route_generation.stop_duration_min, max_block)
            block_length = self.rng.randint(min_block, max_block)
            positions[insertion_index:insertion_index] = [positions[insertion_index]] * block_length
            extra_steps -= block_length
        return positions[:target_duration_steps]

    def _routes_to_frame(self, routes: Sequence[Tuple[str, RouteSample]]) -> pd.DataFrame:
        rows = []
        for route_id, route in routes:
            for time_index, (latitude, longitude) in enumerate(route.positions):
                rows.append(
                    {
                        "route_id": route_id,
                        "time": time_index,
                        "latitude": latitude,
                        "longitude": longitude,
                        "route_type": route.route_type,
                        "target_fog_id": route.target_fog_id if route.target_fog_id is not None else "",
                    }
                )
        return pd.DataFrame(rows)

    def _routes_to_rssi_frame(self, routes: Sequence[Tuple[str, RouteSample]]) -> pd.DataFrame:
        rows = []
        for route_id, route in routes:
            for time_index, (latitude, longitude) in enumerate(route.positions):
                rssi_values = self._vectorized_rssi_values(latitude, longitude)
                if self.config.radio.add_rssi_noise:
                    rssi_values = rssi_values + self.noise_rng.normal(
                        0.0,
                        self.config.radio.rssi_noise_std_db,
                        size=rssi_values.shape,
                    )
                row = {
                    "route_id": route_id,
                    "time": time_index,
                    "latitude": latitude,
                    "longitude": longitude,
                }
                for fog_id, rssi_dbm in zip(self.fog_ids, rssi_values):
                    row[f"RSSI_FOG_{int(fog_id)}"] = float(rssi_dbm)
                rows.append(row)
        return pd.DataFrame(rows)

    def _validate_passing_by(self, positions: Sequence[LatLon], fog: FogNode) -> bool:
        rssi_values = self._rssi_series(positions, fog)
        peak_index = int(np.argmax(rssi_values))
        if peak_index == 0 or peak_index == len(rssi_values) - 1:
            return False
        edge_mean = (rssi_values[0] + rssi_values[-1]) / 2.0
        return rssi_values[peak_index] >= edge_mean + 3.0

    def _validate_handover(self, positions: Sequence[LatLon], fog_a: FogNode, fog_b: FogNode) -> bool:
        a_series = np.asarray(self._rssi_series(positions, fog_a))
        b_series = np.asarray(self._rssi_series(positions, fog_b))
        dominant = np.where(a_series >= b_series, fog_a.fog_id, fog_b.fog_id)
        return (
            dominant[0] == fog_a.fog_id
            and dominant[-1] == fog_b.fog_id
            and np.any(dominant[:-1] != dominant[1:])
            and a_series[0] > a_series[-1]
            and b_series[-1] > b_series[0]
        )

    def _validate_u_turn(self, positions: Sequence[LatLon], fog: FogNode) -> bool:
        rssi_values = np.asarray(self._rssi_series(positions, fog))
        trough_index = int(np.argmin(rssi_values))
        if trough_index == 0 or trough_index == len(rssi_values) - 1:
            return False
        edge_mean = (rssi_values[0] + rssi_values[-1]) / 2.0
        return edge_mean >= rssi_values[trough_index] + 3.0

    def _rssi_series(self, positions: Sequence[LatLon], fog: FogNode) -> List[float]:
        return [
            rssi_from_distance_dbm(haversine_distance_m(lat, lon, fog.latitude, fog.longitude))
            for lat, lon in positions
        ]

    def _required_distance_m(self, duration_steps: int, speed_mps: float) -> float:
        return max(0.0, (duration_steps - 1) * speed_mps * self.config.route_generation.step_seconds)

    def _polyline_length_m(self, polyline: Sequence[LatLon]) -> float:
        return sum(haversine_distance_m(a[0], a[1], b[0], b[1]) for a, b in zip(polyline, polyline[1:]))

    def _sample_point_on_edge(self, u: int, v: int) -> LatLon:
        geometry = list(self.walkway.graph[u][v]["geometry"])
        if len(geometry) == 2:
            ratio = self.rng.random()
            return (
                geometry[0][0] + (geometry[1][0] - geometry[0][0]) * ratio,
                geometry[0][1] + (geometry[1][1] - geometry[0][1]) * ratio,
            )

        total_length = self._polyline_length_m(geometry)
        target_distance = self.rng.random() * total_length
        cumulative = [0.0]
        for start, end in zip(geometry, geometry[1:]):
            cumulative.append(cumulative[-1] + haversine_distance_m(start[0], start[1], end[0], end[1]))
        return self._point_at_distance(geometry, cumulative, target_distance)

    def _build_fog_pairs(self) -> List[Tuple[float, int, int]]:
        pairs: List[Tuple[float, int, int]] = []
        for index, fog_a in enumerate(self.fog_nodes):
            for fog_b in self.fog_nodes[index + 1 :]:
                distance = haversine_distance_m(
                    fog_a.latitude,
                    fog_a.longitude,
                    fog_b.latitude,
                    fog_b.longitude,
                )
                pairs.append((distance, fog_a.fog_id, fog_b.fog_id))
        pairs.sort(key=lambda item: item[0])
        return pairs

    def _pick_path_from_source(
        self,
        start: int,
        required_distance: float,
        max_factor: float,
        candidate_nodes: Optional[Sequence[int]] = None,
    ) -> Optional[List[int]]:
        lengths, paths = nx.single_source_dijkstra(
            self.walkway.graph,
            start,
            cutoff=required_distance * max_factor,
            weight="length_m",
        )
        candidate_set = set(candidate_nodes) if candidate_nodes is not None else None
        candidates = [
            node_id
            for node_id, distance in lengths.items()
            if node_id != start
            and required_distance <= distance <= required_distance * max_factor
            and (candidate_set is None or node_id in candidate_set)
        ]
        if not candidates:
            return None
        end = self.rng.choice(candidates)
        return paths[end]

    def _sorted_nodes_by_distance_to_fog(self, fog: FogNode, limit: int = 25) -> List[int]:
        return self._fog_sorted_node_ids[fog.fog_id][:limit]

    def _random_node_far_from_fog(self, fog: FogNode, minimum_distance_m: float) -> int:
        distances = self._fog_node_distances[fog.fog_id]
        candidates = [self.node_ids[index] for index in np.flatnonzero(distances >= minimum_distance_m)]
        if not candidates:
            raise RuntimeError(f"No graph nodes are at least {minimum_distance_m}m away from fog {fog.fog_id}.")
        return self.rng.choice(candidates)

    def _vectorized_distances_to_fog(self, fog: FogNode) -> np.ndarray:
        radius_m = 6371000.0
        phi1 = np.radians(fog.latitude)
        phi2 = np.radians(self.node_latitudes)
        d_phi = np.radians(self.node_latitudes - fog.latitude)
        d_lam = np.radians(self.node_longitudes - fog.longitude)
        a = np.sin(d_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lam / 2.0) ** 2
        a = np.clip(a, 0.0, 1.0)
        return 2.0 * radius_m * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    def _vectorized_rssi_values(self, latitude: float, longitude: float) -> np.ndarray:
        radius_m = 6371000.0
        phi1 = np.radians(latitude)
        phi2 = np.radians(self.fog_latitudes)
        d_phi = np.radians(self.fog_latitudes - latitude)
        d_lam = np.radians(self.fog_longitudes - longitude)
        a = np.sin(d_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lam / 2.0) ** 2
        a = np.clip(a, 0.0, 1.0)
        distances = 2.0 * radius_m * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        effective = np.maximum(distances, self._rssi_reference_distance_m)
        return self._rssi_reference_dbm - (
            10.0 * self._rssi_environment_coeff * np.log10(effective / self._rssi_reference_distance_m)
        )
