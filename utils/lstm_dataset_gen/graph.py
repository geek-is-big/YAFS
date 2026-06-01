from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import pandas as pd

from .compat import haversine_distance_m


LatLon = Tuple[float, float]


@dataclass(frozen=True)
class FogNode:
    fog_id: int
    latitude: float
    longitude: float
    details: str


@dataclass
class WalkwayGraph:
    graph: nx.Graph
    edge_list: List[Tuple[int, int]]
    node_positions: Dict[int, LatLon]

    def path_length_m(self, path: Sequence[int]) -> float:
        return sum(float(self.graph[u][v]["length_m"]) for u, v in zip(path, path[1:]))

    def polyline_for_path(self, path: Sequence[int]) -> List[LatLon]:
        if len(path) == 1:
            return [self.node_positions[path[0]]]

        polyline: List[LatLon] = []
        for u, v in zip(path, path[1:]):
            geometry = list(self.graph[u][v]["geometry"])
            start = self.node_positions[u]
            if haversine_distance_m(geometry[0][0], geometry[0][1], start[0], start[1]) > haversine_distance_m(
                geometry[-1][0], geometry[-1][1], start[0], start[1]
            ):
                geometry.reverse()

            if not polyline:
                polyline.extend(geometry)
            else:
                polyline.extend(geometry[1:])
        return polyline


def parse_linestring_wkt(linestring_wkt: str) -> List[LatLon]:
    text = str(linestring_wkt).strip()
    if not text.startswith("LINESTRING"):
        raise ValueError(f"Unsupported geometry: {linestring_wkt}")

    body = text[text.find("(") + 1 : text.rfind(")")]
    points: List[LatLon] = []
    for token in body.split(","):
        lon_str, lat_str, *_ = token.strip().split()
        points.append((float(lat_str), float(lon_str)))
    if len(points) < 2:
        raise ValueError(f"Expected at least 2 points in geometry: {linestring_wkt}")
    return points


def _largest_component_nodes(graph: nx.Graph) -> Iterable[int]:
    components = list(nx.connected_components(graph))
    if not components:
        raise ValueError("Walkway graph is empty.")
    return max(components, key=len)


def load_walkway_graph(csv_path: Path) -> WalkwayGraph:
    df = pd.read_csv(csv_path)
    graph = nx.Graph()
    node_positions: Dict[int, LatLon] = {}

    for row in df.itertuples(index=False):
        geometry = parse_linestring_wkt(row.geometry_wkt)
        u = int(row.u)
        v = int(row.v)
        length_m = float(row.length)

        node_positions.setdefault(u, geometry[0])
        node_positions.setdefault(v, geometry[-1])
        graph.add_node(u, latitude=node_positions[u][0], longitude=node_positions[u][1])
        graph.add_node(v, latitude=node_positions[v][0], longitude=node_positions[v][1])
        graph.add_edge(u, v, length_m=length_m, geometry=tuple(geometry), highway=str(row.highway), name=row.name)

    component_nodes = set(_largest_component_nodes(graph))
    filtered_graph = graph.subgraph(component_nodes).copy()
    filtered_positions = {node_id: node_positions[node_id] for node_id in filtered_graph.nodes}
    return WalkwayGraph(
        graph=filtered_graph,
        edge_list=list(filtered_graph.edges()),
        node_positions=filtered_positions,
    )


def load_fog_nodes(csv_path: Path) -> List[FogNode]:
    df = pd.read_csv(csv_path)
    fog_df = df.copy().sort_values("ID")
    return [
        FogNode(
            fog_id=int(row.ID),
            latitude=float(row.Latitude),
            longitude=float(row.Longitude),
            details=str(row.Details),
        )
        for row in fog_df.itertuples(index=False)
    ]
