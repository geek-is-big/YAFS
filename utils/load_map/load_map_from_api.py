from pathlib import Path

import osmnx as ox
from shapely.geometry import Polygon


OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api",
    "https://overpass.kumi.systems/api",
    "https://overpass.openstreetmap.ru/api",
]
REQUEST_TIMEOUT_SECONDS = 30

# Define corners in (lat, lon).
CBD_CORNERS_LAT_LON = {
    "north_west": (-37.813086, 144.951210),
    "north_east": (-37.806889, 144.971577),
    "south_east": (-37.815256, 144.975066),
    "south_west": (-37.821401, 144.954892),
}


def configure_osmnx() -> None:
    ox.settings.max_query_area_size = 50 * 1000 * 50 * 1000
    ox.settings.requests_timeout = REQUEST_TIMEOUT_SECONDS
    ox.settings.overpass_rate_limit = False
    ox.settings.use_cache = True
    ox.settings.log_console = True
    ox.settings.cache_folder = str(Path(__file__).resolve().parent / "cache")


def build_search_polygon() -> Polygon:
    # Shapely expects (lon, lat) and a non-self-intersecting order.
    ordered_lat_lon = [
        CBD_CORNERS_LAT_LON["north_west"],
        CBD_CORNERS_LAT_LON["north_east"],
        CBD_CORNERS_LAT_LON["south_east"],
        CBD_CORNERS_LAT_LON["south_west"],
    ]
    lon_lat = [(lon, lat) for lat, lon in ordered_lat_lon]
    polygon = Polygon(lon_lat)
    if not polygon.is_valid:
        raise ValueError("The CBD polygon is invalid. Check corner ordering.")
    return polygon


def download_graph_with_fallback(bbox: tuple[float, float, float, float]):
    last_error = None
    for endpoint in OVERPASS_ENDPOINTS:
        ox.settings.overpass_url = endpoint
        print(f"Trying Overpass endpoint: {endpoint}")
        try:
            return ox.graph_from_bbox(bbox=bbox, network_type="walk")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"Failed on {endpoint}: {exc}")
    raise RuntimeError(
        "Failed to download OSM walking graph from all configured Overpass endpoints."
    ) from last_error


def main() -> None:
    configure_osmnx()
    search_polygon = build_search_polygon()

    # OSMnx 2.x expects bbox as (left, bottom, right, top) = (west, south, east, north).
    west, south, east, north = search_polygon.bounds
    bbox = (west, south, east, north)
    print(
        f"Calculated bounding box - North: {north}, South: {south}, East: {east}, West: {west}"
    )

    print("Downloading rectangular bounding box area from OSM...")
    graph_box = download_graph_with_fallback(bbox=bbox)

    nodes, edges = ox.graph_to_gdfs(graph_box)
    edges_filtered = edges[edges.intersects(search_polygon)]
    if edges_filtered.empty:
        raise RuntimeError("No walkable edges found inside the target polygon.")

    valid_node_ids = set(edges_filtered.index.get_level_values("u")).union(
        set(edges_filtered.index.get_level_values("v"))
    )
    nodes_filtered = nodes[nodes.index.isin(valid_node_ids)]

    # Rebuild a topologically valid graph for downstream sequence modeling workflows.
    ox.graph_from_gdfs(nodes_filtered, edges_filtered)

    edges_df = edges_filtered.reset_index()
    edges_df["geometry_wkt"] = edges_df["geometry"].apply(lambda geom: geom.wkt)

    columns_to_keep = ["u", "v", "name", "highway", "length", "geometry_wkt"]
    columns_to_save = [col for col in columns_to_keep if col in edges_df.columns]
    final_df = edges_df[columns_to_save]

    output_filename = Path(__file__).resolve().parent / "parallelogram_walkways.csv"
    final_df.to_csv(output_filename, index=False, encoding="utf-8")

    print(f"Done! Successfully extracted {len(final_df)} pedestrian segments inside the polygon.")
    print(f"Data saved to {output_filename}")


if __name__ == "__main__":
    main()
