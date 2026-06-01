from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
from shapely.geometry import Polygon

# Define corners in (lat, lon).
CBD_CORNERS_LAT_LON = {
    "north_west": (-37.813086, 144.951210),
    "north_east": (-37.806889, 144.971577),
    "south_east": (-37.815256, 144.975066),
    "south_west": (-37.821401, 144.954892),
}


def build_search_polygon() -> Polygon:
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


def parse_overpass_ways(osm_json_path: Path) -> gpd.GeoDataFrame:
    data = json.loads(osm_json_path.read_text(encoding="utf-8"))
    elements = data.get("elements", [])
    if not elements:
        raise ValueError(f"No 'elements' found in {osm_json_path}")

    rows: list[dict] = []
    generated_node_id = -1

    for element in elements:
        if element.get("type") != "way":
            continue

        node_ids = element.get("nodes", [])
        geometry_points = element.get("geometry", [])
        tags = element.get("tags", {})
        way_id = element.get("id")

        usable_len = min(len(node_ids), len(geometry_points))
        if usable_len < 2:
            continue

        for i in range(usable_len - 1):
            n1 = node_ids[i]
            n2 = node_ids[i + 1]
            p1 = geometry_points[i]
            p2 = geometry_points[i + 1]

            # Fallback for malformed input where node IDs might be missing.
            if n1 is None:
                n1 = generated_node_id
                generated_node_id -= 1
            if n2 is None:
                n2 = generated_node_id
                generated_node_id -= 1

            line = LineString([(p1["lon"], p1["lat"]), (p2["lon"], p2["lat"])])
            rows.append(
                {
                    "u": int(n1),
                    "v": int(n2),
                    "way_id": int(way_id) if way_id is not None else None,
                    "name": tags.get("name"),
                    "highway": tags.get("highway"),
                    "geometry": line,
                }
            )

    if not rows:
        raise ValueError(f"No valid way segments could be parsed from {osm_json_path}")

    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def compute_length_meters(segments_gdf: gpd.GeoDataFrame) -> pd.Series:
    utm_crs = segments_gdf.estimate_utm_crs()
    if utm_crs is None:
        raise RuntimeError("Could not estimate projected CRS for length calculation.")
    return segments_gdf.to_crs(utm_crs).geometry.length


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build walkable route segments from a predownloaded Overpass JSON file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "maps" / "Melbourne_rect_osm.json",
        help="Path to Overpass JSON map file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "parallelogram_walkways.csv",
        help="Output CSV path for LSTM-ready walkway segments.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    search_polygon = build_search_polygon()
    segments = parse_overpass_ways(args.input)
    segments_filtered = segments[segments.intersects(search_polygon)].copy()

    if segments_filtered.empty:
        raise RuntimeError("No walkable segments found inside the target polygon.")

    segments_filtered["length"] = compute_length_meters(segments_filtered)
    segments_filtered["geometry_wkt"] = segments_filtered.geometry.apply(lambda geom: geom.wkt)

    columns_to_keep = ["u", "v", "name", "highway", "length", "geometry_wkt"]
    final_df = segments_filtered[columns_to_keep]
    final_df.to_csv(args.output, index=False, encoding="utf-8")

    print(f"Parsed segments from: {args.input}")
    print(f"Filtered segments in polygon: {len(final_df)}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
