"""reVRt routing CLI unit tests"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point, LineString
from rasterio.transform import from_origin

from revrt.utilities import LayeredFile
from revrt.routing.utilities import map_to_costs
from revrt.routing.cli import (
    compute_lcp_routes,
    _run_lcp,
    _collect_existing_routes,
    _update_multipliers,
    _route_points_subset,
    _paths_to_compute,
    _split_routes,
    _MILLION_USD_PER_MILE_TO_USD_PER_PIXEL,
)


@pytest.fixture(scope="module")
def sample_layered_data(tmp_path_factory):
    """Create layered routing data mimicking point_to_many tests"""

    data_dir = tmp_path_factory.mktemp("routing_cli_data")

    layered_fp = data_dir / "test_layered.zarr"
    layer_file = LayeredFile(layered_fp)

    height, width = (7, 8)
    cell_size = 1.0
    x0, y0 = 0.0, float(height)
    transform = from_origin(x0, y0, cell_size, cell_size)
    x_coords = (
        x0 + np.arange(width, dtype=np.float32) * cell_size + cell_size / 2
    )
    y_coords = (
        y0 - np.arange(height, dtype=np.float32) * cell_size - cell_size / 2
    )

    layer_values = [
        np.array(
            [
                [
                    [7, 7, 8, 0, 9, 9, 9, 0],
                    [8, 1, 2, 2, 9, 9, 9, 0],
                    [9, 1, 3, 3, 9, 1, 2, 3],
                    [9, 1, 2, 1, 9, 1, 9, 0],
                    [9, 9, 9, 1, 9, 1, 9, 0],
                    [9, 9, 9, 1, 1, 1, 9, 0],
                    [9, 9, 9, 9, 9, 9, 9, 0],
                ]
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [
                    [8, 7, 6, 5, 5, 6, 7, 9],
                    [7, 1, 1, 2, 3, 3, 2, 8],
                    [6, 2, 9, 6, 5, 2, 1, 7],
                    [7, 3, 8, 1, 2, 3, 2, 6],
                    [8, 4, 7, 2, 8, 4, 3, 5],
                    [9, 5, 6, 3, 4, 4, 3, 4],
                    [9, 6, 7, 4, 5, 5, 4, 3],
                ]
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [
                    [6, 6, 6, 6, 6, 7, 8, 9],
                    [5, 2, 2, 3, 4, 5, 6, 8],
                    [4, 3, 7, 7, 6, 4, 5, 7],
                    [5, 4, 6, 2, 3, 4, 4, 6],
                    [6, 5, 5, 3, 7, 5, 5, 5],
                    [7, 6, 6, 4, 5, 5, 4, 4],
                    [8, 7, 7, 5, 6, 5, 4, 3],
                ]
            ],
            dtype=np.float32,
        ),
    ]

    for ind, routing_layer in enumerate(layer_values, start=1):
        da = xr.DataArray(
            routing_layer,
            dims=("band", "y", "x"),
            coords={"y": y_coords, "x": x_coords},
        )
        da = da.rio.write_crs("EPSG:4326")
        da = da.rio.write_transform(transform)

        geotiff_fp = data_dir / f"layer_{ind}.tif"
        da.rio.to_raster(geotiff_fp, driver="GTiff")

        layer_file.write_geotiff_to_file(
            geotiff_fp, f"layer_{ind}", overwrite=True
        )

    return layered_fp


def _build_route_table(layered_fp, rows_cols):
    """Helper to construct route tables with CRS-aligned coordinates"""

    with xr.open_dataset(layered_fp, consolidated=False, engine="zarr") as ds:
        latitudes = ds["latitude"].to_numpy()
        longitudes = ds["longitude"].to_numpy()

    records = []
    for idx, (start, end) in enumerate(rows_cols):
        s_row, s_col = start
        e_row, e_col = end
        records.append(
            {
                "route_id": f"route_{idx}",
                "start_lat": float(latitudes[s_row, s_col]),
                "start_lon": float(longitudes[s_row, s_col]),
                "end_lat": float(latitudes[e_row, e_col]),
                "end_lon": float(longitudes[e_row, e_col]),
                "voltage": 138,
                "polarity": "ac",
            }
        )

    return pd.DataFrame.from_records(records)


def test_compute_lcp_routes_generates_csv(
    sample_layered_data, tmp_path, monkeypatch
):
    """compute_lcp_routes should map points and persist CSV outputs"""

    routes = _build_route_table(
        sample_layered_data,
        rows_cols=[((1, 1), (2, 3)), ((0, 0), (3, 4))],
    )
    route_table_fp = tmp_path / "route_table.csv"
    routes.to_csv(route_table_fp, index=False)

    monkeypatch.setattr(
        "revrt.routing.cli._route_points_subset",
        lambda *_args, **_kwargs: routes,
    )
    monkeypatch.setattr(
        "revrt.routing.cli.parse_config", lambda config=None: config
    )

    out_dir = tmp_path / "routing_outputs"

    cost_layers = [
        {
            "layer_name": "layer_1",
            "apply_row_mult": True,
            "multiplier_scalar": 1,
        },
        {
            "layer_name": "layer_2",
            "apply_polarity_mult": True,
            "multiplier_scalar": 1,
        },
    ]

    transmission_config = {
        "row_width": {"138": 1.5},
        "voltage_polarity_mult": {"138": {"ac": 2e-5}},
    }

    result_fp = compute_lcp_routes(
        cost_fpath=sample_layered_data,
        route_table=route_table_fp,
        cost_layers=cost_layers,
        out_dir=out_dir,
        job_name="run",
        transmission_config=transmission_config,
        save_paths=False,
    )

    output_path = Path(result_fp)
    assert output_path.exists()

    df = pd.read_csv(output_path)
    df = df[df["route_id"] != "route_id"].reset_index(drop=True)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])

    numeric_cols = [
        "cost",
        "length_km",
        "optimized_objective",
        "layer_1_cost",
        "layer_1_dist_km",
        "layer_2_cost",
        "layer_2_dist_km",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col])

    assert len(df) == len(routes)
    assert set(df["route_id"]) == set(routes["route_id"])

    with xr.open_dataset(
        sample_layered_data, consolidated=False, engine="zarr"
    ) as ds:
        mapped_routes = map_to_costs(
            routes.copy(), ds.rio.crs, ds.rio.transform(), ds.rio.shape
        )

    merged = df.merge(
        mapped_routes[
            [
                "route_id",
                "start_row",
                "start_col",
                "end_row",
                "end_col",
            ]
        ],
        on="route_id",
        how="left",
        suffixes=("", "_expected"),
    )

    for col in ["start_row", "start_col", "end_row", "end_col"]:
        assert (
            merged[col].astype(int) == merged[f"{col}_expected"].astype(int)
        ).all()

    assert np.allclose(
        merged["cost"], merged["layer_1_cost"] + merged["layer_2_cost"]
    )
    assert np.all(merged["length_km"] > 0)
    assert np.allclose(
        merged["cost"], merged["optimized_objective"], rtol=1e-5
    )


def test_compute_lcp_routes_returns_none_on_empty_indices(
    sample_layered_data, tmp_path, monkeypatch
):
    """Ensure compute_lcp_routes short-circuits when route points are empty"""

    route_table_fp = tmp_path / "routes.csv"
    _build_route_table(sample_layered_data, [((1, 1), (2, 2))]).to_csv(
        route_table_fp, index=False
    )

    monkeypatch.setattr(
        "revrt.routing.cli._route_points_subset",
        lambda *_, **__: pd.DataFrame([]),
    )
    monkeypatch.setattr(
        "revrt.routing.cli.parse_config", lambda config=None: config
    )

    result = compute_lcp_routes(
        cost_fpath=sample_layered_data,
        route_table=route_table_fp,
        cost_layers=[{"layer_name": "layer_1"}],
        out_dir=tmp_path,
        job_name="no_routes",
    )

    assert result is None
    assert not (tmp_path / "no_routes.csv").exists()


def test_run_lcp_with_save_paths_filters_existing_routes(
    sample_layered_data, tmp_path, monkeypatch
):
    """_run_lcp should skip already processed routes and append geometries"""

    routes = _build_route_table(
        sample_layered_data,
        rows_cols=[((1, 1), (2, 2)), ((2, 2), (4, 4))],
    )

    with xr.open_dataset(
        sample_layered_data, consolidated=False, engine="zarr"
    ) as ds:
        mapped_routes = map_to_costs(
            routes.copy(), ds.rio.crs, ds.rio.transform(), ds.rio.shape
        )

    existing_tuple = (
        int(mapped_routes.iloc[0]["start_row"]),
        int(mapped_routes.iloc[0]["start_col"]),
        int(mapped_routes.iloc[0]["end_row"]),
        int(mapped_routes.iloc[0]["end_col"]),
        routes.iloc[0]["polarity"],
        str(routes.iloc[0]["voltage"]),
    )

    monkeypatch.setattr(
        "revrt.routing.cli._collect_existing_routes",
        lambda _: {existing_tuple},
    )

    saved_calls = []

    def fake_to_file(self, path, driver=None, mode=None, **_kwargs):
        saved_calls.append((path, driver, mode, self.copy(deep=True)))

    monkeypatch.setattr(
        "revrt.routing.cli.gpd.GeoDataFrame.to_file", fake_to_file
    )

    out_fp = tmp_path / "routes.gpkg"

    _run_lcp(
        cost_fpath=sample_layered_data,
        route_points=routes,
        cost_layers=[{"layer_name": "layer_1"}],
        out_fp=out_fp,
        transmission_config={
            "row_width": {"138": 1.0},
            "voltage_polarity_mult": {"138": {"ac": 1.0}},
        },
        cost_multiplier_scalar=1,
        friction_layers=[{"layer_name": "layer_2", "apply_row_mult": True}],
        tracked_layers={"layer_3": "max"},
        use_hard_barrier=True,
    )

    assert len(saved_calls) == 1
    saved_path, driver, mode, saved_gdf = saved_calls[0]
    assert saved_path == out_fp
    assert driver == "GPKG"
    assert mode == "a"
    assert len(saved_gdf) == 1
    assert saved_gdf["route_id"].iloc[0] == routes.iloc[1]["route_id"]

    expected = mapped_routes.iloc[1]
    assert int(saved_gdf["start_row"].iloc[0]) == int(expected["start_row"])
    assert int(saved_gdf["start_col"].iloc[0]) == int(expected["start_col"])
    assert int(saved_gdf["end_row"].iloc[0]) == int(expected["end_row"])
    assert int(saved_gdf["end_col"].iloc[0]) == int(expected["end_col"])

    cost_val = float(saved_gdf["cost"].iloc[0])
    objective_val = float(saved_gdf["optimized_objective"].iloc[0])
    length_val = float(saved_gdf["length_km"].iloc[0])

    assert cost_val > 0
    assert length_val > 0
    assert objective_val == pytest.approx(cost_val, rel=1e-5)

    geom = saved_gdf.geometry.iloc[0]
    assert isinstance(geom, LineString)
    assert len(geom.coords) >= 2


def test_run_lcp_returns_immediately_when_no_routes(tmp_path):
    """_run_lcp should exit early when route_points is empty"""

    _run_lcp(
        cost_fpath="unused",  # cost file is ignored in this branch
        route_points=pd.DataFrame(),
        cost_layers=[],
        out_fp=tmp_path / "unused.csv",
    )


def test_collect_existing_routes_csv(tmp_path):
    """_collect_existing_routes should read CSV outputs"""

    data = pd.DataFrame(
        [
            {
                "start_row": 0,
                "start_col": 1,
                "end_row": 2,
                "end_col": 3,
                "polarity": "ac",
                "voltage": "230",
            }
        ]
    )
    csv_fp = tmp_path / "routes.csv"
    data.to_csv(csv_fp, index=False)

    result = _collect_existing_routes(csv_fp)
    assert result == {(0, 1, 2, 3, "ac", "230")}


def test_collect_existing_routes_gpkg(monkeypatch, tmp_path):
    """_collect_existing_routes should support GeoPackage outputs"""

    gpkg_fp = tmp_path / "routes.gpkg"
    gpkg_fp.touch()

    gdf = gpd.GeoDataFrame(
        {
            "start_row": [1],
            "start_col": [2],
            "end_row": [3],
            "end_col": [4],
            "polarity": ["unknown"],
            "voltage": ["unknown"],
            "geometry": [Point(0, 0)],
        }
    )

    monkeypatch.setattr("revrt.routing.cli.gpd.read_file", lambda _: gdf)

    result = _collect_existing_routes(gpkg_fp)
    assert result == {(1, 2, 3, 4, "unknown", "unknown")}


def test_collect_existing_routes_when_missing(tmp_path):
    """Missing outputs should result in an empty existing route set"""

    assert _collect_existing_routes(None) == set()
    assert _collect_existing_routes(tmp_path / "missing.csv") == set()


def test_route_points_subset_with_chunking(monkeypatch):
    """_route_points_subset should slice sorted features by chunk"""

    features = pd.DataFrame(
        {
            "start_lat": [5.0, 1.0, 3.0, 7.0],
            "start_lon": [0.0, 1.0, 2.0, 3.0],
        }
    )

    monkeypatch.setattr("revrt.routing.cli.pd.read_csv", lambda _: features)

    first_chunk = _route_points_subset(
        "dummy", ["start_lat", "start_lon"], (0, 2)
    )
    assert first_chunk["start_lat"].tolist() == [1.0, 3.0]
    assert first_chunk["start_lon"].tolist() == [1.0, 2.0]

    second_chunk = _route_points_subset(
        "dummy", ["start_lat", "start_lon"], (1, 2)
    )
    assert second_chunk["start_lat"].tolist() == [5.0, 7.0]
    assert second_chunk["start_lon"].tolist() == [0.0, 3.0]


def test_paths_to_compute_inserts_missing_columns(tmp_path):
    """_paths_to_compute should back-fill missing polarity/voltage columns"""

    route_points = pd.DataFrame(
        {
            "start_row": [0],
            "start_col": [1],
            "end_row": [2],
            "end_col": [3],
        }
    )

    groups = list(_paths_to_compute(route_points, tmp_path / "not_there.csv"))
    assert groups
    polarity, voltage, grouped_routes = groups[0]
    assert polarity == "unknown"
    assert voltage == "unknown"
    assert grouped_routes.iloc[0]["start_row"] == 0


def test_split_routes_handles_local_and_cluster():
    """_split_routes should configure chunking for local and cluster modes"""

    local_config = {"execution_control": {"option": "local", "nodes": 4}}
    result_local = _split_routes(local_config)
    assert result_local["_split_params"] == [(0, 1)]
    assert result_local["execution_control"]["nodes"] == 4

    cluster_config = {"execution_control": {"nodes": 3}}
    result_cluster = _split_routes(cluster_config)
    assert result_cluster["_split_params"] == [(0, 3), (1, 3), (2, 3)]
    assert "nodes" not in result_cluster["execution_control"]


def test_update_multipliers_applies_row_and_polarity():
    """_update_multipliers should apply configured scalar adjustments"""

    layers = [
        {
            "layer_name": "layer_1",
            "multiplier_scalar": 2,
            "apply_row_mult": True,
        },
        {"layer_name": "layer_2", "apply_polarity_mult": True},
    ]

    transmission_config = {
        "row_width": {"138": 1.5},
        "voltage_polarity_mult": {"138": {"ac": 0.5}},
    }

    updated = _update_multipliers(
        layers,
        polarity="ac",
        voltage=138,
        transmission_config=transmission_config,
    )

    # original input remains unchanged
    assert layers[0]["apply_row_mult"] is True
    assert updated[0]["multiplier_scalar"] == pytest.approx(3)
    assert updated[1]["multiplier_scalar"] == pytest.approx(
        0.5 * _MILLION_USD_PER_MILE_TO_USD_PER_PIXEL
    )

    # Voltage marked as unknown should skip multiplier lookups
    unchanged = _update_multipliers(
        [{"layer_name": "layer_3"}], "dc", "unknown", transmission_config
    )
    assert unchanged[0]["layer_name"] == "layer_3"


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
