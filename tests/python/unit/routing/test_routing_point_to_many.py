"""reVrt tests for routing one point to many endpoints"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
import numpy as np
import xarray as xr
from rasterio.transform import from_origin

from revrt.utilities import LayeredFile
from revrt.routing.point_to_many import find_all_routes, RoutingScenario


@pytest.fixture(scope="module")
def sample_layered_data(tmp_path_factory):
    """Sample layered data files to use across tests"""
    data_dir = tmp_path_factory.mktemp("routing_data")

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

    layer_1 = np.array(
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
    )

    layer_2 = np.array(
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
    )

    layer_3 = np.array(
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
    )

    layer_4 = np.array(
        [
            [
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ]
        ],
        dtype=np.float32,
    )

    layer_5 = np.array(
        [
            [
                [0, 0, 0, 1, 1, 1, 1, 1],
                [0, 0, 0, 1, 1, 1, 1, 1],
                [0, 0, 0, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 1, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ]
        ],
        dtype=np.float32,
    )

    for ind, routing_layer in enumerate(
        [
            layer_1,
            layer_2,
            layer_3,
            layer_4,
            layer_5,
        ],
        start=1,
    ):
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


def test_basic_single_route_layered_file_short_path(sample_layered_data):
    """Test routing using a LayeredFile-generated cost surface"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
    )

    output = find_all_routes(
        scenario,
        route_definitions=[
            ((1, 1), [(1, 2)], {}),
        ],
        save_paths=False,
    )

    assert len(output) == 1
    assert output.iloc[0]["cost"] == pytest.approx((1 + 2) / 2)
    assert output.iloc[0]["length_km"] == 1 / 1000
    assert output.iloc[0]["cost"] == output.iloc[0]["optimized_objective"]


def test_basic_single_route_layered_file(sample_layered_data):
    """Test routing using a LayeredFile-generated cost surface"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
    )

    output = find_all_routes(
        scenario,
        route_definitions=[
            ((1, 1), [(2, 6)], {}),
            ((1, 2), [(2, 6)], {}),
        ],
        save_paths=False,
    )

    assert len(output) == 2
    assert output.iloc[0]["cost"] == pytest.approx(11.192389)
    assert output.iloc[0]["length_km"] == pytest.approx(0.0090710678)
    assert np.isclose(
        output.iloc[0]["cost"], output.iloc[0]["optimized_objective"]
    )

    assert output.iloc[1]["cost"] == pytest.approx(12.278174)
    assert output.iloc[1]["length_km"] == pytest.approx(0.008656854)
    assert np.isclose(
        output.iloc[1]["cost"], output.iloc[1]["optimized_objective"]
    )


def test_multi_layer_route_layered_file(sample_layered_data):
    """Test routing across multiple cost layers"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[
            {"layer_name": "layer_1"},
            {"layer_name": "layer_2"},
        ],
    )

    output = find_all_routes(
        scenario,
        route_definitions=[
            ((1, 1), [(2, 6)], {}),
            ((1, 2), [(2, 6)], {}),
        ],
        save_paths=False,
    )

    assert len(output) == 2

    first_route = output.iloc[0]
    assert first_route["cost"] == pytest.approx(
        27.606602,
        rel=1e-4,
    )
    assert first_route["length_km"] == pytest.approx(
        0.005414,
        rel=1e-4,
    )
    assert first_route["layer_1_cost"] == pytest.approx(
        17.571068,
        rel=1e-4,
    )
    assert first_route["layer_2_cost"] == pytest.approx(
        10.035534,
        rel=1e-4,
    )
    assert np.isclose(
        first_route["cost"], first_route["optimized_objective"], rtol=1e-6
    )

    second_route = output.iloc[1]
    assert second_route["cost"] == pytest.approx(
        25.106602,
        rel=1e-4,
    )
    assert second_route["length_km"] == pytest.approx(
        0.004414,
        rel=1e-4,
    )
    assert second_route["layer_1_cost"] == pytest.approx(
        16.071068,
        rel=1e-4,
    )
    assert second_route["layer_2_cost"] == pytest.approx(
        9.035534,
        rel=1e-4,
    )
    assert np.isclose(
        second_route["cost"], second_route["optimized_objective"], rtol=1e-6
    )


def test_save_paths_returns_expected_geometry(sample_layered_data):
    """Saving paths returns expected geometries for each route"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
    )

    output = find_all_routes(
        scenario,
        route_definitions=[
            ((1, 1), [(2, 6)], {}),
            ((1, 2), [(2, 6)], {}),
        ],
        save_paths=True,
    )

    assert isinstance(output, list)
    assert len(output) == 2

    expected_geometries = [
        [
            (1.5, 5.5),
            (1.5, 4.5),
            (2.5, 3.5),
            (3.5, 2.5),
            (4.5, 1.5),
            (5.5, 2.5),
            (5.5, 3.5),
            (6.5, 4.5),
        ],
        [
            (2.5, 5.5),
            (2.5, 4.5),
            (3.5, 3.5),
            (3.5, 2.5),
            (4.5, 1.5),
            (5.5, 2.5),
            (5.5, 3.5),
            (6.5, 4.5),
        ],
    ]

    for geom, expected_coords in zip(
        output.geometry, expected_geometries, strict=True
    ):
        assert geom.geom_type == "LineString"
        assert np.allclose(
            np.asarray(geom.coords), np.asarray(expected_coords)
        )


def test_empty_route_definitions_returns_empty_dataframe(sample_layered_data):
    """Empty route definitions return an empty dataframe"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
    )

    output = find_all_routes(
        scenario,
        route_definitions=[],
        save_paths=False,
    )

    assert isinstance(output, list)
    assert not output


def test_empty_route_definitions_returns_empty_geo_dataframe(
    sample_layered_data,
):
    """Empty route definitions return an empty dataframe"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
    )

    output = find_all_routes(
        scenario,
        route_definitions=[],
        save_paths=True,
    )

    assert isinstance(output, list)
    assert not output


def test_multi_layer_route_with_multiplier(sample_layered_data):
    """Test routing with multiple layers and a scalar multiplier"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[
            {"layer_name": "layer_1"},
            {
                "layer_name": "layer_2",
                "multiplier_scalar": 0.5,
            },
        ],
    )

    output = find_all_routes(
        scenario,
        route_definitions=[
            ((1, 1), [(2, 6)], {}),
            ((1, 2), [(2, 6)], {}),
        ],
        save_paths=False,
    )

    assert len(output) == 2

    first_route = output.iloc[0]
    assert first_route["cost"] == pytest.approx(
        22.588835,
        rel=1e-4,
    )
    assert first_route["length_km"] == pytest.approx(
        0.005414,
        rel=1e-4,
    )
    assert first_route["layer_1_cost"] == pytest.approx(
        17.571068,
        rel=1e-4,
    )
    assert first_route["layer_2_cost"] == pytest.approx(
        5.017767,
        rel=1e-4,
    )
    assert np.isclose(
        first_route["cost"],
        first_route["optimized_objective"],
        rtol=1e-4,
    )

    second_route = output.iloc[1]
    assert second_route["cost"] == pytest.approx(
        20.588835,
        rel=1e-4,
    )
    assert second_route["length_km"] == pytest.approx(
        0.004414,
        rel=1e-4,
    )
    assert second_route["layer_1_cost"] == pytest.approx(
        16.071068,
        rel=1e-4,
    )
    assert second_route["layer_2_cost"] == pytest.approx(
        4.517767,
        rel=1e-4,
    )
    assert np.isclose(
        second_route["cost"],
        second_route["optimized_objective"],
        rtol=1e-4,
    )


def test_multi_layer_route_with_scalar_and_layer_multipliers(
    sample_layered_data,
):
    """Test routing when combining scalar and layer multipliers"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[
            {"layer_name": "layer_1"},
            {"layer_name": "layer_2", "multiplier_scalar": 0.5},
            {
                "layer_name": "layer_3",
                "multiplier_layer": "layer_4",
            },
            {
                "layer_name": "layer_5",
                "multiplier_scalar": 2,
                "multiplier_layer": "layer_4",
            },
        ],
    )

    output = find_all_routes(
        scenario,
        route_definitions=[
            ((1, 1), [(1, 2)], {}),
        ],
        save_paths=False,
    )

    assert len(output) == 1

    route = output.iloc[0]
    assert route["cost"] == pytest.approx(2.0, rel=1e-4)
    assert route["length_km"] == pytest.approx(0.001, rel=1e-4)
    assert route["layer_1_cost"] == pytest.approx(1.5, rel=1e-4)
    assert route["layer_2_cost"] == pytest.approx(0.5, rel=1e-4)
    assert route["layer_3_cost"] == pytest.approx(0.0, abs=1e-8)
    assert route["layer_5_cost"] == pytest.approx(0.0, abs=1e-8)
    assert route["layer_1_dist_km"] == pytest.approx(0.001, rel=1e-4)
    assert route["layer_2_dist_km"] == pytest.approx(0.001, rel=1e-4)
    assert route["layer_3_dist_km"] == pytest.approx(0.0, abs=1e-8)
    assert route["layer_5_dist_km"] == pytest.approx(0.0, abs=1e-8)
    assert np.isclose(route["cost"], route["optimized_objective"], rtol=1e-6)


def test_routing_with_tracked_layers(sample_layered_data):
    """Tracked layers report aggregated stats alongside routing results"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
        tracked_layers={
            "layer_1": "mean",
            "layer_2": "max",
            "layer_3": "min",
        },
    )

    output = find_all_routes(
        scenario,
        route_definitions=[
            ((1, 1), [(1, 2)], {}),
        ],
        save_paths=False,
    )

    assert len(output) == 1
    route = output.iloc[0]

    assert {
        "layer_1_mean",
        "layer_2_max",
        "layer_3_min",
    }.issubset(output.columns)

    assert route["layer_1_mean"] == pytest.approx(1.5)
    assert route["layer_2_max"] == pytest.approx(1.0)
    assert route["layer_3_min"] == pytest.approx(2.0)


def test_start_point_on_barrier_returns_no_route(
    sample_layered_data, assert_message_was_logged
):
    """If the start point is on a barrier (cost <= 0) no route is returned"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
    )

    # (0, 3) in layer_1 is 0 -> treated as barrier
    output = find_all_routes(
        scenario,
        route_definitions=[
            ((0, 3), [(2, 6)], {}),
        ],
        save_paths=False,
    )
    assert_message_was_logged(
        "Start idx (0, 3) does not have a valid cost: -1.00 (must be > 0)!",
        "ERROR",
    )

    assert isinstance(output, list)
    assert not output


def test_some_endpoints_include_barriers_but_one_valid(sample_layered_data):
    """If some end points <=0 but at least one is valid, route is found"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
    )

    # include one barrier end (0,3) and one valid end (2,6)
    output = find_all_routes(
        scenario,
        route_definitions=[
            ((1, 1), [(0, 3), (2, 6)], {}),
        ],
        save_paths=False,
    )

    assert len(output) == 1
    # At least one valid endpoint must be reached and cost must be positive.
    assert output.iloc[0]["cost"] > 0

    end_row = int(output.iloc[0]["end_row"])
    end_col = int(output.iloc[0]["end_col"])
    assert (end_row, end_col) == (2, 6)


def test_all_endpoints_are_barriers_returns_no_route(
    sample_layered_data, assert_message_was_logged
):
    """If all end points are barriers, no route is returned"""

    scenario = RoutingScenario(
        cost_fpath=sample_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
    )

    output = find_all_routes(
        scenario,
        route_definitions=[
            ((1, 1), [(0, 3), (0, 7)], {}),
        ],
        save_paths=False,
    )
    assert_message_was_logged(
        "None of the end idx [(0, 3), (0, 7)] have a valid cost (must be > 0)",
        "ERROR",
    )

    assert isinstance(output, list)
    assert not output


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
