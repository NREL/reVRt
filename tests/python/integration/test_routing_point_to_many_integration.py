"""reVrt tests for routing utilities"""

import random
from pathlib import Path

import pytest
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
from rasterio.transform import from_origin

from revrt.utilities import LayeredFile, features_to_route_table
from revrt.costs.config import TransmissionConfig, parse_cap_class
from revrt.routing.point_to_many import BatchRouteProcessor, RoutingScenario
from revrt.routing.cli import _convert_to_route_definitions
from revrt.routing.utilities import map_to_costs

DEFAULT_CONFIG = TransmissionConfig()
DEFAULT_BARRIER_CONFIG = {
    "multiplier_layer": "transmission_barrier",
    "multiplier_scalar": 100,
}
CHECK_COLS = ("start_index", "length_km", "cost", "index")


def _cap_class_to_cap(capacity):
    """Get capacity for a capacity class"""
    capacity_class = parse_cap_class(capacity)
    return DEFAULT_CONFIG["power_classes"][capacity_class]


def check(truth, test, check_cols=CHECK_COLS):
    """Compare values in truth and test for given columns"""
    if check_cols is None:
        check_cols = truth.columns

    truth = truth.sort_values(["start_index", "index"])
    test = test.sort_values(["start_index", "index"])

    for c in check_cols:
        msg = f"values for {c} do not match!"
        c_truth = truth[c]
        c_test = test[c]
        assert np.allclose(c_truth, c_test, equal_nan=True), msg


@pytest.fixture(scope="module")
def routing_data_dir(test_data_dir):
    """Generate test BA regions and network nodes from ISO shapes"""
    return test_data_dir / "routing"


@pytest.fixture(scope="module")
def route_table(revx_transmission_layers, routing_data_dir):
    """Generate test BA regions and network nodes from ISO shapes"""

    with xr.open_dataset(
        revx_transmission_layers, consolidated=False, engine="zarr"
    ) as f:
        cost_crs = f.rio.crs
        features = routing_data_dir / "ri_county_centroids.gpkg"
        route_feats = gpd.read_file(features).to_crs(cost_crs)
        route_points = features_to_route_table(route_feats)
        return map_to_costs(
            route_points,
            crs=f.rio.crs,
            transform=f.rio.transform(),
            shape=f.rio.shape,
        )


@pytest.fixture(scope="module")
def sample_large_layered_data(tmp_path_factory):
    """Sample (large) layered data files to use across tests"""
    data_dir = tmp_path_factory.mktemp("routing_data")

    layered_fp = data_dir / "test_large_layered.zarr"
    layer_file = LayeredFile(layered_fp)

    height, width = (1000, 1000)
    cell_size = 1.0
    x0, y0 = 0.0, float(height)
    transform = from_origin(x0, y0, cell_size, cell_size)
    x_coords = (
        x0 + np.arange(width, dtype=np.float32) * cell_size + cell_size / 2
    )
    y_coords = (
        y0 - np.arange(height, dtype=np.float32) * cell_size - cell_size / 2
    )

    layer_1 = -1 * np.ones((1, height, width), dtype=np.float32)
    layer_1[0, 5, 0] = 1000
    layer_1[0, 5, 900] = 100

    for ind, routing_layer in enumerate(
        [
            layer_1,
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


@pytest.mark.parametrize("ignore_invalid_costs", [True, False])
def test_soft_barrier_with_large_dataset(
    sample_large_layered_data, ignore_invalid_costs, tmp_path
):
    """Test that soft barriers work as expected in point-to-many routing"""
    scenario = RoutingScenario(
        cost_fpath=sample_large_layered_data,
        cost_layers=[{"layer_name": "layer_1"}],
        ignore_invalid_costs=ignore_invalid_costs,
    )

    out_gpkg = tmp_path / "routes.gpkg"

    route_computer = BatchRouteProcessor(
        routing_scenario=scenario, route_definitions=[([(5, 0)], [(5, 900)])]
    )
    route_computer.process(out_fp=out_gpkg, save_paths=True)

    if ignore_invalid_costs:
        assert not out_gpkg.exists()
    else:
        output = gpd.read_file(out_gpkg)
        assert len(output) == 1
        route = output.iloc[0]
        assert route["cost"] == pytest.approx(550.0)
        assert route["length_km"] == pytest.approx(0.9)
        x, y = route["geometry"].xy
        assert np.allclose(x, np.linspace(0.5, 900.5, num=901))
        assert np.allclose(y, 994.5)


@pytest.mark.parametrize("capacity", [100, 200, 400, 1000, 3000])
def test_revx_capacity_class(
    revx_transmission_layers, capacity, route_table, tmp_path, routing_data_dir
):
    """Test reVX capacity class routing against known outputs"""
    cap = _cap_class_to_cap(capacity)
    routing_scenario = RoutingScenario(
        cost_fpath=revx_transmission_layers,
        cost_layers=[{"layer_name": f"tie_line_costs_{cap}MW"}],
        friction_layers=[DEFAULT_BARRIER_CONFIG],
    )

    out_fp = tmp_path / f"least_cost_paths_{capacity}MW.csv"
    route_definitions, route_attrs = _convert_to_route_definitions(route_table)
    route_computer = BatchRouteProcessor(
        routing_scenario=routing_scenario,
        route_definitions=route_definitions,
        route_attrs=route_attrs,
    )
    route_computer.process(out_fp=out_fp, save_paths=False)

    truth = routing_data_dir / f"least_cost_paths_{capacity}MW.csv"
    test = pd.read_csv(out_fp)
    truth = pd.read_csv(truth)
    check(truth, test)


def test_revx_invariant_costs(
    revx_transmission_layers, route_table, tmp_path, routing_data_dir
):
    """Test reVX invariant cost layer routing against known outputs"""

    capacity = random.choice([100, 200, 400, 1000, 3000])  # noqa: S311
    cap = _cap_class_to_cap(capacity)
    base_costs = {
        "layer_name": f"tie_line_costs_{cap}MW",
        "multiplier_scalar": 1e-9,
    }
    cost_layer = {
        "layer_name": f"tie_line_costs_{cap}MW",
        "is_invariant": True,
    }
    routing_scenario = RoutingScenario(
        cost_fpath=revx_transmission_layers,
        cost_layers=[base_costs, cost_layer],
        friction_layers=[DEFAULT_BARRIER_CONFIG],
        ignore_invalid_costs=False,
    )

    out_fp = tmp_path / f"least_cost_paths_{capacity}MW.csv"
    route_definitions, route_attrs = _convert_to_route_definitions(route_table)
    route_computer = BatchRouteProcessor(
        routing_scenario=routing_scenario,
        route_definitions=route_definitions,
        route_attrs=route_attrs,
    )
    route_computer.process(out_fp=out_fp, save_paths=False)

    truth = routing_data_dir / f"least_cost_paths_{capacity}MW.csv"
    test = pd.read_csv(out_fp)
    truth = pd.read_csv(truth)

    truth = truth.sort_values(["start_index", "index"])
    test = test.sort_values(["start_index", "index"])

    assert (test["cost"].to_numpy() < truth["cost"].to_numpy()).all()


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
