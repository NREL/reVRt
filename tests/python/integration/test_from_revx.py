"""reVrt tests ported from reVX"""

import os
import json
import random
import shutil
import platform
import traceback
from pathlib import Path

import pytest
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from revrt.utilities import LayeredFile, features_to_route_table
from revrt.costs.config import TransmissionConfig, parse_cap_class
from revrt.routing.base import BatchRouteProcessor, RoutingScenario
from revrt.routing.cli import (
    _convert_to_route_definitions,
    _MILLION_USD_PER_MILE_TO_USD_PER_PIXEL,
)
from revrt._cli import main
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


def _run_lcp(tmp_path, routing_scenario, route_table):
    """Run least cost path computation and return resulting dataframe"""
    out_fp = tmp_path / "least_cost_paths_test.csv"
    route_definitions, route_attrs = _convert_to_route_definitions(route_table)
    route_computer = BatchRouteProcessor(
        routing_scenario=routing_scenario,
        route_definitions=route_definitions,
        route_attrs=route_attrs,
    )
    route_computer.process(out_fp=out_fp, save_paths=False)

    return pd.read_csv(out_fp)


def _run_cli(
    tmp_path,
    routing_data_dir,
    revx_transmission_layers,
    row_config,
    polarity_config,
    route_table,
    cost_layer_config,
    cli_runner,
    save_paths=False,
):
    """Run reVRt CLI with given configs and return test and truth dataframes"""
    capacity = random.choice([100, 200, 400, 1000, 3000])  # noqa: S311
    cost_layer = f"tie_line_costs_{_cap_class_to_cap(capacity)}MW"
    cost_layer_config["layer_name"] = cost_layer
    truth = routing_data_dir / f"least_cost_paths_{capacity}MW.csv"
    truth = pd.read_csv(truth)

    row_config_path = tmp_path / "config_row.json"
    row_config_path.write_text(json.dumps(row_config))

    polarity_config_path = tmp_path / "config_polarity.json"
    polarity_config_path.write_text(json.dumps(polarity_config))

    routes_fp = tmp_path / "routes.csv"
    route_table.to_csv(routes_fp, index=False)

    config = {
        "log_directory": str(tmp_path),
        "execution_control": {"option": "local"},
        "transmission_config": {
            "row_width": str(row_config_path),
            "voltage_polarity_mult": str(polarity_config_path),
        },
        "cost_fpath": str(revx_transmission_layers),
        "route_table": str(routes_fp),
        "save_paths": save_paths,
        "cost_layers": [cost_layer_config],
        "friction_layers": [DEFAULT_BARRIER_CONFIG],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    result = cli_runner.invoke(main, ["route-points", "-c", config_path])
    msg = f"Failed with error {traceback.print_exception(*result.exc_info)}"
    assert result.exit_code == 0, msg

    if save_paths:
        test = gpd.read_file(tmp_path / f"{tmp_path.stem}_route_points.gpkg")
        assert test.geometry is not None
    else:
        test = pd.read_csv(tmp_path / f"{tmp_path.stem}_route_points.csv")
    return test, truth


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


@pytest.mark.parametrize("capacity", [100, 200, 400, 1000, 3000])
def test_capacity_class(
    revx_transmission_layers, capacity, route_table, tmp_path, routing_data_dir
):
    """Test reVX capacity class routing against known outputs"""
    cap = _cap_class_to_cap(capacity)
    routing_scenario = RoutingScenario(
        cost_fpath=revx_transmission_layers,
        cost_layers=[{"layer_name": f"tie_line_costs_{cap}MW"}],
        friction_layers=[DEFAULT_BARRIER_CONFIG],
    )

    test = _run_lcp(tmp_path, routing_scenario, route_table)

    truth = routing_data_dir / f"least_cost_paths_{capacity}MW.csv"
    truth = pd.read_csv(truth)

    check(truth, test)


def test_invariant_costs(
    revx_transmission_layers, route_table, tmp_path, routing_data_dir
):
    """Test reVX invariant cost layer routing against known outputs"""

    capacity = random.choice([100, 200, 400, 1000, 3000])  # noqa: S311
    cap = _cap_class_to_cap(capacity)
    base_costs = {
        "layer_name": f"tie_line_costs_{cap}MW",
        "multiplier_scalar": 1e-9,
    }
    invariant_cost_layer = {
        "layer_name": f"tie_line_costs_{cap}MW",
        "is_invariant": True,
    }
    routing_scenario = RoutingScenario(
        cost_fpath=revx_transmission_layers,
        cost_layers=[base_costs, invariant_cost_layer],
        friction_layers=[DEFAULT_BARRIER_CONFIG],
        ignore_invalid_costs=False,
    )

    test = _run_lcp(tmp_path, routing_scenario, route_table)

    truth = routing_data_dir / f"least_cost_paths_{capacity}MW.csv"
    truth = pd.read_csv(truth)

    truth = truth.sort_values(["start_index", "index"])
    test = test.sort_values(["start_index", "index"])

    assert (test["cost"].to_numpy() < truth["cost"].to_numpy()).all()


def test_cost_multiplier_layer(
    revx_transmission_layers, route_table, tmp_path, routing_data_dir
):
    """Test routing with a cost_multiplier_layer"""

    capacity = random.choice([100, 200, 400, 1000, 3000])  # noqa: S311
    cap = _cap_class_to_cap(capacity)
    cost_layer = {"layer_name": f"tie_line_costs_{cap}MW"}

    temp_layer_file = tmp_path / "temp_multiplier_layer.zarr"
    shutil.copytree(revx_transmission_layers, temp_layer_file)

    lf = LayeredFile(temp_layer_file)
    lf.write_layer(
        np.ones(lf.shape, dtype=np.float32) * 7, "test_layer", overwrite=True
    )

    routing_scenario = RoutingScenario(
        cost_fpath=temp_layer_file,
        cost_layers=[cost_layer],
        friction_layers=[DEFAULT_BARRIER_CONFIG],
        cost_multiplier_layer="test_layer",
    )
    test = _run_lcp(tmp_path, routing_scenario, route_table)

    truth = routing_data_dir / f"least_cost_paths_{capacity}MW.csv"
    truth = pd.read_csv(truth)

    truth = truth.sort_values(["start_index", "index"])
    test = test.sort_values(["start_index", "index"])

    assert np.allclose(test["length_km"], truth["length_km"])
    assert np.allclose(test["cost"].to_numpy(), truth["cost"].to_numpy() * 7)


def test_cost_multiplier_scalar(
    revx_transmission_layers, route_table, tmp_path, routing_data_dir
):
    """Test routing with a cost_multiplier_scalar"""

    capacity = random.choice([100, 200, 400, 1000, 3000])  # noqa: S311
    cap = _cap_class_to_cap(capacity)
    cost_layer = {"layer_name": f"tie_line_costs_{cap}MW"}

    routing_scenario = RoutingScenario(
        cost_fpath=revx_transmission_layers,
        cost_layers=[cost_layer],
        friction_layers=[DEFAULT_BARRIER_CONFIG],
        cost_multiplier_scalar=5,
    )
    test = _run_lcp(tmp_path, routing_scenario, route_table)

    truth = routing_data_dir / f"least_cost_paths_{capacity}MW.csv"
    truth = pd.read_csv(truth)

    truth = truth.sort_values(["start_index", "index"])
    test = test.sort_values(["start_index", "index"])

    assert np.allclose(test["length_km"], truth["length_km"])
    assert np.allclose(test["cost"].to_numpy(), truth["cost"].to_numpy() * 5)


def test_not_hard_barrier(revx_transmission_layers, tmp_path):
    """Test routing to cut off points using `ignore_invalid_costs=False`"""

    temp_layer_file = tmp_path / "temp_multiplier_layer.zarr"
    shutil.copytree(revx_transmission_layers, temp_layer_file)

    lf = LayeredFile(temp_layer_file)
    costs = np.ones(shape=lf.shape)
    costs[0, 3] = costs[1, 3] = costs[2, 3] = costs[3, 3] = -1
    costs[3, 0] = costs[3, 1] = costs[3, 2] = -1
    lf.write_layer(costs, "test_layer", overwrite=True)

    route_feats = gpd.GeoDataFrame(
        data={"index": [0, 1]},
        geometry=[Point(-70.868065, 40.85588), Point(-71.9096, 42.016506)],
        crs="EPSG:4326",
    ).to_crs(lf.profile["crs"])
    route_points = features_to_route_table(route_feats)
    route_table = map_to_costs(
        route_points,
        crs=lf.profile["crs"],
        transform=lf.profile["transform"],
        shape=lf.shape,
    )
    route_definitions, route_attrs = _convert_to_route_definitions(route_table)

    routing_scenario = RoutingScenario(
        cost_fpath=temp_layer_file,
        cost_layers=[{"layer_name": "test_layer"}],
        friction_layers=[DEFAULT_BARRIER_CONFIG],
        cost_multiplier_layer="test_layer",
        ignore_invalid_costs=True,
    )

    out_fp = tmp_path / "least_cost_paths_barrier.csv"
    assert not out_fp.exists()
    route_computer = BatchRouteProcessor(
        routing_scenario=routing_scenario,
        route_definitions=route_definitions,
        route_attrs=route_attrs,
    )
    route_computer.process(out_fp=out_fp, save_paths=False)
    assert not out_fp.exists()

    routing_scenario = RoutingScenario(
        cost_fpath=temp_layer_file,
        cost_layers=[{"layer_name": "test_layer"}],
        friction_layers=[DEFAULT_BARRIER_CONFIG],
        cost_multiplier_layer="test_layer",
        ignore_invalid_costs=False,
    )

    out_fp = tmp_path / "least_cost_paths_barrier.csv"
    assert not out_fp.exists()
    route_computer = BatchRouteProcessor(
        routing_scenario=routing_scenario,
        route_definitions=route_definitions,
        route_attrs=route_attrs,
    )
    route_computer.process(out_fp=out_fp, save_paths=False)
    assert out_fp.exists()

    test = pd.read_csv(out_fp)
    assert (test["length_km"] > 193).all()


@pytest.mark.parametrize("save_paths", [False, True])
@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_cli(
    revx_transmission_layers,
    route_table,
    tmp_path,
    routing_data_dir,
    save_paths,
    cli_runner,
):
    """Test reVX invariant cost layer routing against known outputs"""

    row_config = polarity_config = cost_layer_config = {}
    test, truth = _run_cli(
        tmp_path,
        routing_data_dir,
        revx_transmission_layers,
        row_config,
        polarity_config,
        route_table,
        cost_layer_config,
        cli_runner,
        save_paths=save_paths,
    )

    check(truth, test)


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_config_given_but_no_mult_in_layers(
    revx_transmission_layers,
    route_table,
    tmp_path,
    routing_data_dir,
    cli_runner,
):
    """Test Least cost path with transmission config but no volt in points"""

    row_config = {"138": 2}
    polarity_config = {"138": {"ac": 2, "dc": 3}}
    cost_layer_config = {}
    test, truth = _run_cli(
        tmp_path,
        routing_data_dir,
        revx_transmission_layers,
        row_config,
        polarity_config,
        route_table,
        cost_layer_config,
        cli_runner,
    )

    check(truth, test)


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_apply_row_mult(
    revx_transmission_layers,
    route_table,
    tmp_path,
    routing_data_dir,
    cli_runner,
):
    """Test applying row multiplier"""

    route_table["voltage"] = 138
    route_table["polarity"] = "dc"

    row_config = {"138": 2}
    polarity_config = {"138": {"ac": 2, "dc": 3}}
    cost_layer_config = {"apply_row_mult": True}
    test, truth = _run_cli(
        tmp_path,
        routing_data_dir,
        revx_transmission_layers,
        row_config,
        polarity_config,
        route_table,
        cost_layer_config,
        cli_runner,
    )
    test["cost"] /= 2

    check(truth, test)


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_apply_polarity_mult(
    revx_transmission_layers,
    route_table,
    tmp_path,
    routing_data_dir,
    cli_runner,
):
    """Test applying polarity multiplier"""

    route_table["voltage"] = 138
    route_table["polarity"] = "dc"

    row_config = {"138": 2}
    polarity_config = {"138": {"ac": 2, "dc": 3}}
    cost_layer_config = {"apply_polarity_mult": True}
    test, truth = _run_cli(
        tmp_path,
        routing_data_dir,
        revx_transmission_layers,
        row_config,
        polarity_config,
        route_table,
        cost_layer_config,
        cli_runner,
    )
    test["cost"] /= 3 * _MILLION_USD_PER_MILE_TO_USD_PER_PIXEL

    check(truth, test)


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_apply_row_and_polarity_mult(
    revx_transmission_layers,
    route_table,
    tmp_path,
    routing_data_dir,
    cli_runner,
):
    """Test applying row multiplier"""

    route_table["voltage"] = 138
    route_table["polarity"] = "dc"

    row_config = {"138": 2}
    polarity_config = {"138": {"ac": 2, "dc": 3}}
    cost_layer_config = {"apply_row_mult": True, "apply_polarity_mult": True}
    test, truth = _run_cli(
        tmp_path,
        routing_data_dir,
        revx_transmission_layers,
        row_config,
        polarity_config,
        route_table,
        cost_layer_config,
        cli_runner,
    )
    test["cost"] /= 6 * _MILLION_USD_PER_MILE_TO_USD_PER_PIXEL

    check(truth, test)


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_apply_row_and_polarity_with_existing_mult(
    revx_transmission_layers,
    route_table,
    tmp_path,
    routing_data_dir,
    cli_runner,
):
    """Test applying both row and polarity multiplier when mult exists"""

    route_table["voltage"] = 138
    route_table["polarity"] = "dc"

    row_config = {"138": 2}
    polarity_config = {"138": {"ac": 2, "dc": 3}}
    cost_layer_config = {
        "multiplier_scalar": 5,
        "apply_row_mult": True,
        "apply_polarity_mult": True,
    }
    test, truth = _run_cli(
        tmp_path,
        routing_data_dir,
        revx_transmission_layers,
        row_config,
        polarity_config,
        route_table,
        cost_layer_config,
        cli_runner,
    )
    test["cost"] /= 30 * _MILLION_USD_PER_MILE_TO_USD_PER_PIXEL

    check(truth, test)


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_apply_multipliers_by_route(
    revx_transmission_layers,
    route_table,
    tmp_path,
    routing_data_dir,
    cli_runner,
):
    """Test applying unique multipliers per route"""

    idx_to_volt = {0: 138, 1: 69, 2: 345, 3: 500}
    idx_to_polarity = {0: "ac", 1: "dc", 2: "ac", 3: "dc", 4: "dc"}
    for idx, volt in idx_to_volt.items():
        mask = route_table["start_index"] == idx
        route_table.loc[mask, "voltage"] = volt
    for idx, polarity in idx_to_polarity.items():
        mask = route_table["start_index"] == idx
        route_table.loc[mask, "polarity"] = polarity

    row_config = {"138": 2, "69": 2.5, "345": 3, "500": 3.5}
    polarity_config = {
        "138": {"ac": 4, "dc": 4.5},
        "69": {"ac": 5, "dc": 5.5},
        "345": {"ac": 6, "dc": 6.5},
        "500": {"ac": 7, "dc": 7.5},
    }
    cost_layer_config = {
        "multiplier_scalar": 1.2,
        "apply_row_mult": True,
        "apply_polarity_mult": True,
    }
    test, truth = _run_cli(
        tmp_path,
        routing_data_dir,
        revx_transmission_layers,
        row_config,
        polarity_config,
        route_table,
        cost_layer_config,
        cli_runner,
    )
    divisors = []
    for __, row in test.iterrows():
        voltage = str(int(row["voltage"]))
        polarity = row["polarity"]
        divisors.append(
            1.2
            * row_config[voltage]
            * polarity_config[voltage][polarity]
            * _MILLION_USD_PER_MILE_TO_USD_PER_PIXEL
        )
    test["cost"] /= divisors

    check(truth, test)


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
