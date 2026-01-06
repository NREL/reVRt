"""reVRt routing CLI functions and helpers"""

import time
import logging
from pathlib import Path
from copy import deepcopy

import pandas as pd
import geopandas as gpd
import xarray as xr

from revrt.routing.point_to_point import BatchRouteProcessor, RoutingScenario
from revrt.routing.utilities import map_to_costs
from revrt.exceptions import revrtKeyError


logger = logging.getLogger(__name__)
_MILLION_USD_PER_MILE_TO_USD_PER_PIXEL = 55923.40730136006
"""Conversion from million dollars/mile to $/pixel

1,000,000 [$/million dollars]
* 90 [meters/pixel]
/ 1609.344 [meters/mile]
= 55923.40730136006 [$/pixel]
"""


def _run_lcp(
    cost_fpath,
    route_points,
    cost_layers,
    out_fp,
    transmission_config=None,
    cost_multiplier_layer=None,
    cost_multiplier_scalar=1,
    friction_layers=None,
    tracked_layers=None,
    ignore_invalid_costs=True,
):
    """Execute least-cost path routing for the prepared route subset"""

    ts = time.monotonic()
    out_fp = Path(out_fp)
    save_paths = out_fp.suffix.lower() == ".gpkg"

    if route_points.empty:
        logger.info("Found no routes to compute!")
        return

    with xr.open_dataset(cost_fpath, consolidated=False, engine="zarr") as ds:
        route_points = map_to_costs(
            route_points,
            crs=ds.rio.crs,
            transform=ds.rio.transform(),
            shape=ds.rio.shape,
        )

    logger.info("Computing best routes for %d point pairs", len(route_points))
    for polarity, voltage, routes in _paths_to_compute(route_points, out_fp):
        logger.info(
            "Computing routes for %d points with polarity: %r and voltage: %r",
            len(routes),
            polarity,
            voltage,
        )
        route_cl = update_multipliers(
            cost_layers, polarity, voltage, transmission_config
        )
        route_fl = update_multipliers(
            friction_layers or [], polarity, voltage, transmission_config
        )
        route_definitions, route_attrs = _convert_to_route_definitions(routes)

        scenario = RoutingScenario(
            cost_fpath=cost_fpath,
            cost_layers=route_cl,
            friction_layers=route_fl,
            tracked_layers=tracked_layers,
            cost_multiplier_layer=cost_multiplier_layer,
            cost_multiplier_scalar=cost_multiplier_scalar,
            ignore_invalid_costs=ignore_invalid_costs,
        )

        route_computer = BatchRouteProcessor(
            routing_scenario=scenario,
            route_definitions=route_definitions,
            route_attrs=route_attrs,
        )
        route_computer.process(out_fp=out_fp, save_paths=save_paths)

    time_elapsed = f"{(time.monotonic() - ts) / 3600:.4f} hour(s)"
    logger.info(
        "Routing for %d points completed in %s",
        len(route_points),
        time_elapsed,
    )


def _paths_to_compute(route_points, out_fp):
    """Yield route groups that still require computation"""
    existing_routes = _collect_existing_routes(out_fp)

    group_cols = ["polarity", "voltage"]
    for check_col in group_cols:
        if check_col not in route_points.columns:
            route_points[check_col] = "unknown"

    for group_info, routes in route_points.groupby(group_cols):
        if existing_routes:
            mask = routes.apply(
                lambda row: (
                    int(row["start_row"]),
                    int(row["start_col"]),
                    int(row["end_row"]),
                    int(row["end_col"]),
                    str(row.get("polarity", "unknown")),
                    str(row.get("voltage", "unknown")),
                )
                not in existing_routes,
                axis=1,
            )
            routes = routes[mask]  # noqa: PLW2901

        if routes.empty:
            continue

        yield *group_info, routes


def _convert_to_route_definitions(routes):
    """Convert route DataFrame to route definitions format"""
    start_point_cols = ["start_row", "start_col"]
    end_point_cols = ["end_row", "end_col"]
    num_unique_start_points = len(routes.groupby(start_point_cols))
    num_unique_end_points = len(routes.groupby(end_point_cols))
    if num_unique_end_points > num_unique_start_points:
        logger.info(
            "Less unique starting points detected! Swapping start and "
            "end point set for optimal routing performance"
        )
        start_point_cols = ["end_row", "end_col"]
        end_point_cols = ["start_row", "start_col"]

    route_definitions = []
    route_attrs = {}
    for route_id, (end_idx, sub_routes) in enumerate(
        routes.groupby(end_point_cols)
    ):
        start_points = []
        for __, info in sub_routes.iterrows():
            start_idx = tuple(info[start_point_cols].astype("int32"))
            route_attrs[(route_id, start_idx)] = info.to_dict()
            start_points.append(start_idx)

        route_definitions.append(
            (route_id, start_points, [tuple(map(int, end_idx))])
        )

    return route_definitions, route_attrs


def _collect_existing_routes(out_fp):
    """Collect already computed routes from an existing output file"""

    if out_fp is None or not out_fp.exists():
        return set()

    if out_fp.suffix.lower() == ".gpkg":
        existing_df = gpd.read_file(out_fp)
    else:
        existing_df = pd.read_csv(out_fp)

    return {
        (
            int(row["start_row"]),
            int(row["start_col"]),
            int(row["end_row"]),
            int(row["end_col"]),
            str(row.get("polarity", "unknown")),
            str(row.get("voltage", "unknown")),
        )
        for __, row in existing_df.iterrows()
    }


def update_multipliers(layers, polarity, voltage, transmission_config):
    """Update layer multipliers based on user input"""
    output_layers = deepcopy(layers)
    polarity = "unknown" if polarity in {None, "unknown"} else str(polarity)
    voltage = "unknown" if voltage in {None, "unknown"} else str(int(voltage))

    for layer in output_layers:
        if layer.pop("apply_row_mult", False):
            row_multiplier = _get_row_multiplier(transmission_config, voltage)
            layer["multiplier_scalar"] = (
                layer.get("multiplier_scalar", 1) * row_multiplier
            )

        if layer.pop("apply_polarity_mult", False):
            polarity_multiplier = _get_polarity_multiplier(
                transmission_config, voltage, polarity
            )
            layer["multiplier_scalar"] = (
                layer.get("multiplier_scalar", 1)
                * polarity_multiplier
                * _MILLION_USD_PER_MILE_TO_USD_PER_PIXEL
            )

    return output_layers


def _get_row_multiplier(transmission_config, voltage):
    """Get right-of-way width multiplier for a given voltage"""
    try:
        row_widths = transmission_config["row_width"]
    except KeyError as e:
        msg = (
            "`apply_row_mult` was set to `True`, but 'row_width' "
            "not found in transmission config!"
        )
        raise revrtKeyError(msg) from e

    try:
        row_multiplier = row_widths[voltage]
    except KeyError as e:
        msg = (
            "`apply_row_mult` was set to `True`, but voltage ' "
            f"{voltage}' not found in transmission config "
            "'row_width' settings. Available voltages: "
            f"{list(row_widths)}"
        )
        raise revrtKeyError(msg) from e

    return row_multiplier


def _get_polarity_multiplier(transmission_config, voltage, polarity):
    """Get multiplier for a given voltage and polarity"""
    try:
        polarity_config = transmission_config["voltage_polarity_mult"]
    except KeyError as e:
        msg = (
            "`apply_polarity_mult` was set to `True`, but "
            "'voltage_polarity_mult' not found in transmission config!"
        )
        raise revrtKeyError(msg) from e

    try:
        polarity_voltages = polarity_config[voltage]
    except KeyError as e:
        msg = (
            "`apply_polarity_mult` was set to `True`, but voltage ' "
            f"{voltage}' not found in polarity config. Available voltages: "
            f"{list(polarity_config)}"
        )
        raise revrtKeyError(msg) from e

    try:
        polarity_multiplier = polarity_voltages[polarity]
    except KeyError as e:
        msg = (
            "`apply_polarity_mult` was set to `True`, but polarity ' "
            f"{polarity}' not found in voltage config. Available polarities: "
            f"{list(polarity_voltages)}"
        )
        raise revrtKeyError(msg) from e

    return polarity_multiplier
