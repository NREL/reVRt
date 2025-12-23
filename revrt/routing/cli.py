"""reVRt routing CLI functions"""

import time
import logging
import contextlib
from math import ceil
from pathlib import Path
from copy import deepcopy

import pandas as pd
import geopandas as gpd
import xarray as xr
from gaps.cli import CLICommandFromFunction
from gaps.config import load_config

from revrt.costs.config import parse_config
from revrt.routing.point_to_many import (
    find_all_routes,
    RoutingScenario,
    RoutingLayers,
)
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


def compute_lcp_routes(  # noqa: PLR0913, PLR0917
    cost_fpath,
    route_table,
    cost_layers,
    out_dir,
    job_name,
    friction_layers=None,
    tracked_layers=None,
    cost_multiplier_layer=None,
    cost_multiplier_scalar=1,
    transmission_config=None,
    save_paths=False,
    ignore_invalid_costs=False,
    _split_params=None,
):
    r"""Run least-cost path routing for pairs of points

    Parameters
    ----------
    cost_fpath : path-like
        Path to layered Zarr file containing cost and other required
        routing layers.
    route_table : path-like
        Path to CSV file defining the start and
        end points of all routes. Must have the following columns:

            - "start_lat": Stating point latitude
            - "start_lon": Stating point longitude
            - "end_lat": Ending point latitude
            - "end_lon": Ending point longitude

    cost_layers : list
        List of dictionaries defining the layers that are summed to
        determine total costs raster used for routing. Each layer is
        pre-processed before summation according to the user input.
        Each dict in the list should have the following keys:

            - "layer_name": (REQUIRED) Name of layer in layered file
              containing cost data.
            - "multiplier_layer": (OPTIONAL) Name of layer in layered
              file containing spatially explicit multiplier values to
              apply to this cost layer before summing it with the
              others. Default is ``None``.
            - "multiplier_scalar": (OPTIONAL) Scalar value to multiply
              this layer by before summing it with the others. Default
              is ``1``.
            - "is_invariant": (OPTIONAL) Boolean flag indicating whether
              this layer is length invariant (i.e. should NOT be
              multiplied by path length; values should be $). Default is
              ``False``.
            - "include_in_final_cost": (OPTIONAL) Boolean flag
              indicating whether this layer should contribute to the
              final cost output for each route in the LCP table.
              Default is ``True``.
            - "include_in_report": (OPTIONAL) Boolean flag indicating
              whether the costs and distances for this layer should be
              output in the final LCP table. Default is ``True``.
            - "apply_row_mult": (OPTIONAL) Boolean flag indicating
              whether the right-of-way width multiplier should be
              applied for this layer. If ``True``, then the transmission
              config should have a "row_width" dictionary that maps
              voltages to right-of-way width multipliers. Also, the
              routing table input should have a "voltage" entry for
              every route. Every "voltage" value in the routing table
              must be given in the "row_width" dictionary in the
              transmission config, otherwise an error will be thrown.
              Default is ``False``.
            - "apply_polarity_mult": (OPTIONAL) Boolean flag indicating
              whether the polarity multiplier should be applied for this
              layer. If ``True``, then the transmission config should
              have a "voltage_polarity_mult" dictionary that maps
              voltages to a new dictionary, the latter mapping
              polarities to multipliers. For example, a valid
              "voltage_polarity_mult" dictionary might be
              ``{"138": {"ac": 1.15, "dc": 2}}``.
              In addition, the routing table input should have a
              "voltage" **and** a "polarity" entry for every route.
              Every "voltage" + "polarity" combination in the routing
              table must be given in the "voltage_polarity_mult"
              dictionary in the transmission config, otherwise an error
              will be thrown.

              .. IMPORTANT::
                 The multiplier in this config is assumed to be in units
                 of "million $ per mile" and will be converted to
                 "$ per pixel" before being applied to the layer!

              Default is ``False``.

        The summed layers define the cost routing surface, which
        determines the cost output for each route. Specifically, the
        cost at each pixel is multiplied by the length that the route
        takes through the pixel, and all of these values are summed for
        each route to determine the final cost.

        .. IMPORTANT::
           If a pixel has a final cost of :math:`\leq 0`, it is treated
           as a barrier (i.e. no paths can ever cross this pixel).

    out_dir : path-like
        Directory where routing outputs should be written.
    job_name : str
        Label used to name the generated output file.
    friction_layers : list, optional
        Layers to be multiplied onto the aggregated cost layer to
        influence routing but NOT be reported in final cost
        (i.e. friction, barriers, etc.). These layers are first
        aggregated, and then the aggregated friction layer is applied
        to the aggregated cost. The cost at each pixel is therefore
        computed as:

        .. math::

            C = (\sum_{i} c_i) * (1 + \sum_{j} f_j)

        where :math:`C` is the final cost at each pixel, :math:`c_i` are
        the individual cost layers, and :math:`f_j` are the individual
        friction layers.

        .. NOTE:: :math:`\sum_{j} f_j` is always clamped to be
           :math:`\gt -1` to prevent zero or negative routing costs.
           In other words, :math:`(1 + \sum_{j} f_j) > 0` always holds.
           This means friction can scale costs to/away from zero but
           never cause the sign of the cost layer to flip (even if
           friction values themselves are negative). This means all
           "barrier" pixels (i.e. cost value :math:`\leq 0`) will remain
           barriers after friction is applied.

        Each item in this list should be a dictionary containing the
        following keys:

            - "multiplier_layer" or "mask": (REQUIRED) Name of layer in
              layered file containing the spatial friction multipliers
              or mask that will be turned into the friction multipliers
              by applying the `multiplier_scalar`.
            - "multiplier_scalar": (OPTIONAL) Scalar value to multiply
              the spatial friction layer by before using it as a
              multiplier on the aggregated costs. Default is ``1``.
            - "include_in_report": (OPTIONAL) Boolean flag indicating
              whether the routing and distances for this layer should be
              output in the final LCP table. Default is ``False``.
            - "apply_row_mult": (OPTIONAL) Boolean flag indicating
              whether the right-of-way width multiplier should be
              applied for this layer. If ``True``, then the transmission
              config should have a "row_width" dictionary that maps
              voltages to right-of-way width multipliers. Also, the
              routing table input should have a "voltage" entry for
              every route. Every "voltage" value in the routing table
              must be given in the "row_width" dictionary in the
              transmission config, otherwise an error will be thrown.
              Default is ``False``.
            - "apply_polarity_mult": (OPTIONAL) Boolean flag indicating
              whether the polarity multiplier should be applied for this
              layer. If ``True``, then the transmission config should
              have a "voltage_polarity_mult" dictionary that maps
              voltages to a new dictionary, the latter mapping
              polarities to multipliers. For example, a valid
              "voltage_polarity_mult" dictionary might be
              ``{"138": {"ac": 1.15, "dc": 2}}``.
              In addition, the routing table input should have a
              "voltage" **and** a "polarity" entry for every route.
              Every "voltage" + "polarity" combination in the routing
              table must be given in the "voltage_polarity_mult"
              dictionary in the transmission config, otherwise an error
              will be thrown.

              .. IMPORTANT::
                 The multiplier in this config is assumed to be in units
                 of "million $ per mile" and will be converted to
                 "$ per pixel" before being applied to the layer!

              Default is ``False``.

        By default, ``None``.
    tracked_layers : dict, optional
        Dictionary mapping layer names to strings, where the strings are
        dask aggregation methods (similar to what numpy has) that
        should be applied to the layer along the LCP to be included as a
        characterization column in the output. By default, ``None``.
    cost_multiplier_layer : str, optional
        Name of the spatial multiplier layer applied to final costs.
        By default, ``None``.
    cost_multiplier_scalar : int, default=1
        Scalar multiplier applied to the final cost surface.
        By default, ``1``.
    transmission_config : path-like or dict, optional
        Dictionary of transmission cost configuration values, or
        path to JSON/JSON5 file containing this dictionary. The
        dictionary should have a subset of the following keys:

            - base_line_costs
            - iso_lookup
            - iso_multipliers
            - land_use_classes
            - new_substation_costs
            - power_classes
            - power_to_voltage
            - transformer_costs
            - upgrade_substation_costs
            - voltage_polarity_mult
            - row_width

        Each of these keys should point to another dictionary or
        path to JSON/JSON5 file containing a dictionary of
        configurations for each section. For the expected contents
        of each dictionary, see the default config. If ``None``,
        values from the default config are used.
        By default, ``None``.
    save_paths : bool, default=False
        Save outputs as a GeoPackage with path geometries when ``True``.
        Defaults to ``False``.
    ignore_invalid_costs : bool, optional
        Optional flag to treat any cost values <= 0 as impassable
        (i.e. no paths can ever cross this). If ``False``, cost values
        of <= 0 are set to a large value to simulate a strong but
        permeable "quasi-barrier". By default, ``False``.

    Returns
    -------
    str or None
        Path to the output table if any routes were computed.
    """

    start_time = time.time()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.debug("Tracked layers input: %r", tracked_layers)
    logger.debug("Transmission config input: %r", transmission_config)

    transmission_config = parse_config(config=transmission_config)

    route_points = _route_points_subset(
        route_table,
        sort_cols=["start_lat", "start_lon"],
        split_params=_split_params,
    )
    if len(route_points) == 0:
        logger.info("No routes to process!")
        return None

    out_fp = (
        out_dir / f"{job_name}.gpkg"
        if save_paths
        else out_dir / f"{job_name}.csv"
    )

    _run_lcp(
        cost_fpath,
        route_points,
        cost_layers,
        out_fp=out_fp,
        transmission_config=transmission_config,
        cost_multiplier_layer=cost_multiplier_layer,
        cost_multiplier_scalar=cost_multiplier_scalar,
        friction_layers=friction_layers,
        tracked_layers=tracked_layers,
        ignore_invalid_costs=ignore_invalid_costs,
    )

    elapsed_time = (time.time() - start_time) / 60
    logger.info("Processing took %.2f minutes", elapsed_time)

    return str(out_fp)


def build_routing_layer(lcp_config_fp, out_dir, polarity=None, voltage=None):
    """Build out the routing layers used by reVRt

    Parameters
    ----------
    lcp_config_fp : path-like
        Path to LCP config file for which the routing layer should be
        created.
    out_dir : path-like
        Path to directory where to store the outputs.
    polarity : str, optional
        Polarity to use when building the routing layer. This input is
        required if any cost or friction layers that have
        `apply_polarity_mult` set to `True` - they will have the
        appropriate multiplier applied based on this polarity.
        By default, ``None``.
    voltage : str, optional
        Voltage to use when building the routing layer. This input is
        required if any cost or friction layers that have
        `apply_row_mult` or `apply_polarity_mult` set to `True` - they
        will have the appropriate multiplier applied based on this
        voltage. By default, ``None``.

    Returns
    -------
    list
        List of paths to the GeoTIFF files that were created.
    """
    # TODO: Add dask client here??
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(lcp_config_fp)

    route_cl = _update_multipliers(
        config["cost_layers"],
        polarity,
        voltage,
        config.get("transmission_config"),
    )
    route_fl = _update_multipliers(
        config.get("friction_layers") or [],
        polarity,
        voltage,
        config.get("transmission_config"),
    )

    routing_scenario = RoutingScenario(
        cost_fpath=config["cost_fpath"],
        cost_layers=route_cl,
        friction_layers=route_fl,
        cost_multiplier_layer=config.get("cost_multiplier_layer"),
        cost_multiplier_scalar=config.get("cost_multiplier_scalar", 1),
        ignore_invalid_costs=config.get("ignore_invalid_costs", False),
    )

    rl = RoutingLayers(routing_scenario)
    rl.build()

    cost_out_fp = out_dir / "agg_costs.tif"
    logger.debug("\t- Writing costs to %s", cost_out_fp)
    rl.cost.rio.to_raster(cost_out_fp, driver="GTiff", nodata=-1)

    frl_out_fp = out_dir / "final_routing_layer.tif"
    logger.debug("\t- Writing final routing layer to %s", frl_out_fp)
    rl.final_routing_layer.rio.to_raster(frl_out_fp, driver="GTiff", nodata=-1)

    return [str(cost_out_fp), str(frl_out_fp)]


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
        route_cl = _update_multipliers(
            cost_layers, polarity, voltage, transmission_config
        )
        route_fl = _update_multipliers(
            friction_layers or [], polarity, voltage, transmission_config
        )
        route_definitions = [
            (
                [(info["start_row"], info["start_col"])],
                [(info["end_row"], info["end_col"])],
                info.to_dict(),
            )
            for __, info in routes.iterrows()
        ]

        scenario = RoutingScenario(
            cost_fpath=cost_fpath,
            cost_layers=route_cl,
            friction_layers=route_fl,
            tracked_layers=tracked_layers,
            cost_multiplier_layer=cost_multiplier_layer,
            cost_multiplier_scalar=cost_multiplier_scalar,
            ignore_invalid_costs=ignore_invalid_costs,
        )

        find_all_routes(
            scenario,
            route_definitions=route_definitions,
            save_paths=save_paths,
            out_fp=out_fp,
        )

    time_elapsed = f"{(time.monotonic() - ts) / 3600:.4f} hour(s)"
    logger.info(
        "Routing for %d points completed in %s",
        len(route_points),
        time_elapsed,
    )


def _paths_to_compute(route_points, out_fp):
    """Yield route groups that still require computation"""
    existing_routes = _collect_existing_routes(out_fp)

    group_cols = ["start_row", "start_col", "polarity", "voltage"]
    for check_col in ["polarity", "voltage"]:
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

        *__, polarity, voltage = group_info
        yield polarity, voltage, routes


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


def _update_multipliers(layers, polarity, voltage, transmission_config):
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


def _route_points_subset(route_table, sort_cols, split_params):
    """Get indices of points that are sorted by location"""

    with contextlib.suppress(TypeError, UnicodeDecodeError):
        route_points = pd.read_csv(route_table)

    route_points = route_points.sort_values(sort_cols).reset_index(drop=True)

    start_ind, n_chunks = split_params or (0, 1)
    chunk_size = ceil(len(route_points) / n_chunks)
    return route_points.iloc[
        start_ind * chunk_size : (start_ind + 1) * chunk_size
    ]


def _split_routes(config):
    """Compute route split params inside of config"""
    exec_control = config.get("execution_control", {})
    if exec_control.get("option") == "local":
        num_nodes = 1
    else:
        num_nodes = exec_control.pop("nodes", 1)

    config["_split_params"] = [(i, num_nodes) for i in range(num_nodes)]
    return config


route_points_command = CLICommandFromFunction(
    compute_lcp_routes,
    name="route-points",
    add_collect=False,
    split_keys={"_split_params"},
    config_preprocessor=_split_routes,
)

build_route_costs_command = CLICommandFromFunction(
    build_routing_layer, name="build-route-costs", add_collect=False
)
