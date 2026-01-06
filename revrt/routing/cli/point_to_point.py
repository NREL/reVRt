"""reVRt point-to-point routing CLI command"""

import time
import logging
import contextlib
from math import ceil
from pathlib import Path

import pandas as pd
from gaps.cli import CLICommandFromFunction

from revrt.routing.cli.base import _run_lcp
from revrt.costs.config import parse_config


logger = logging.getLogger(__name__)


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
