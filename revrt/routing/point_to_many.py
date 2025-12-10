"""reVRt routing from a point to many points"""

import json
import time
import logging
from warnings import warn
from functools import cached_property

import rasterio
import numpy as np
import xarray as xr
import dask.array as da
from shapely.geometry import Point
from shapely.geometry.linestring import LineString

from revrt import find_paths

from revrt.exceptions import (
    revrtKeyError,
    revrtLeastCostPathNotFoundError,
)
from revrt.warn import revrtWarning

logger = logging.getLogger(__name__)
LCP_AGG_COST_LAYER_NAME = "lcp_agg_costs"
"""Special name reserved for internally-built cost layer"""


class RoutingScenario:
    """Container for routing scenario configuration"""

    def __init__(
        self,
        cost_fpath,
        cost_layers,
        friction_layers=None,
        tracked_layers=None,
        cost_multiplier_layer=None,
        cost_multiplier_scalar=1,
        use_hard_barrier=True,
    ):
        """

        Parameters
        ----------
        cost_fpath : path-like
            Path to the cost layer Zarr store used for routing.
        cost_layers : list
            List of dictionaries containing layer definitions
            contributing to the summed routing cost.
        friction_layers : list, optional
            List of dictionaries defining layers that influence routing
            but are excluded from reports.
        tracked_layers : dict, optional
            Layers to summarize along the path, mapped to aggregation
            names.
        cost_multiplier_layer : str, optional
            Layer name providing spatial multipliers for total cost.
        cost_multiplier_scalar : int or float, optional
            Scalar multiplier applied to the final cost surface.
        use_hard_barrier : bool, optional
            Flag indicating whether non-positive costs block traversal.
        """
        self.cost_fpath = cost_fpath
        self.cost_layers = cost_layers
        self.friction_layers = friction_layers or []
        self.tracked_layers = tracked_layers or {}
        self.cost_multiplier_layer = cost_multiplier_layer
        self.cost_multiplier_scalar = cost_multiplier_scalar
        self.use_hard_barrier = use_hard_barrier

    def __repr__(self):
        return (
            "RoutingScenario:"
            f"\n\t- cost_layers: {self.cost_layers}"
            f"\n\t- friction_layers: {self.friction_layers}"
            f"\n\t- cost_multiplier_layer: {self.cost_multiplier_layer}"
            f"\n\t- cost_multiplier_scalar: {self.cost_multiplier_scalar}"
        )

    @cached_property
    def cl_as_json(self):
        """str: JSON string describing configured cost layers"""
        return json.dumps({"cost_layers": self.cost_layers})


class RoutingLayers:
    """Class to build a routing layer from user input"""

    SOFT_BARRIER_MULTIPLIER = 100
    """Multiplier to apply to max cost to use for barriers

    This value is only used if ``use_hard_barrier=False``.
    """

    def __init__(self, routing_scenario, chunks="auto"):
        """

        Parameters
        ----------
        routing_scenario : RoutingScenario
            Scenario containing cost, friction, and tracking metadata.
        chunks : str or mapping, default="auto"
            Chunk specification forwarded to ``xarray.open_dataset``.
            By default, ``"auto"``.
        """
        self.routing_scenario = routing_scenario
        self._layer_fh = xr.open_dataset(
            self.routing_scenario.cost_fpath,
            chunks=chunks,
            consolidated=False,
            engine="zarr",
        )
        self.li_cost_layer_map = {}
        self.tracked_layers = []

        self.transform = self._layer_fh.rio.transform()
        self._full_shape = self._layer_fh.rio.shape
        self.cost_crs = self._layer_fh.rio.crs
        self._layers = set(self._layer_fh.variables)

        self.cost = None
        self.li_cost = None
        self.final_routing_layer = None

    def __repr__(self):
        return f"RoutingLayers for {self.routing_scenario!r}"

    @property
    def latitudes(self):
        """xarray.DataArray: Latitude coordinates for the cost grid"""
        return self._layer_fh["latitude"]

    @property
    def longitudes(self):
        """xarray.DataArray: Longitude coordinates for the cost grid"""
        return self._layer_fh["longitude"]

    def _verify_layer_exists(self, layer_name):
        """Verify that layer exists in cost file"""
        if layer_name not in self._layers:
            msg = (
                f"Did not find layer {layer_name!r} in cost "
                f"file {str(self.routing_scenario.cost_fpath)!r}"
            )
            raise revrtKeyError(msg)

    def close(self):
        """Close the underlying xarray file handle"""
        self._layer_fh.close()

    def build(self):
        """Build lazy routing arrays from layered file"""

        logger.debug("Building %r", self)
        self._build_cost_layer()
        self._build_final_routing_layer()
        self._build_tracked_layers()

        return self

    def _build_cost_layer(self):
        """Build out the main cost layer"""
        self.cost = da.zeros(self._full_shape, dtype=np.float32)
        self.li_cost = da.zeros(self._full_shape, dtype=np.float32)
        for layer_info in self.routing_scenario.cost_layers:
            layer_name = layer_info["layer_name"]
            is_li = layer_info.get("is_invariant", False)
            cost = self._extract_and_scale_layer(layer_info)

            if is_li:
                self.li_cost += cost
                self.li_cost_layer_map[layer_name] = cost
            else:
                self.cost += cost

            if layer_info.get("include_in_report", True):
                self.tracked_layers.append(
                    CharacterizedLayer(
                        layer_name, cost, is_length_invariant=is_li
                    )
                )

        if mll := self.routing_scenario.cost_multiplier_layer:
            self._verify_layer_exists(mll)
            self.cost *= self._layer_fh[mll].isel(band=0).astype(np.float32)

        self.cost *= self.routing_scenario.cost_multiplier_scalar
        self.li_cost += self.cost * 0

    def _build_final_routing_layer(self):
        """Build out the routing array"""
        self.final_routing_layer = self.cost.copy()
        self.final_routing_layer += self.li_cost

        friction_costs = da.zeros(self._full_shape, dtype=np.float32)
        for layer_info in self.routing_scenario.friction_layers:
            layer_name = layer_info["layer_name"]
            friction_layer = self._extract_and_scale_layer(
                layer_info, allow_cl=True
            )
            if layer_info.get("include_in_report", False):
                self.tracked_layers.append(
                    CharacterizedLayer(layer_name, friction_layer)
                )

            friction_costs += friction_layer

        # Must happen at end of loop so that "lcp_agg_cost"
        # remains constant
        self.final_routing_layer += friction_costs

        max_val = (
            da.max(self.final_routing_layer) * self.SOFT_BARRIER_MULTIPLIER
        )
        self.final_routing_layer = da.where(
            self.final_routing_layer <= 0,
            -1 if self.routing_scenario.use_hard_barrier else max_val,
            self.final_routing_layer,
        ) + (self.cost * 0)

    def _extract_and_scale_layer(self, layer_info, allow_cl=False):
        """Extract layer based on name and scale according to input"""
        cost = self._extract_layer(layer_info["layer_name"], allow_cl=allow_cl)

        multiplier_layer_name = layer_info.get("multiplier_layer")
        if multiplier_layer_name:
            cost *= self._extract_layer(
                multiplier_layer_name, allow_cl=allow_cl
            )

        cost *= layer_info.get("multiplier_scalar", 1)
        return cost

    def _extract_layer(self, layer_name, allow_cl=False):
        """Extract layer based on name"""
        if allow_cl and layer_name == LCP_AGG_COST_LAYER_NAME:
            return self.final_routing_layer.copy()

        self._verify_layer_exists(layer_name)
        return self._layer_fh[layer_name].isel(band=0).astype(np.float32)

    def _build_tracked_layers(self):
        """Build out a dictionary of tracked layers"""
        for (
            tracked_layer,
            method,
        ) in self.routing_scenario.tracked_layers.items():
            if getattr(da, method, None) is None:
                msg = (
                    f"Did not find method {method!r} in dask.array! "
                    f"Skipping tracked layer {tracked_layer!r}"
                )
                warn(msg, revrtWarning)
                continue

            if tracked_layer not in self._layers:
                msg = (
                    f"Did not find layer {tracked_layer!r} in cost file "
                    f"{str(self.routing_scenario.cost_fpath)!r}. "
                    "Skipping..."
                )
                warn(msg, revrtWarning)
                continue

            layer = (
                self._layer_fh[tracked_layer].isel(band=0).astype(np.float32)
            )
            self.tracked_layers.append(
                CharacterizedLayer(tracked_layer, layer, agg_method=method)
            )


class CharacterizedLayer:
    """Encapsulate tracked routing layer metadata"""

    def __init__(
        self, name, layer, is_length_invariant=False, agg_method=None
    ):
        """

        Parameters
        ----------
        name : str
            Identifier used when reporting layer-derived metrics.
        layer : xarray.DataArray or dask.array.Array
            Data values sampled from the routing stack.
        is_length_invariant : bool, default=False
            Flag signaling cost values should ignore segment length.
            By default, ``False``.
        agg_method : str, optional
            Name of dask aggregation used when tracking statistics.
            By default, ``None``.
        """
        self.name = name
        self.layer = layer
        self.is_length_invariant = is_length_invariant
        self.agg_method = agg_method

    def __repr__(self):
        return (
            f"CharacterizedLayer(name={self.name!r}, "
            f"is_length_invariant={self.is_length_invariant}, "
            f"agg_method={self.agg_method!r})"
        )

    def compute(self, route, cell_size):
        """Compute layer metrics along a route

        Parameters
        ----------
        route : sequence
            Ordered ``(row, col)`` indices describing the path.
        cell_size : float
            Raster cell size in meters for distance calculations.

        Returns
        -------
        dict
            Mapping of metric names to aggregated layer values.
        """
        rows, cols = np.array(route).T
        layer_values = self.layer.isel(
            y=xr.DataArray(rows, dims="points"),
            x=xr.DataArray(cols, dims="points"),
        )

        if self.agg_method is None:
            return self._compute_total_and_length(
                layer_values, route, cell_size
            )

        return self._compute_agg(layer_values)

    def _compute_total_and_length(self, layer_values, route, cell_size):
        """Compute total cost and length metrics for the layer"""
        lens, __ = _compute_lens(route, cell_size)

        layer_data = getattr(layer_values, "data", layer_values)
        if not isinstance(layer_data, da.Array):
            layer_data = da.asarray(layer_data)

        if self.is_length_invariant:
            layer_cost = da.sum(layer_data)
        else:
            layer_cost = da.sum(layer_data * lens)

        layer_length = da.sum(lens[layer_data > 0]) * cell_size / 1000

        return {
            f"{self.name}_cost": layer_cost.astype(np.float32).compute(),
            f"{self.name}_dist_km": (
                layer_length.astype(np.float32).compute()
            ),
        }

    def _compute_agg(self, layer_values):
        """Compute aggregated statistic for tracked layer"""
        aggregate = getattr(da, self.agg_method)(layer_values).astype(
            np.float32
        )
        return {f"{self.name}_{self.agg_method}": aggregate.compute()}


class RouteResult:
    """Class to compute route characteristics given layer cost maps"""

    def __init__(
        self,
        routing_layers,
        route,
        optimized_objective,
        add_geom=False,
        attrs=None,
    ):
        """

        Parameters
        ----------
        routing_layers : RoutingLayers
            Routing layer manager containing cost and tracker arrays.
        route : list
            Ordered row and column indices defining the path.
        optimized_objective : float
            Objective value returned by the routing solver.
        add_geom : bool, default=False
            Include shapely geometry in the output when ``True``.
            By default, ``False``.
        attrs : dict, optional
            Additional metadata merged into the result dictionary.
            By default, ``None``.
        """
        self._routing_layers = routing_layers
        self._route = route
        self._optimized_objective = optimized_objective
        self.__lens = self._total_path_length = None
        self._by_layer_results = {}
        self._add_geom = add_geom
        self._attrs = attrs or {}

    @property
    def cell_size(self):
        """float: Raster cell size in meters"""
        return abs(self._routing_layers.transform.a)

    @property
    def _lens(self):
        """array-like: Cached per-cell travel distances"""
        if self.__lens is None:
            self._compute_path_length()
        return self.__lens

    @property
    def total_path_length(self):
        """float: Total path length in kilometers"""
        if self._total_path_length is None:
            self._compute_path_length()
        return self._total_path_length

    @property
    def cost(self):
        """float: Optimized objective evaluated over the route"""
        rows, cols = np.array(self._route).T
        cell_costs = self._routing_layers.cost.isel(
            y=xr.DataArray(rows, dims="points"),
            x=xr.DataArray(cols, dims="points"),
        )

        # Multiple distance travel through cell by cost and sum it!
        return da.sum(cell_costs * self._lens).astype(np.float32).compute()

    @property
    def end_lat(self):
        """float: Latitude of the terminal path cell"""
        row, col = self._route[-1]
        return (
            self._routing_layers.latitudes.isel(y=row, x=col)
            .astype(np.float32)
            .compute()
            .item()
        )

    @property
    def end_lon(self):
        """float: Longitude of the terminal path cell"""
        row, col = self._route[-1]
        return (
            self._routing_layers.longitudes.isel(y=row, x=col)
            .astype(np.float32)
            .compute()
            .item()
        )

    @property
    def geom(self):
        """shapely.GeometryType: Geometry for the computed path"""
        rows, cols = np.array(self._route).T
        x, y = rasterio.transform.xy(
            self._routing_layers.transform, rows, cols
        )
        geom = Point if len(self._route) == 1 else LineString
        return geom(list(zip(x, y, strict=True)))

    def build(self):
        """Assemble route metrics and optional geometry payload"""
        results = {
            "length_km": self.total_path_length,
            "cost": self.cost,
            "poi_lat": self.end_lat,
            "poi_lon": self.end_lon,
            "start_row": self._route[0][0],
            "start_col": self._route[0][1],
            "end_row": self._route[-1][0],
            "end_col": self._route[-1][1],
            "optimized_objective": self._optimized_objective,
        }
        for check_key in ["start_row", "start_col", "end_row", "end_col"]:
            if (
                check_key in self._attrs
                and results[check_key] != self._attrs[check_key]
            ):
                msg = (
                    f"Computed {check_key}={results[check_key]} does "
                    f"not match expected {check_key}="
                    f"{self._attrs[check_key]}!"
                )
                warn(msg, revrtWarning)

        results.update(self._attrs)
        for layer in self._routing_layers.tracked_layers:
            layer_result = layer.compute(self._route, self.cell_size)
            results.update(layer_result)

        if self._add_geom:
            results["geometry"] = self.geom

        return results

    def _compute_path_length(self):
        """Compute the total length and cell by cell length of LCP"""
        self.__lens, self._total_path_length = _compute_lens(
            self._route, self.cell_size
        )


def find_all_routes(routing_scenario, route_definitions, save_paths=False):
    """Compute least-cost routes for each start and destination set

    Parameters
    ----------
    routing_scenario : RoutingScenario
        Scenario describing the cost layers and routing options.
    route_definitions : Iterable
        Sequence of ``(start_point, end_points, attrs)`` tuples
        defining which points to route between. The `attrs` dictionary
        will be appended to the output for each route.
    save_paths : bool, default=False
        Include shapely geometries in the output when ``True``.
        By default, ``False``.

    Returns
    -------
    list
        Route summaries for each successfully computed path.
    """
    if not route_definitions:
        return []

    ts = time.monotonic()

    routing_layers = RoutingLayers(routing_scenario).build()
    try:
        routes = _compute_routes(
            routing_scenario,
            route_definitions,
            routing_layers,
            save_paths=save_paths,
        )
    finally:
        routing_layers.close()

    time_elapsed = f"{(time.monotonic() - ts) / 60:.4f} min"
    logger.debug("Least Cost tie-line computed in %s", time_elapsed)

    return routes


def _compute_routes(
    routing_scenario, route_definitions, routing_layers, save_paths
):
    """Evaluate route definitions and build result records"""
    routes = []
    for start_point, end_points, attrs in route_definitions:
        try:
            indices, optimized_objective = _compute_valid_path(
                routing_scenario, routing_layers, start_point, end_points
            )
        except revrtLeastCostPathNotFoundError:
            continue

        route = RouteResult(
            routing_layers,
            indices,
            optimized_objective,
            add_geom=save_paths,
            attrs=attrs,
        )
        routes.append(route.build())

    return routes


def _compute_valid_path(
    routing_scenario, routing_layers, start_point, end_points
):
    """Validate provided indices then solve for the least-cost path"""
    _validate_starting_point(routing_layers, start_point)
    _validate_end_points(routing_layers, end_points)

    try:
        route_result = find_paths(
            zarr_fp=routing_scenario.cost_fpath,
            cost_layers=routing_scenario.cl_as_json,
            start=[start_point],
            end=end_points,
        )[0]
    except Exception as ex:
        msg = (
            f"Unable to find path from {start_point} any of {end_points}: {ex}"
        )
        logger.exception(msg)

        raise revrtLeastCostPathNotFoundError(msg) from ex

    return route_result


def _validate_starting_point(routing_layers, start_point):
    """Raise when the starting cell lacks a positive traversal cost"""
    start_row, start_col = start_point
    start_cost = (
        routing_layers.final_routing_layer.isel(y=start_row, x=start_col)
        .compute()
        .item()
    )

    if start_cost <= 0:
        msg = (
            f"Start idx {start_point} does not have a valid cost: "
            f"{start_cost:.2f} (must be > 0)!"
        )
        raise revrtLeastCostPathNotFoundError(msg)


def _validate_end_points(routing_layers, end_points):
    """Raise when no end cell provides a positive traversal cost"""
    end_rows, end_cols = np.array(end_points).T
    end_costs = routing_layers.final_routing_layer.isel(
        y=xr.DataArray(end_rows, dims="points"),
        x=xr.DataArray(end_cols, dims="points"),
    )
    if not np.any(end_costs.compute() > 0):
        msg = (
            f"None of the end idx {end_points} have a valid cost "
            f"(must be > 0)!"
        )
        raise revrtLeastCostPathNotFoundError(msg)


def _compute_lens(route, cell_size):
    """Compute the total length and cell by cell length of LCP"""
    # Use Pythagorean theorem to calculate length between cells (km)
    # Use c**2 = a**2 + b**2 to determine length of individual paths
    lens = np.sqrt(np.sum(np.diff(route, axis=0) ** 2, axis=1))
    total_path_length = np.sum(lens) * cell_size / 1000

    # Need to determine distance coming into and out of any cell.
    # Assume paths start and end at the center of a cell. Therefore,
    # distance traveled in the cell is half the distance entering it
    # and half the distance exiting it. Duplicate all lengths,
    # pad 0s on ends for start  and end cells, and divide all
    # distance by half.
    lens = np.repeat(lens, 2)
    lens = np.insert(np.append(lens, 0), 0, 0)
    lens /= 2

    # Group entrance and exits distance together, and add them
    lens = lens.reshape((int(lens.shape[0] / 2), 2))
    lens = np.sum(lens, axis=1)
    return lens, total_path_length
