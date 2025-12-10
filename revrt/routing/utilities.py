"""reVRt routing module utilities"""

import logging
from warnings import warn

import rasterio
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

from revrt.warn import revrtWarning


logger = logging.getLogger(__name__)


def map_to_costs(route_points, crs, transform, shape):
    """Map route table to cost indices and drop out-of-bounds rows

    Parameters
    ----------
    route_points : pandas.DataFrame
        Route definitions table with at least `start_lat`, `start_lon`,
        `end_lat`, and `end_lon` coordinate columns.
    crs : str or pyproj.crs.CRS
        Coordinate reference system for the cost raster.
    transform : affine.Affine
        Rasterio affine transform giving pixel origin and resolution.
    shape : tuple
        Raster height and width for bounds checking.

    Returns
    -------
    pandas.DataFrame
        Updated route table filtered to routes within the cost domain.
    """
    route_points = _get_start_end_point_cost_indices(
        route_points, crs, transform
    )
    return _filter_points_outside_cost_domain(route_points, shape)


def _get_start_end_point_cost_indices(route_points, cost_crs, transform):
    """Populate start/end row and column indices for each route"""

    logger.debug("Map %d routes to cost raster", len(route_points))
    logger.debug("First few routes:\n%s", route_points.head())
    logger.debug("Transform:\n%s", transform)

    start_lat = route_points["start_lat"].astype("float32")
    start_lon = route_points["start_lon"].astype("float32")
    start_row, start_col = _transform_lat_lon_to_row_col(
        transform, cost_crs, start_lat, start_lon
    )
    end_lat = route_points["end_lat"].astype("float32")
    end_lon = route_points["end_lon"].astype("float32")
    end_row, end_col = _transform_lat_lon_to_row_col(
        transform, cost_crs, end_lat, end_lon
    )

    logger.debug("Mapping done!")

    route_points["start_row"] = start_row
    route_points["start_col"] = start_col
    route_points["end_row"] = end_row
    route_points["end_col"] = end_col

    return route_points


def _filter_points_outside_cost_domain(route_points, shape):
    """Drop routes whose indices fall outside the cost domain"""

    logger.debug("Filtering out points outside cost domain...")
    mask = route_points["start_row"] >= 0
    mask &= route_points["start_row"] < shape[0]
    mask &= route_points["start_col"] >= 0
    mask &= route_points["start_col"] < shape[1]
    mask &= route_points["end_row"] >= 0
    mask &= route_points["end_row"] < shape[0]
    mask &= route_points["end_col"] >= 0
    mask &= route_points["end_col"] < shape[1]

    logger.debug("Mask computed!")

    if any(~mask):
        msg = (
            "The following features are outside of the cost exclusion "
            f"domain and will be dropped:\n{route_points.loc[~mask]}"
        )
        warn(msg, revrtWarning)
        route_points = route_points.loc[mask].reset_index(drop=True)

    return route_points


def _transform_lat_lon_to_row_col(transform, cost_crs, lat, lon):
    """Convert WGS84 coordinates to cost grid row and column arrays"""
    feats = gpd.GeoDataFrame(
        geometry=[Point(*p) for p in zip(lon, lat, strict=True)]
    )
    coords = feats.set_crs("EPSG:4326").to_crs(cost_crs)["geometry"].centroid
    row, col = rasterio.transform.rowcol(
        transform, coords.x.values, coords.y.values
    )
    row = np.array(row)
    col = np.array(col)
    return row, col
