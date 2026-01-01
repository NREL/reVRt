"""reVRt routing module utilities"""

import logging
from pathlib import Path
from warnings import warn

import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from revrt.warn import revrtWarning
from revrt.utilities.base import region_mapper
from revrt.utilities.handlers import IncrementalWriter
from revrt.exceptions import revrtValueError


logger = logging.getLogger(__name__)


class PointToFeatureMapper:
    """Map points to features within specified regions and/or radii"""

    def __init__(
        self,
        crs,
        features_fp,
        regions=None,
        region_identifier_column="rid",
        feature_identifier_column="end_feat_id",
    ):
        """

        Parameters
        ----------
        crs : str or pyproj.crs.CRS
            Coordinate reference system to use for all spatial data.
        features_fp : path-like
            File path to transmission features to map points to.
        regions : geopandas.GeoDataFrame or path-like, optional
            Regions to use for clipping features around points. Features
            will not extend beyond these regions, and point will only
            map to features within their own region. This input can be
            paired with the `radius` parameter (e.g. to forbid any
            connections to cross state boundaries, even if that
            connection would be shorter than some desired length). At
            least one of `regions` or `radius` must be provided;
            otherwise, an error is raised. By default, ``None``.
        region_identifier_column : str, default="rid"
            Column in `regions` data to use as the identifier for that
            region. If not given, a simple index will be put under the
            `rid` column. By default, ``"rid"``.
        feature_identifier_column : str, optional
            Column in output data (both features and points) that will
            be used to link the points to the features that should be
            routed to. By default, ``"end_feat_id"``.
        """
        self._crs = crs
        self._features_fp = features_fp
        self._regions = None
        self._rid_column = region_identifier_column
        self._feature_id_column = feature_identifier_column
        self._set_regions(regions)

    def _set_regions(self, regions):
        """Set the regions GeoDataFrame."""
        if regions is None:
            return

        try:
            self._regions = regions.to_crs(self._crs)
        except AttributeError:
            self._regions = gpd.read_file(regions).to_crs(self._crs)

        if self._rid_column not in self._regions.columns:
            self._regions[self._rid_column] = range(len(self._regions))

    def map_points(
        self,
        points,
        feature_out_fp,
        radius=None,
        expand_radius=True,
        batch_size=500,
    ):
        """Map points to features within the point region

        Parameters
        ----------
        points : geopandas.GeoDataFrame
            Points to map to features.
        feature_out_fp : path-like
            File path to save clipped features to. This should end in
            '.gpkg' to ensure proper format. If not, the extension will
            be added automatically.
        radius : float, optional
            Radius (in CRS units) around each point to clip features to.
            If ``None``, only regions are used for clipping.
            By default, ``None``.
        expand_radius : bool, optional
            If ``True``, the radius is expanded until at least one
            feature is found. By default, ``True``.
        batch_size : int, default=500
            Number of points to process in each batch when writing
            clipped features to file. By default, ``500``.

        Returns
        -------
        geopandas.GeoDataFrame
            Features clipped to the point regions.
        """
        if self._regions is None and radius is None:
            msg = (
                "Must provide either `regions` or a radius to map points "
                "to features!"
            )
            raise revrtValueError(msg)

        points = points.to_crs(self._crs)

        if self._regions is not None:
            map_func = region_mapper(self._regions, self._rid_column)
            points[self._rid_column] = points.centroid.apply(map_func)

        writer = _init_streaming_writer(feature_out_fp)
        batches = []
        for ind, (row_ind, point) in enumerate(points.iterrows()):
            clipped_features = self._clip_to_point(
                point, radius, expand_radius
            )
            clipped_features[self._feature_id_column] = ind
            points.loc[row_ind, self._feature_id_column] = ind
            batches.append(clipped_features)

            if len(batches) >= batch_size:
                logger.debug("Dumping batch of features to file...")
                writer.save(pd.concat(batches))
                batches = []

        if batches:
            logger.debug("Dumping batch of features to file...")
            writer.save(pd.concat(batches))

        return points

    def _clip_to_point(self, point, radius=None, expand_radius=True):
        """Clip features to be within the point region"""
        logger.debug("Clipping features to point:\n%s", point)

        features = None
        if self._regions is not None:
            features = self._clip_to_region(point)

        if radius is not None:
            features = self._clip_to_radius(
                point, radius, features, expand_radius
            )

        return _filter_transmission_features(features)

    def _clip_to_region(self, point):
        """Clip features to region and record the region ID"""
        rid = point[self._rid_column]
        mask = self._regions[self._rid_column] == rid
        region = self._regions[mask]

        logger.debug("  - Clipping features to region: %s", rid)
        features = self._clipped_features(region.geometry, features=None)
        features[self._rid_column] = rid

        logger.debug(
            "%d transmission features found in region with id %s ",
            len(features),
            rid,
        )
        return features

    def _clip_to_radius(
        self, point, radius, input_features=None, expand_radius=True
    ):
        """Clip features to radius

        If no features are found within the initial radius, it is
        expanded (linearly by incrementally increasing the clipping
        buffer) until at least one connection feature is found.
        """
        if radius is None:
            return input_features

        if input_features is not None and len(input_features) == 0:
            return input_features

        clipped_features = self._clipped_features(
            point.geometry.buffer(radius), features=input_features
        )

        clipping_buffer = 1
        while expand_radius and len(clipped_features) <= 0:
            clipping_buffer += 0.05
            clipped_features = self._clipped_features(
                point.geometry.buffer(radius * clipping_buffer),
                input_features,
            )

        logger.info(
            "%d transmission features found in clipped area with radius %.2f",
            len(clipped_features),
            radius * clipping_buffer,
        )
        return clipped_features.copy(deep=True)

    def _clipped_features(self, region, features=None):
        """Clip features to region"""
        if features is None:
            features = gpd.read_file(self._features_fp, mask=region).to_crs(
                self._crs
            )

        return gpd.clip(features, region)


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

    route_points["start_row"] = start_row.astype("int32")
    route_points["start_col"] = start_col.astype("int32")
    route_points["end_row"] = end_row.astype("int32")
    route_points["end_col"] = end_col.astype("int32")

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


def _init_streaming_writer(feature_out_fp):
    """Initialize an incremental writer for clipped features"""
    feature_out_fp = Path(feature_out_fp)
    if feature_out_fp.suffix.lower() != ".gpkg":
        msg = (
            "Output feature file should have a '.gpkg' extension to "
            f"ensure proper format! Got input file: '{feature_out_fp}'. "
            "Adding '.gpkg' extension... "
        )
        warn(msg, revrtWarning)
        feature_out_fp = feature_out_fp.with_suffix(".gpkg")

    return IncrementalWriter(feature_out_fp)


def _filter_transmission_features(features):
    """Filter loaded transmission features"""
    features = features.drop(
        columns=["bgid", "egid", "cap_left"], errors="ignore"
    )
    features = features.rename(columns={"gid": "trans_gid"})
    if "category" in features.columns:
        features = _drop_empty_categories(features)

    return features.reset_index(drop=True)


def _drop_empty_categories(features):
    """Drop features with empty category field"""
    mask = features["category"].isna()
    if mask.any():
        msg = f"Dropping {mask.sum():,} features with NaN category!"
        warn(msg, revrtWarning)
        features = features[~mask].reset_index(drop=True)

    return features
