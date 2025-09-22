"""revrt zonal characterization module"""

import logging
import warnings
from functools import cached_property

import dask
import rasterio
import rasterio.errors
import rioxarray
import geopandas as gpd

from revrt.spatial_characterization.stats import ComputableStats
from revrt.exceptions import revrtTypeError


_GPKG_GEOM_COL = "geometry"
logger = logging.getLogger(__name__)


class ZonalStats:
    """Class to compute zonal statistics"""

    def __init__(
        self,
        stats=None,
        nodata=None,
        all_touched=True,
        category_map=None,
        add_stats=None,
        zone_func=None,
    ):
        """

        Parameters
        ----------
        stats : str | iterable of str, optional
            Names of all statistics to compute. Statistics must be one
            of the members of
            :class:`~revrt.spatial_characterization.stats.Stat` or
            :class:`~revrt.spatial_characterization.stats.FractionalStat`,
            or must start with the ``percentile_`` prefix and end with
            an int or float representing the percentile to compute (e.g.
            ``percentile_10.5``). If only one statistic is to be
            computed, you can provide it directly as a string.
            Otherwise, provide a list of statistic names or a string
            with the names separated by a space. You can also provide
            the string ``"ALL"`` or ``"*"`` to specify that all
            statistics should be computed (i.e. all options from *both*
            :class:`~revrt.spatial_characterization.stats.Stat` and
            :class:`~revrt.spatial_characterization.stats.FractionalStat`).
            If no input, empty input, or ``None`` is provided, then only
            the base stats ("count", "min", "max", "mean") are
            configured. To summarize, all of the following are valid
            inputs:

                - ``stats="*"`` or ``stats="ALL"`` or ``stats="All"``
                - ``stats="min"``
                - ``stats="min max"``
                - ``stats=["min"]``
                - ``stats=["min", "max", "percentile_10.5"]``

            By default, ``None``.
        nodata : int | float, optional
            Value in the raster that represents `nodata`. This value
            will not show up in any statistics except for the `nodata`
            statistic itself, which computes the number of `nodata`
            values within the zone. Note that this value is used **in
            addition to** any `NODATA` value in the raster's metadata.
            By default, ``None``.
        all_touched : bool, optional
            Flag indicating whether to include every raster cell touched
            by a geometry (``True``), or only those having a center
            point within the polygon (``False``). By default, ``True``.
        category_map : dict, optional
            Dictionary mapping raster values to new names. If given,
            this mapping will be applied to the pixel count dictionary,
            so you can use it to map raster values to human-readable
            category names. By default, ``None``.
        add_stats : dict, optional
            Dictionary mapping extra stat names to callable functions
            that can be used to compute that stat. The functions must
            take exactly three arguments as input:

                - ``processed_raster``: Array representing the
                                        zone-masked raster.
                - ``feat``: Pandas Series object containing information
                            about the zone.
                - ``rasterized_zone``: Unit8 array of the same shape
                                       as ``processed_raster``
                                       representing the rasterized zone.

            By default, ``None``.
        zone_func : callable, optional
            Callable function to apply to the zone-masked raster array
            prior to computing stats. Must take exactly one input: the
            zone-masked raster array. By default, ``None``
        """
        self.nodata = nodata
        self.all_touched = all_touched
        self._category_map = category_map or {}
        self.add_stats = add_stats
        self.zone_func = zone_func
        self._stats_input = stats

    @cached_property
    def computable_stats(self):
        """:class:`~revrt.spatial_characterization.stats.ComputableStats`"""
        return ComputableStats.from_iter(self._stats_input)

    @cached_property
    def category_map(self):
        """dict: Map of values to category names"""
        out = {}
        out.update(self._category_map)
        for k, v in self._category_map.items():
            try:
                float_key = float(k)
            except ValueError:
                continue
            out[float_key] = v
        return out

    def from_files(
        self,
        zones_fp,
        raster_fp,
        prefix=None,
        copy_properties=None,
        parallel=False,
    ):
        """Compute zonal statistics for data stored in files

        Parameters
        ----------
        zones_fp : path-like
            Path to GeoPackage defining zone polygons.
        raster_fp : path-like
            Path to GeoTiff representing the data that should be used
            for statistics computations.
        prefix : str, optional
            A string representing a prefix to add to each stat name. If
            you wish to have the prefix separated by a delimiter, you
            must include it in this string (e.g. ``prefix="test_"``).
            By default, ``None``.
        copy_properties : iterable of str, optional
            Iterable of columns names to copy over from the zone
            feature. By default, ``None``.
        parallel : bool, optional
            Option to perform processing in parallel using dask.
            By default, ``False``.

        Returns
        -------
        list
            List of dictionaries, each of which containing computed
            statistics for a zone.
        """
        zones = gpd.read_file(zones_fp)
        rds = rioxarray.open_rasterio(raster_fp, chunks="auto")
        return list(
            self.from_array(
                zones,
                rds,
                rds.rio.transform(),
                prefix=prefix,
                copy_properties=copy_properties,
                parallel=parallel,
            )
        )

    def from_array(
        self,
        zones,
        raster_array,
        affine_transform,
        prefix=None,
        copy_properties=None,
        parallel=False,
    ):
        """Compute zonal statistics for data in memory

        Parameters
        ----------
        zones : :class:`geopandas.GeoDataFrame`
            GeoDataFrame containing the zone polygons.
        raster_array : :class:`xarray.DataArray`
            Xarray DataArray representing the data that should be used
            to compute statistics.
        affine_transform : affine.Affine
            Affine transform object representing the raster
            transformation. This is used to compute the raster shapes
            for statistics computations as well as the pixel area
            for each pixel value in the raster.
        prefix : str, optional
            A string representing a prefix to add to each stat name. If
            you wish to have the prefix separated by a delimiter, you
            must include it in this string (e.g. ``prefix="test_"``).
            By default, ``None``.
        copy_properties : iterable of str, optional
            Iterable of columns names to copy over from the zone
            feature. By default, ``None``.
        parallel : bool, optional
            Option to perform processing in parallel using dask.
            By default, ``False``.

        Yields
        ------
        dict
            Dictionary of statistics computed for a single zone.
        """
        if parallel:
            computed_stats = self._compute_stats_parallel(
                zones, raster_array, affine_transform
            )
        else:
            computed_stats = self._compute_stats_serial(
                zones, raster_array, affine_transform
            )

        for feat, feature_stats in computed_stats:
            out_stats = _prefix_stat_keys(prefix, feature_stats)
            out_stats.update(_requested_feat_properties(copy_properties, feat))
            yield out_stats

    def _compute_stats_serial(self, zones, raster_array, affine):
        """Compute all stats sequentially"""
        total_num_zones = len(zones)
        for ind, (__, feat) in enumerate(zones.iterrows(), start=1):
            yield feat, self._compile_stats(feat, raster_array, affine)
            logger.debug(
                "Computed stats for %d/%d zones (%.2f%)",
                ind,
                total_num_zones,
                ind / total_num_zones * 100,
            )

    def _compute_stats_parallel(self, zones, raster_array, affine):
        """Use dask delayed to compute stats in parallel"""
        results = [
            dask.delayed(self._compute_with_no_warning_and_return_feat)(
                feat=feat, raster_array=raster_array, affine=affine
            )
            for __, feat in zones.iterrows()
        ]
        yield from dask.compute(*results)

    def _compute_with_no_warning_and_return_feat(
        self, feat, raster_array, affine
    ):
        """Don't throw warnings when running in parallel"""
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore",
                # cspell:disable-next-line
                rasterio.errors.NotGeoreferencedWarning,
            )

            out = self._compile_stats(feat, raster_array, affine)

        return feat, out

    def _compile_stats(self, feat, raster_array, affine):
        """Compile stats for a single zone"""

        zone = feat[_GPKG_GEOM_COL]
        logger.debug("Sub-setting raster to zone window")
        windowed_raster, window_transform = _subset_raster_to_window(
            raster_array, zone.bounds, affine
        )
        logger.debug("Windowed raster shape=%r", windowed_raster.shape)

        if windowed_raster.size == 0:
            feature_stats = self.computable_stats.empty
            feature_stats.update(_empty_extra_user_stats(self.add_stats))
            return feature_stats

        logger.debug("    - Applying zone mask to raster")
        masked_raster, rasterized_zone = _mask_to_zone(
            windowed_raster, window_transform, zone, self.all_touched
        )
        processed_raster = _mask_nodata(masked_raster, self.nodata)
        processed_raster = _safe_apply_func(self.zone_func, processed_raster)

        logger.debug("    - Computing basic stats")
        feature_stats = self.computable_stats.computed_base_stats(
            processed_raster=processed_raster,
            category_map=self.category_map,
            masked_raster=masked_raster,
            nodata=self.nodata,
        )
        logger.debug("    - Computing fractional stats")
        feature_stats.update(
            self.computable_stats.computed_fractional_stats(
                zone,
                processed_raster,
                window_transform,
                self.nodata,
                self.category_map,
            )
        )
        logger.debug("    - Computing percentile stats")
        feature_stats.update(
            self.computable_stats.computed_percentiles(processed_raster)
        )
        feature_stats.update(
            _compute_extra_user_stats(
                self.add_stats, processed_raster, feat, rasterized_zone
            )
        )
        return feature_stats


def _subset_raster_to_window(raster_array, geom_bounds, affine):
    """Subset the raster to the window defined by the geometry"""
    window = rasterio.windows.from_bounds(*geom_bounds, transform=affine)
    y, x = window.toslices()
    window = rasterio.windows.Window.from_slices(y, x, affine)
    window_transform = rasterio.windows.transform(
        window=window, transform=affine
    )
    windowed_raster = raster_array.isel(x=x, y=y)
    return windowed_raster, window_transform


def _mask_to_zone(raster, transform, geom, all_touched):
    """Mask array to the rasterized zone geometry"""
    rasterized_zone = rasterio.features.rasterize(
        [(geom, 1)],
        out_shape=(raster.sizes["y"], raster.sizes["x"]),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=all_touched,
    )
    masked_raster = raster.where(rasterized_zone == 1)
    return masked_raster, rasterized_zone


def _mask_nodata(raster, nodata):
    """Mask out nodata values in array"""
    return raster.where(
        (raster != nodata) & (raster != raster.attrs.get("nodata"))
    )


def _safe_apply_func(zone_func, processed_raster):
    """Apply zone function if it's provided in the correct form"""
    if zone_func is None:
        return processed_raster

    if not callable(zone_func):
        msg = (
            "zone_func must be a callable function "
            "which accepts a single `raster` arg."
        )
        raise revrtTypeError(msg)

    return zone_func(processed_raster)


def _empty_extra_user_stats(extra_user_stats):
    """Compute user stats if requested"""
    if not extra_user_stats:
        return {}

    return dict.fromkeys(extra_user_stats)


def _compute_extra_user_stats(
    extra_user_stats, processed_raster, feat, rasterized_zone
):
    """Compute user stats if requested"""
    if not extra_user_stats:
        return {}

    return {
        stat_name: stat_func(processed_raster, feat, rasterized_zone)
        for stat_name, stat_func in extra_user_stats.items()
    }


def _prefix_stat_keys(prefix, stats):
    """Prefix output stats if requested"""
    if not prefix:
        return stats

    return {f"{prefix}{key}": val for key, val in stats.items()}


def _requested_feat_properties(user_request, feat):
    """Compile requested feature properties"""
    properties = feat.to_dict()
    if not user_request:
        user_request = properties
        user_request.pop(_GPKG_GEOM_COL, None)

    return {prop: properties.get(prop) for prop in user_request}
