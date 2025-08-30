"""Base reVRt utilities"""

import shutil
from pathlib import Path

import rioxarray
import numpy as np
import xarray as xr

from revrt.exceptions import revrtProfileCheckError, revrtValueError


def buffer_routes(
    routes, row_widths=None, row_width_ranges=None, row_width_key="voltage"
):
    """Buffer routes by specified row widths or row width ranges

    Parameters
    ----------
    routes : geopandas.GeoDataFrame
        GeoDataFrame of routes to buffer. This dataframe must contain
        the route geometry as well as the `row_width_key` column.
    row_widths : dict
        A dictionary specifying the row widths in the following format:
        ``{"row_width_id": row_width_meters}``. The ``row_width_id`` is
        a value used to match each route with a particular ROW width
        (this is typically a voltage). The value should be found under
        the ``row_width_key`` entry of the ``routes``.
    row_width_ranges : list, optional
        Optional list of dictionaries, where each dictionary contains
        the keys "min", "max", and "width". This can be used to specify
        row widths based on ranges of values (e.g. voltage). For
        example, the following input::

            [
                {"min": 0, "max": 70, "width": 20},
                {"min": 70, "max": 150, "width": 30},
                {"min": 200, "max": 350, "width": 40},
                {"min": 400, "max": 500, "width": 50},
            ]

        would map voltages in the range ``0 <= volt < 70`` to a row
        width of 20 meters, ``70 <= volt < 150`` to a row width of 30
        meters, ``200 <= volt < 350`` to a row width of 40 meters,
        and so-on.

        .. IMPORTANT::
            Any values in the `row_widths` dict will take precedence
            over these ranges. So if a voltage of 138 kV is mapped to a
            row width of 25 meters in the `row_widths` dict, that value
            will be used instead of the 30 meter width specified by the
            ranges above.

        By default, ``None``.
    row_width_key : str, default="voltage"
        Name of column in vector file of routes used to map to the ROW
        widths. By default, ``"voltage"``.

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    revrtValueError
        _description_
    """
    if not (row_widths or row_width_ranges):
        msg = "Must provide either `row_widths` or `row_width_ranges` input!"
        raise revrtValueError(msg)

    half_width = None
    if row_width_ranges:
        half_width = _compute_half_width_using_ranges(
            routes, row_width_ranges, row_width_key=row_width_key
        )

    if row_widths:
        hw_from_volts = _compute_half_width_using_voltages(
            routes, row_widths, row_width_key=row_width_key
        )
        if half_width is None:
            half_width = hw_from_volts
        else:
            half_width[hw_from_volts > 0] = hw_from_volts[hw_from_volts > 0]

    routes["geometry"] = routes.buffer(half_width, cap_style="flat")

    return routes


def delete_data_file(fp):
    """Delete data file (can be Zarr, which is a directory)

    Parameters
    ----------
    fp : path-like
        Path to data file (or directory in case of Zarr).
    """
    fp = Path(fp)
    if not fp.exists():
        return

    if fp.is_dir():
        shutil.rmtree(fp)
    else:
        fp.unlink()


def check_geotiff(layer_file_fp, geotiff, transform_atol=0.01):
    """Compare GeoTIFF with exclusion layer and raise errors if mismatch

    Parameters
    ----------
    layer_file_fp : path-like
        Path to data representing a :class:`LayeredFile` instance.
    geotiff : path-like
        Path to GeoTIFF file.
    transform_atol : float, default=0.01
        Absolute tolerance parameter when comparing GeoTIFF transform
        data.

    Raises
    ------
    revrtProfileCheckError
        If shape, profile, or transform don't match between layered file
        and GeoTIFF file.
    """
    with (
        xr.open_dataset(
            layer_file_fp, consolidated=False, engine="zarr"
        ) as ds,
        rioxarray.open_rasterio(geotiff) as tif,
    ):
        if len(tif.band) > 1:
            msg = f"{geotiff} contains more than one band!"
            raise revrtProfileCheckError(msg)

        layered_file_shape = ds.sizes["band"], ds.sizes["y"], ds.sizes["x"]
        if layered_file_shape != tif.shape:
            msg = (
                f"Shape of layer data in {geotiff} and {layer_file_fp} "
                f"do not match!\n {tif.shape} !=\n {layered_file_shape}"
            )
            raise revrtProfileCheckError(msg)

        layered_file_crs = ds.rio.crs
        tif_crs = tif.rio.crs
        if layered_file_crs != tif_crs:
            msg = (
                f'Geospatial "CRS" in {geotiff} and {layer_file_fp} do not '
                f"match!\n {tif_crs} !=\n {layered_file_crs}"
            )
            raise revrtProfileCheckError(msg)

        layered_file_transform = ds.rio.transform()
        tif_transform = tif.rio.transform()
        if not np.allclose(
            layered_file_transform, tif_transform, atol=transform_atol
        ):
            msg = (
                f'Geospatial "transform" in {geotiff} and {layer_file_fp} '
                f"do not match!\n {tif_transform} !=\n "
                f"{layered_file_transform}"
            )
            raise revrtProfileCheckError(msg)


def _compute_half_width_using_ranges(
    routes, row_width_ranges, row_width_key="voltage"
):
    """Compute half-width for routes using row width ranges"""

    ranges = [(r["min"], r["max"], r["width"]) for r in row_width_ranges]

    def get_half_width(value):
        for min_val, max_val, width in ranges:
            if min_val <= value < max_val:
                return width / 2
        return -1

    return routes[row_width_key].map(get_half_width)


def _compute_half_width_using_voltages(
    routes, row_widths, row_width_key="voltage"
):
    """Compute half-width for routes using row width ranges"""
    row_widths = {float(k): v for k, v in row_widths.items()}

    def get_half_width(value):
        for voltage, width in row_widths.items():
            if np.isclose(value, voltage):
                return width / 2
        return -1

    return routes[row_width_key].map(get_half_width)
