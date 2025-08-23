"""Handler for file containing GeoTIFF layers"""

import logging
from pathlib import Path
from functools import cached_property

import zarr
import dask
from pyproj import Transformer
import rioxarray
import numpy as np
import xarray as xr

from revrt.exceptions import (
    revrtFileExistsError,
    revrtFileNotFoundError,
    revrtKeyError,
    revrtProfileCheckError,
    revrtValueError,
)


logger = logging.getLogger(__name__)
ZARR_COMPRESSORS = zarr.codecs.BloscCodec(
    cname="zstd", clevel=9, shuffle=zarr.codecs.BloscShuffle.shuffle
)


class LayeredFile:
    """Handler for file containing GeTIFF layers

    This handler represents a file that stores various layers
    (i.e. exclusion layers, setback layers, transmission layers, etc).
    This file contains profile information, and this handler can be used
    to convert to and from such files.
    """

    SUPPORTED_FILE_ENDINGS = {".zarr", ".tif", ".tiff"}
    """Supported template file endings"""

    LATITUDE = "latitude"
    """Name of latitude values layer in :class:`LayeredFile`"""

    LONGITUDE = "longitude"
    """Name of longitude values layer in :class:`LayeredFile`"""

    def __init__(self, fp, chunks=(128, 128)):
        """

        Parameters
        ----------
        fp : path-like
            Path to layered file on disk. If this file is to be created,
            a `template_file` must be provided (and must exist on disk).
            Otherwise, the `template_file` input can be ignored and this
            input will be used as the template file.
        chunks : tuple, optional
            Chunk size of exclusions in layered file and any output
            GeoTIFFs. By default, ``(128, 128)``.
        template_file : path-like, optional
            Path to template GeoTIFF (``*.tif`` or ``*.tiff``) or Zarr
        self._chunks = chunks

    def __repr__(self):
        return f"{self.__class__.__name__}({self.fp})"

    def __str__(self):
        num_layers = len(self.data_layers)
        if num_layers == 1:
            return f"{self.__class__.__name__} with 1 layer"
        return f"{self.__class__.__name__} with {num_layers:,d} layers"

        # unlikely to be useful in practice since it loads the entire
        # layer data all at once
        if layer not in self.layers:
            msg = f"{layer!r} is not present in {self.fp}"
            raise revrtKeyError(msg)

        logger.debug("\t- Extracting %s from %s", layer, self.fp)
        with xr.open_dataset(self.fp) as ds:
            profile = _layer_profile_from_open_ds(layer, ds)
            values = ds[layer].values

        return profile, values

    @cached_property
    def profile(self):
        """dict: Template layer profile"""
        open_method = (
            xr.open_dataset
        with xr.open_dataset(self.fp) as ds:
                "height": ds.rio.height,
                "crs": ds.rio.crs,
        with xr.open_dataset(self.fp) as ds:
        return self.profile["height"], self.profile["width"]

    @cached_property
    def layers(self):
        """list: All available layers in file"""
        if not self.fp.exists():
                "transform": ds.rio.transform(),
            }

    @property
    def shape(self):
        """tuple: Template layer shape"""
        return self.profile["height"], self.profile["width"]

    @cached_property
    def layers(self):
        """list: All available layers in file"""
        if not self.fp.exists():
            msg = f"File {self.fp!r} not found"
            raise revrtFileNotFoundError(msg)

        with xr.open_dataset(self.fp) as ds:
            return list(ds.variables)

    @cached_property
    def data_layers(self):
        """list: Available data layers in file"""
        return [
            layer_name
            for layer_name in self.layers
            if layer_name
            not in {"band", "x", "y", "latitude", "longitude", "spatial_ref"}
        ]

    def layer_profile(self, layer):
        """Get layer profile as dictionary

        Parameters
        ----------
        layer : str
            Name of layer in file to get profile for.

        Returns
        -------
        dict
            Dictionary containing layer profile information, including
            the following keys:

                - "nodata": NoData value for layer
                - "width": width of layer
                - "height": height of layer
                - "crs": :class:`pyproj.crs.CRS` object for layer
                - "count": number of bands in layer
                - "dtype": data type of layer
                - "transform": :class:`Affine` transform for layer

        """
        with xr.open_dataset(self.fp) as ds:
            return self._layer_profile_from_open_ds(layer, ds)

            return _layer_profile_from_open_ds(layer, ds)

    def create_new(self, template_file, overwrite=False):
        """Create a new layered file

        Parameters
        ----------
        template_file : path-like, optional
            Path to template GeoTIFF (``*.tif`` or ``*.tiff``) or Zarr
            (``*.zarr``) file containing the profile and transform to be
            used for the layered file. If ``None``, then the `fp`
            input is used as the template. By default, ``None``.
        overwrite : bool, optional
            Overwrite file if is exists. By default, ``False``.
        """
        if self.fp.exists() and not overwrite:
            msg = f"File {self.fp!r} exits and overwrite=False"
            raise revrtFileExistsError(msg)

        _validate_template(template_file)

        logger.debug("\t- Initializing %s from %s", self.fp, template_file)

        try:
            _init_zarr_file_from_tiff_template(template_file, self.fp)
            logger.info(
                "Layered file %s created from %s!", self.fp, template_file
            )
        except Exception:
            logger.exception("Error initializing %s", self.fp)
            if self.fp.exists():
                self.fp.unlink()


class LayeredTransmissionFile(LayeredFile):
    """Handle reading and writing H5 files and GeoTiffs"""

    def __init__(
        self, fp, chunks=(128, 128), template_file=None, layer_dir="."
    ):
        """

        Parameters
        ----------
        fp : path-like
            Path to layered transmission file. If this file is to
            be created, a `template_file` must be provided (and must
            exist on disk). Otherwise, the `template_file` input can be
            ignored and this input will be used as the template file.
            This input can be set to `None` if only the tiff conversion
            utilities are required, but the `template_file` input must
            be provided in this case. By default, ``None``.
        chunks : tuple, optional
            Chunk size of exclusions in layered file and any output
            GeoTIFFs. By default, ``(128, 128)``.
        template_file : path-like, optional
            Path to template GeoTIFF (``*.tif`` or ``*.tiff``) or Zarr
            (``*.zarr``) file containing the profile and transform to be
            used for the layered transmission file. If ``None``, then
            the `fp` input is used as the template. If ``None`` and
            the `fp` input is also ``None``, an error is thrown.
            By default, ``None``.
        layer_dir : path-like, optional
            Directory to search for layers in, if not found in current
            directory. By default, ``'.'``.
        """
        super().__init__(fp=fp, chunks=chunks, template_file=template_file)
        self._layer_dir = Path(layer_dir)

    def load_data_using_h5_profile(
        self, geotiff, band=1, reproject=False, skip_profile_test=False
    ):
        """Load GeoTIFF data, converting to H5 profile if necessary

        Parameters
        ----------
        geotiff : str
            Path to GeoTIFF from which data should be read. If just the
            file name is provided, the class `layer_dir` attribute value
            is prepended to get the full path.
        band : int, optional
            Band to load from GeoTIFF. By default, ``1``.
        reproject : bool, optional
            Reproject raster to standard CRS and transform if True.
            By default, ``False``.
        skip_profile_test: bool, optional
            Skip checking that shape, transform, and CRS match template
            raster if ``True``. By default, ``False``.

        Returns
        -------
        array-like
            Raster data.
        """
        full_fname = Path(geotiff)
        if not full_fname.exists():
            full_fname = self._layer_dir / geotiff
            if not full_fname.exists():
                msg = f"Unable to find file {geotiff}"
                raise revrtFileNotFoundError(msg)

        skip_test = skip_profile_test
        return super().load_data_using_h5_profile(
            geotiff=full_fname,
            band=band,
            reproject=reproject,
            skip_profile_test=skip_test,
        )


def check_geotiff(h5, geotiff, chunks=(128, 128), transform_atol=0.01):
    """Compare GeoTIFF with exclusion layer and raise errors if mismatch

    Parameters
    ----------
    h5 : :class:`LayeredFile`
        ``LayeredFile`` instance containing `shape`, `profile`, and
        attributes.
    geotiff : str
        Path to GeoTIFF file.
    chunks : tuple
        Chunk size of exclusions in GeoTIFF,
    transform_atol : float
        Absolute tolerance parameter when comparing GeoTIFF transform
        data.

    Returns
    -------
    profile : dict
        GeoTIFF profile (attributes).
    values : ndarray
        GeoTIFF data.

    Raises
    ------
    revrtProfileCheckError
        If shape, profile, or transform don;t match between layered file
        and GeoTIFF file.
    """
    with rioxarray.open_rasterio(geotiff, chunks=chunks) as tif:
        if tif.band > 1:
            msg = f"{geotiff} contains more than one band!"
            raise revrtProfileCheckError(msg)

        if not np.array_equal(h5.shape, tif.shape[1:]):
            msg = (
                f"Shape of exclusion data in {geotiff} and {h5.fp} "
                "do not match!"
            )
            raise revrtProfileCheckError(msg)

        h5_crs = CRS.from_string(h5.profile["crs"]).to_dict()
        tif_crs = tif.rio.crs.to_dict()
        if not crs_match(h5_crs, tif_crs):
            msg = (
                f'Geospatial "CRS" in {geotiff} and {h5.fp} do not '
                f"match!\n {tif_crs} !=\n {h5_crs}"
            )
            raise revrtProfileCheckError(msg)

        if not np.allclose(
            h5.profile["transform"], tif.rio.transform(), atol=transform_atol
        ):
            msg = (
                f'Geospatial "transform" in {geotiff} and {h5.fp} '
                f"do not match!\n {h5.profile['transform']} !=\n "
                f"{tif.rio.transform()}"
            )
            raise revrtProfileCheckError(msg)


def crs_match(baseline_crs, test_crs, ignore_keys=("no_defs",)):
    """Compare baseline and test CRS values

    Parameters
    ----------
    baseline_crs : dict
        Baseline CRS to use a truth, must be a dict
    test_crs : dict
        Test CRS to compare with baseline, must be a dictionary.
    ignore_keys : tuple, optional
        Keys to not check. By default, ``('no_defs',)``.

    Returns
    -------
    crs_match : bool
        ``True`` if crs' match, ``False`` otherwise
    """
    for k, true_v in baseline_crs.items():
        if k not in ignore_keys:
            test_v = test_crs.get(k, true_v)
            if true_v != test_v:
                return False

    return True


def _layer_profile_from_open_ds(layer, ds):
    """Get layer profile from open dataset"""
    return {
        "nodata": ds[layer].rio.nodata,
        "width": ds.rio.width,
        "height": ds.rio.height,
        "crs": ds.rio.crs,
        "count": ds[layer].rio.count,
        "dtype": ds[layer].dtype,
        "transform": ds.rio.transform(),
    }


def _validate_template(template_file):
    """Validate template file"""
    template_file = Path(template_file)
    valid_file_ending = any(
        template_file.suffix == fe for fe in LayeredFile.SUPPORTED_FILE_ENDINGS
    )
    if not valid_file_ending:
        msg = (
            f"Template file {template_file!r} format is not "
            "supported! File must end in one of: "
            f"{LayeredFile.SUPPORTED_FILE_ENDINGS}"
        )
        raise revrtValueError(msg)

    if not template_file.exists():
        msg = f"Template file {template_file!r} not found on disk!"
        raise revrtFileNotFoundError(msg)


def _init_zarr_file_from_tiff_template(template_file, out_fp):
    """Initialize Zarr file from GeoTIFF template"""
    with rioxarray.open_rasterio(template_file) as geo:
        transform = geo.rio.transform()
        src_crs = geo.rio.crs.to_string()
        main_attrs = {"crs": src_crs, "transform": transform}

        x, y, lat, lon = _compute_lat_lon(
            geo.sizes["y"], geo.sizes["x"], src_crs, transform
        )

        out_ds = _compile_ds_and_save(x, y, lat, lon, main_attrs)
        _save_ds_as_zarr_with_encodings(out_ds, x, y, lat, lon, out_fp)


def _compute_lat_lon(ny, nx, src_crs, transform):
    """Compute latitude and longitude arrays from transform and CRS"""
    xx = dask.array.arange(nx, chunks="auto", dtype="float32") + 0.5
    yy = dask.array.arange(ny, chunks="auto", dtype="float32") + 0.5
    x_mesh, y_mesh = dask.array.meshgrid(xx, yy)  # shapes (y, x), chunked

    x = transform.c + xx * transform.a
    y = transform.f + yy * transform.e

    x_mesh_transformed = (
        transform.c + x_mesh * transform.a + y_mesh * transform.b
    )
    y_mesh_transformed = (
        transform.f + x_mesh * transform.d + y_mesh * transform.e
    )

    lon, lat = dask.array.map_blocks(
        _proj_to_lon_lat,
        x_mesh_transformed,
        y_mesh_transformed,
        src_crs,
        dtype="float32",
        new_axis=(0,),  # we add a new leading axis of length 2
        chunks=((2,), *x_mesh_transformed.chunks),  # chunk sizes for [2, y, x]
    )
    return x, y, lat, lon


def _compile_ds_and_save(x, y, lat, lon, attrs):
    """Create an xarray Dataset with coordinates and attributes"""
    out_ds = xr.Dataset(attrs=attrs)
    return out_ds.assign_coords(
        band=(("band"), [0]),
        y=(("y"), y.astype(np.float32)),
        x=(("x"), x.astype(np.float32)),
        longitude=(("y", "x"), lon),
        latitude=(("y", "x"), lat),
    )


def _save_ds_as_zarr_with_encodings(out_ds, x, y, lat, lon, out_fp):
    """Write dataset to Zarr file with encodings"""
    encoding = {
        "x": {
            "dtype": "float32",
            "chunks": tuple(c[0] for c in x.chunks),
        },
        "y": {
            "dtype": "float32",
            "chunks": tuple(c[0] for c in y.chunks),
        },
        "longitude": {
            "compressors": ZARR_COMPRESSORS,
            "dtype": "float32",
            "chunks": tuple(c[0] for c in lon.chunks),
        },
        "latitude": {
            "compressors": ZARR_COMPRESSORS,
            "dtype": "float32",
            "chunks": tuple(c[0] for c in lat.chunks),
        },
    }
    out_ds.to_zarr(out_fp, mode="w", encoding=encoding)


def _proj_to_lon_lat(xx_block, yy_block, src):
    """Block-wise transform to lon/lat; returns array shape [2, y, x]"""
    # create transformer inside the block to avoid pickling issues
    tr = Transformer.from_crs(src, "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(xx_block, yy_block)
    out = np.empty((2, *xx_block.shape), dtype="float32")
    out[0] = lon
    out[1] = lat
    return out
            Overwrite file if is exists. By default, ``False``.
        """
        if self.fp.exists() and not overwrite:
            msg = f"File {self.fp!r} exits and overwrite=False"
            raise revrtFileExistsError(msg)

        _validate_template(template_file)

        logger.debug("\t- Initializing %s from %s", self.fp, template_file)

        try:
            _init_zarr_file_from_tiff_template(template_file, self.fp)
            logger.info(
                "Layered file %s created from %s!", self.fp, template_file
            )
        except Exception:
            logger.exception("Error initializing %s", self.fp)
            if self.fp.exists():
                self.fp.unlink()


class LayeredTransmissionFile(LayeredFile):
    """Handle reading and writing H5 files and GeoTiffs"""

    def __init__(
        self, fp, chunks=(128, 128), template_file=None, layer_dir="."
    ):
        """

        Parameters
        ----------
        fp : path-like
            Path to layered transmission file. If this file is to
            be created, a `template_file` must be provided (and must
            exist on disk). Otherwise, the `template_file` input can be
            ignored and this input will be used as the template file.
            This input can be set to `None` if only the tiff conversion
            utilities are required, but the `template_file` input must
            be provided in this case. By default, ``None``.
        chunks : tuple, optional
            Chunk size of exclusions in layered file and any output
            GeoTIFFs. By default, ``(128, 128)``.
        template_file : path-like, optional
            Path to template GeoTIFF (``*.tif`` or ``*.tiff``) or Zarr
            (``*.zarr``) file containing the profile and transform to be
            used for the layered transmission file. If ``None``, then
            the `fp` input is used as the template. If ``None`` and
            the `fp` input is also ``None``, an error is thrown.
            By default, ``None``.
        layer_dir : path-like, optional
            Directory to search for layers in, if not found in current
            directory. By default, ``'.'``.
        """
        super().__init__(fp=fp, chunks=chunks, template_file=template_file)
        self._layer_dir = Path(layer_dir)

    def load_data_using_h5_profile(
        self, geotiff, band=1, reproject=False, skip_profile_test=False
    ):
        """Load GeoTIFF data, converting to H5 profile if necessary

        Parameters
        ----------
        geotiff : str
            Path to GeoTIFF from which data should be read. If just the
            file name is provided, the class `layer_dir` attribute value
            is prepended to get the full path.
        band : int, optional
            Band to load from GeoTIFF. By default, ``1``.
        reproject : bool, optional
            Reproject raster to standard CRS and transform if True.
            By default, ``False``.
        skip_profile_test: bool, optional
            Skip checking that shape, transform, and CRS match template
            raster if ``True``. By default, ``False``.

        Returns
        -------
        array-like
            Raster data.
        """
        full_fname = Path(geotiff)
        if not full_fname.exists():
            full_fname = self._layer_dir / geotiff
            if not full_fname.exists():
                msg = f"Unable to find file {geotiff}"
                raise revrtFileNotFoundError(msg)

        skip_test = skip_profile_test
        return super().load_data_using_h5_profile(
            geotiff=full_fname,
            band=band,
            reproject=reproject,
            skip_profile_test=skip_test,
        )


def check_geotiff(h5, geotiff, chunks=(128, 128), transform_atol=0.01):
    """Compare GeoTIFF with exclusion layer and raise errors if mismatch

    Parameters
    ----------
    h5 : :class:`LayeredFile`
        ``LayeredFile`` instance containing `shape`, `profile`, and
        attributes.
    geotiff : str
        Path to GeoTIFF file.
    chunks : tuple
        Chunk size of exclusions in GeoTIFF,
    transform_atol : float
        Absolute tolerance parameter when comparing GeoTIFF transform
        data.

    Returns
    -------
    profile : dict
        GeoTIFF profile (attributes).
    values : ndarray
        GeoTIFF data.

    Raises
    ------
    revrtProfileCheckError
        If shape, profile, or transform don;t match between layered file
        and GeoTIFF file.
    """
    with rioxarray.open_rasterio(geotiff, chunks=chunks) as tif:
        if tif.band > 1:
            msg = f"{geotiff} contains more than one band!"
            raise revrtProfileCheckError(msg)

        if not np.array_equal(h5.shape, tif.shape[1:]):
            msg = (
                f"Shape of exclusion data in {geotiff} and {h5.fp} "
                "do not match!"
            )
            raise revrtProfileCheckError(msg)

        h5_crs = CRS.from_string(h5.profile["crs"]).to_dict()
        tif_crs = tif.rio.crs.to_dict()
        if not crs_match(h5_crs, tif_crs):
            msg = (
                f'Geospatial "CRS" in {geotiff} and {h5.fp} do not '
                f"match!\n {tif_crs} !=\n {h5_crs}"
            )
            raise revrtProfileCheckError(msg)

        if not np.allclose(
            h5.profile["transform"], tif.rio.transform(), atol=transform_atol
        ):
            msg = (
                f'Geospatial "transform" in {geotiff} and {h5.fp} '
                f"do not match!\n {h5.profile['transform']} !=\n "
                f"{tif.rio.transform()}"
            )
            raise revrtProfileCheckError(msg)


def crs_match(baseline_crs, test_crs, ignore_keys=("no_defs",)):
    """Compare baseline and test CRS values

    Parameters
    ----------
    baseline_crs : dict
        Baseline CRS to use a truth, must be a dict
    test_crs : dict
        Test CRS to compare with baseline, must be a dictionary.
    ignore_keys : tuple, optional
        Keys to not check. By default, ``('no_defs',)``.

    Returns
    -------
    crs_match : bool
        ``True`` if crs' match, ``False`` otherwise
    """
    for k, true_v in baseline_crs.items():
        if k not in ignore_keys:
            test_v = test_crs.get(k, true_v)
            if true_v != test_v:
                return False

    return True


def _layer_profile_from_open_ds(layer, ds):
    """Get layer profile from open dataset"""
    return {
        "nodata": ds[layer].rio.nodata,
        "width": ds.rio.width,
        "height": ds.rio.height,
        "crs": ds.rio.crs,
        "count": ds[layer].rio.count,
        "dtype": ds[layer].dtype,
        "transform": ds.rio.transform(),
    }


def _validate_template(template_file):
    """Validate template file"""
    template_file = Path(template_file)
    valid_file_ending = any(
        template_file.suffix == fe for fe in LayeredFile.SUPPORTED_FILE_ENDINGS
    )
    if not valid_file_ending:
        msg = (
            f"Template file {template_file!r} format is not "
            "supported! File must end in one of: "
            f"{LayeredFile.SUPPORTED_FILE_ENDINGS}"
        )
        raise revrtValueError(msg)

    if not template_file.exists():
        msg = f"Template file {template_file!r} not found on disk!"
        raise revrtFileNotFoundError(msg)


def _init_zarr_file_from_tiff_template(template_file, out_fp):
    """Initialize Zarr file from GeoTIFF template"""
    with rioxarray.open_rasterio(template_file) as geo:
        transform = geo.rio.transform()
        src_crs = geo.rio.crs.to_string()
        main_attrs = {"crs": src_crs, "transform": transform}

        x, y, lat, lon = _compute_lat_lon(
            geo.sizes["y"], geo.sizes["x"], src_crs, transform
        )

        out_ds = _compile_ds_and_save(x, y, lat, lon, main_attrs)
        _save_ds_as_zarr_with_encodings(out_ds, x, y, lat, lon, out_fp)


def _compute_lat_lon(ny, nx, src_crs, transform):
    """Compute latitude and longitude arrays from transform and CRS"""
    xx = dask.array.arange(nx, chunks="auto", dtype="float32") + 0.5
    yy = dask.array.arange(ny, chunks="auto", dtype="float32") + 0.5
    x_mesh, y_mesh = dask.array.meshgrid(xx, yy)  # shapes (y, x), chunked

    x = transform.c + xx * transform.a
    y = transform.f + yy * transform.e

    x_mesh_transformed = (
        transform.c + x_mesh * transform.a + y_mesh * transform.b
    )
    y_mesh_transformed = (
        transform.f + x_mesh * transform.d + y_mesh * transform.e
    )

    lon, lat = dask.array.map_blocks(
        _proj_to_lon_lat,
        x_mesh_transformed,
        y_mesh_transformed,
        src_crs,
        dtype="float32",
        new_axis=(0,),  # we add a new leading axis of length 2
        chunks=((2,), *x_mesh_transformed.chunks),  # chunk sizes for [2, y, x]
    )
    return x, y, lat, lon


def _compile_ds_and_save(x, y, lat, lon, attrs):
    """Create an xarray Dataset with coordinates and attributes"""
    out_ds = xr.Dataset(attrs=attrs)
    return out_ds.assign_coords(
        band=(("band"), [0]),
        y=(("y"), y.astype(np.float32)),
        x=(("x"), x.astype(np.float32)),
        longitude=(("y", "x"), lon),
        latitude=(("y", "x"), lat),
    )


def _save_ds_as_zarr_with_encodings(out_ds, x, y, lat, lon, out_fp):
    """Write dataset to Zarr file with encodings"""
    encoding = {
        "x": {
            "dtype": "float32",
            "chunks": tuple(c[0] for c in x.chunks),
        },
        "y": {
            "dtype": "float32",
            "chunks": tuple(c[0] for c in y.chunks),
        },
        "longitude": {
            "compressors": ZARR_COMPRESSORS,
            "dtype": "float32",
            "chunks": tuple(c[0] for c in lon.chunks),
        },
        "latitude": {
            "compressors": ZARR_COMPRESSORS,
            "dtype": "float32",
            "chunks": tuple(c[0] for c in lat.chunks),
        },
    }
    out_ds.to_zarr(out_fp, mode="w", encoding=encoding)


def _proj_to_lon_lat(xx_block, yy_block, src):
    """Block-wise transform to lon/lat; returns array shape [2, y, x]"""
    # create transformer inside the block to avoid pickling issues
    tr = Transformer.from_crs(src, "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(xx_block, yy_block)
    out = np.empty((2, *xx_block.shape), dtype="float32")
    out[0] = lon
    out[1] = lat
    return out
            Overwrite file if is exists. By default, ``False``.
        """
        if self.fp.exists() and not overwrite:
            msg = f"File {self.fp!r} exits and overwrite=False"
            raise revrtFileExistsError(msg)

        _validate_template(template_file)

        logger.debug("\t- Initializing %s from %s", self.fp, template_file)

        try:
            _init_zarr_file_from_tiff_template(template_file, self.fp)
            logger.info(
                "Layered file %s created from %s!", self.fp, template_file
            )
        except Exception:
            logger.exception("Error initializing %s", self.fp)
            if self.fp.exists():
                self.fp.unlink()


class LayeredTransmissionFile(LayeredFile):
    """Handle reading and writing H5 files and GeoTiffs"""

    def __init__(
        self, fp, chunks=(128, 128), template_file=None, layer_dir="."
    ):
        """

        Parameters
        ----------
        fp : path-like
            Path to layered transmission file. If this file is to
            be created, a `template_file` must be provided (and must
            exist on disk). Otherwise, the `template_file` input can be
            ignored and this input will be used as the template file.
            This input can be set to `None` if only the tiff conversion
            utilities are required, but the `template_file` input must
            be provided in this case. By default, ``None``.
        chunks : tuple, optional
            Chunk size of exclusions in layered file and any output
            GeoTIFFs. By default, ``(128, 128)``.
        template_file : path-like, optional
            Path to template GeoTIFF (``*.tif`` or ``*.tiff``) or Zarr
            (``*.zarr``) file containing the profile and transform to be
            used for the layered transmission file. If ``None``, then
            the `fp` input is used as the template. If ``None`` and
            the `fp` input is also ``None``, an error is thrown.
            By default, ``None``.
        layer_dir : path-like, optional
            Directory to search for layers in, if not found in current
            directory. By default, ``'.'``.
        """
        super().__init__(fp=fp, chunks=chunks, template_file=template_file)
        self._layer_dir = Path(layer_dir)

    def load_data_using_h5_profile(
        self, geotiff, band=1, reproject=False, skip_profile_test=False
    ):
        """Load GeoTIFF data, converting to H5 profile if necessary

        Parameters
        ----------
        geotiff : str
            Path to GeoTIFF from which data should be read. If just the
            file name is provided, the class `layer_dir` attribute value
            is prepended to get the full path.
        band : int, optional
            Band to load from GeoTIFF. By default, ``1``.
        reproject : bool, optional
            Reproject raster to standard CRS and transform if True.
            By default, ``False``.
        skip_profile_test: bool, optional
            Skip checking that shape, transform, and CRS match template
            raster if ``True``. By default, ``False``.

        Returns
        -------
        array-like
            Raster data.
        """
        full_fname = Path(geotiff)
        if not full_fname.exists():
            full_fname = self._layer_dir / geotiff
            if not full_fname.exists():
                msg = f"Unable to find file {geotiff}"
                raise revrtFileNotFoundError(msg)

        skip_test = skip_profile_test
        return super().load_data_using_h5_profile(
            geotiff=full_fname,
            band=band,
            reproject=reproject,
            skip_profile_test=skip_test,
        )


def check_geotiff(h5, geotiff, chunks=(128, 128), transform_atol=0.01):
    """Compare GeoTIFF with exclusion layer and raise errors if mismatch

    Parameters
    ----------
    h5 : :class:`LayeredFile`
        ``LayeredFile`` instance containing `shape`, `profile`, and
        attributes.
    geotiff : str
        Path to GeoTIFF file.
    chunks : tuple
        Chunk size of exclusions in GeoTIFF,
    transform_atol : float
        Absolute tolerance parameter when comparing GeoTIFF transform
        data.

    Returns
    -------
    profile : dict
        GeoTIFF profile (attributes).
    values : ndarray
        GeoTIFF data.

    Raises
    ------
    revrtProfileCheckError
        If shape, profile, or transform don;t match between layered file
        and GeoTIFF file.
    """
    with rioxarray.open_rasterio(geotiff, chunks=chunks) as tif:
        if tif.band > 1:
            msg = f"{geotiff} contains more than one band!"
            raise revrtProfileCheckError(msg)

        if not np.array_equal(h5.shape, tif.shape[1:]):
            msg = (
                f"Shape of exclusion data in {geotiff} and {h5.fp} "
                "do not match!"
            )
            raise revrtProfileCheckError(msg)

        h5_crs = CRS.from_string(h5.profile["crs"]).to_dict()
        tif_crs = tif.rio.crs.to_dict()
        if not crs_match(h5_crs, tif_crs):
            msg = (
                f'Geospatial "CRS" in {geotiff} and {h5.fp} do not '
                f"match!\n {tif_crs} !=\n {h5_crs}"
            )
            raise revrtProfileCheckError(msg)

        if not np.allclose(
            h5.profile["transform"], tif.rio.transform(), atol=transform_atol
        ):
            msg = (
                f'Geospatial "transform" in {geotiff} and {h5.fp} '
                f"do not match!\n {h5.profile['transform']} !=\n "
                f"{tif.rio.transform()}"
            )
            raise revrtProfileCheckError(msg)


def crs_match(baseline_crs, test_crs, ignore_keys=("no_defs",)):
    """Compare baseline and test CRS values

    Parameters
    ----------
    baseline_crs : dict
        Baseline CRS to use a truth, must be a dict
    test_crs : dict
        Test CRS to compare with baseline, must be a dictionary.
    ignore_keys : tuple, optional
        Keys to not check. By default, ``('no_defs',)``.

    Returns
    -------
    crs_match : bool
        ``True`` if crs' match, ``False`` otherwise
    """
    for k, true_v in baseline_crs.items():
        if k not in ignore_keys:
            test_v = test_crs.get(k, true_v)
            if true_v != test_v:
                return False

    return True


def _layer_profile_from_open_ds(layer, ds):
    """Get layer profile from open dataset"""
    return {
        "nodata": ds[layer].rio.nodata,
        "width": ds.rio.width,
        "height": ds.rio.height,
        "crs": ds.rio.crs,
        "count": ds[layer].rio.count,
        "dtype": ds[layer].dtype,
        "transform": ds.rio.transform(),
    }


def _validate_template(template_file):
    """Validate template file"""
    template_file = Path(template_file)
    valid_file_ending = any(
        template_file.suffix == fe for fe in LayeredFile.SUPPORTED_FILE_ENDINGS
    )
    if not valid_file_ending:
        msg = (
            f"Template file {template_file!r} format is not "
            "supported! File must end in one of: "
            f"{LayeredFile.SUPPORTED_FILE_ENDINGS}"
        )
        raise revrtValueError(msg)

    if not template_file.exists():
        msg = f"Template file {template_file!r} not found on disk!"
        raise revrtFileNotFoundError(msg)


def _init_zarr_file_from_tiff_template(template_file, out_fp):
    """Initialize Zarr file from GeoTIFF template"""
    with rioxarray.open_rasterio(template_file) as geo:
        transform = geo.rio.transform()
        src_crs = geo.rio.crs.to_string()
        main_attrs = {"crs": src_crs, "transform": transform}

        x, y, lat, lon = _compute_lat_lon(
            geo.sizes["y"], geo.sizes["x"], src_crs, transform
        )

        out_ds = _compile_ds_and_save(x, y, lat, lon, main_attrs)
        _save_ds_as_zarr_with_encodings(out_ds, x, y, lat, lon, out_fp)


def _compute_lat_lon(ny, nx, src_crs, transform):
    """Compute latitude and longitude arrays from transform and CRS"""
    xx = dask.array.arange(nx, chunks="auto", dtype="float32") + 0.5
    yy = dask.array.arange(ny, chunks="auto", dtype="float32") + 0.5
    x_mesh, y_mesh = dask.array.meshgrid(xx, yy)  # shapes (y, x), chunked

    x = transform.c + xx * transform.a
    y = transform.f + yy * transform.e

    x_mesh_transformed = (
        transform.c + x_mesh * transform.a + y_mesh * transform.b
    )
    y_mesh_transformed = (
        transform.f + x_mesh * transform.d + y_mesh * transform.e
    )

    lon, lat = dask.array.map_blocks(
        _proj_to_lon_lat,
        x_mesh_transformed,
        y_mesh_transformed,
        src_crs,
        dtype="float32",
        new_axis=(0,),  # we add a new leading axis of length 2
        chunks=((2,), *x_mesh_transformed.chunks),  # chunk sizes for [2, y, x]
    )
    return x, y, lat, lon


def _compile_ds_and_save(x, y, lat, lon, attrs):
    """Create an xarray Dataset with coordinates and attributes"""
    out_ds = xr.Dataset(attrs=attrs)
    return out_ds.assign_coords(
        band=(("band"), [0]),
        y=(("y"), y.astype(np.float32)),
        x=(("x"), x.astype(np.float32)),
        longitude=(("y", "x"), lon),
        latitude=(("y", "x"), lat),
    )


def _save_ds_as_zarr_with_encodings(out_ds, x, y, lat, lon, out_fp):
    """Write dataset to Zarr file with encodings"""
    encoding = {
        "x": {
            "dtype": "float32",
            "chunks": tuple(c[0] for c in x.chunks),
        },
        "y": {
            "dtype": "float32",
            "chunks": tuple(c[0] for c in y.chunks),
        },
        "longitude": {
            "compressors": ZARR_COMPRESSORS,
            "dtype": "float32",
            "chunks": tuple(c[0] for c in lon.chunks),
        },
        "latitude": {
            "compressors": ZARR_COMPRESSORS,
            "dtype": "float32",
            "chunks": tuple(c[0] for c in lat.chunks),
        },
    }
    out_ds.to_zarr(out_fp, mode="w", encoding=encoding)


def _proj_to_lon_lat(xx_block, yy_block, src):
    """Block-wise transform to lon/lat; returns array shape [2, y, x]"""
    # create transformer inside the block to avoid pickling issues
    tr = Transformer.from_crs(src, "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(xx_block, yy_block)
    out = np.empty((2, *xx_block.shape), dtype="float32")
    out[0] = lon
    out[1] = lat
    return out
class LayeredTransmissionFile(LayeredFile):
    """Handle reading and writing layered files and GeoTiffs"""

    def __init__(
        self,
        fp,
        chunks=(128, 128),
        template_file=None,
        layer_dir=".",
        block_size=None,
    ):
        """

        Parameters
        ----------
        fp : path-like
            Path to layered transmission file. If this file is to
            be created, a `template_file` must be provided (and must
            exist on disk). Otherwise, the `template_file` input can be
            ignored and this input will be used as the template file.
            This input can be set to `None` if only the tiff conversion
            utilities are required, but the `template_file` input must
            be provided in this case. By default, ``None``.
        chunks : tuple, optional
            Chunk size of exclusions in layered file and any output
            GeoTIFFs. By default, ``(128, 128)``.
        template_file : path-like, optional
            Path to template GeoTIFF (``*.tif`` or ``*.tiff``) or Zarr
            (``*.zarr``) file containing the profile and transform to be
            used for the layered transmission file. If ``None``, then
            the `fp` input is used as the template. If ``None`` and
            the `fp` input is also ``None``, an error is thrown.
            By default, ``None``.
        layer_dir : path-like, optional
            Directory to search for layers in, if not found in current
            directory. By default, ``'.'``.
        block_size : int, optional
            Optional block size to use when building lat/lon datasets.
            Setting this value can help reduce memory issues when
            building a ``LayeredFile`` file. If ``None``, the lat/lon
            arrays are processed in full. By default, ``None``.
        """
        super().__init__(
            fp=fp,
            chunks=chunks,
            template_file=template_file,
            block_size=block_size,
        )
        self._layer_dir = Path(layer_dir)


def _layer_profile_from_open_ds(layer, ds):
    """Get layer profile from open dataset"""
    return {
        "nodata": ds[layer].rio.nodata,
        "width": ds.rio.width,
        "height": ds.rio.height,
        "crs": ds.rio.crs,
        "count": ds[layer].rio.count,
        "dtype": ds[layer].dtype,
        "transform": ds.rio.transform(),
    }


def _validate_template(template_file):
    """Validate template file"""
    template_file = Path(template_file)
    valid_file_ending = any(
        template_file.suffix == fe for fe in LayeredFile.SUPPORTED_FILE_ENDINGS
    )
    if not valid_file_ending:
        msg = (
            f"Template file {template_file!r} format is not "
            "supported! File must end in one of: "
            f"{LayeredFile.SUPPORTED_FILE_ENDINGS}"
        )
        raise revrtValueError(msg)

    if not template_file.exists():
        msg = f"Template file {template_file!r} not found on disk!"
        raise revrtFileNotFoundError(msg)


def _init_zarr_file_from_tiff_template(template_file, out_fp):
    """Initialize Zarr file from GeoTIFF template"""
    with rioxarray.open_rasterio(template_file) as geo:
        transform = geo.rio.transform()
        src_crs = geo.rio.crs.to_string()
        main_attrs = {"crs": src_crs, "transform": transform}

        x, y, lat, lon = _compute_lat_lon(
            geo.sizes["y"], geo.sizes["x"], src_crs, transform
        )

        out_ds = _compile_ds_and_save(x, y, lat, lon, main_attrs)
        _save_ds_as_zarr_with_encodings(out_ds, x, y, lat, lon, out_fp)


def _compute_lat_lon(ny, nx, src_crs, transform):
    """Compute latitude and longitude arrays from transform and CRS"""
    xx = dask.array.arange(nx, chunks="auto", dtype="float32") + 0.5
    yy = dask.array.arange(ny, chunks="auto", dtype="float32") + 0.5
    x_mesh, y_mesh = dask.array.meshgrid(xx, yy)  # shapes (y, x), chunked

    x = transform.c + xx * transform.a
    y = transform.f + yy * transform.e

    x_mesh_transformed = (
        transform.c + x_mesh * transform.a + y_mesh * transform.b
    )
    y_mesh_transformed = (
        transform.f + x_mesh * transform.d + y_mesh * transform.e
    )

    lon, lat = dask.array.map_blocks(
        _proj_to_lon_lat,
        x_mesh_transformed,
        y_mesh_transformed,
        src_crs,
        dtype="float32",
        new_axis=(0,),  # we add a new leading axis of length 2
        chunks=((2,), *x_mesh_transformed.chunks),  # chunk sizes for [2, y, x]
    )
    return x, y, lat, lon


def _compile_ds_and_save(x, y, lat, lon, attrs):
    """Create an xarray Dataset with coordinates and attributes"""
    out_ds = xr.Dataset(attrs=attrs)
    return out_ds.assign_coords(
        band=(("band"), [0]),
        y=(("y"), y.astype(np.float32)),
        x=(("x"), x.astype(np.float32)),
        longitude=(("y", "x"), lon),
        latitude=(("y", "x"), lat),
    )


def _save_ds_as_zarr_with_encodings(out_ds, x, y, lat, lon, out_fp):
    """Write dataset to Zarr file with encodings"""
    encoding = {
        "x": {
            "dtype": "float32",
            "chunks": tuple(c[0] for c in x.chunks),
        },
        "y": {
            "dtype": "float32",
            "chunks": tuple(c[0] for c in y.chunks),
        },
        "longitude": {
            "compressors": ZARR_COMPRESSORS,
            "dtype": "float32",
            "chunks": tuple(c[0] for c in lon.chunks),
        },
        "latitude": {
            "compressors": ZARR_COMPRESSORS,
            "dtype": "float32",
            "chunks": tuple(c[0] for c in lat.chunks),
        },
    }
    out_ds.to_zarr(out_fp, mode="w", encoding=encoding)


def _proj_to_lon_lat(xx_block, yy_block, src):
    """Block-wise transform to lon/lat; returns array shape [2, y, x]"""
    # create transformer inside the block to avoid pickling issues
    tr = Transformer.from_crs(src, "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(xx_block, yy_block)
    out = np.empty((2, *xx_block.shape), dtype="float32")
    out[0] = lon
    out[1] = lat
    return out
