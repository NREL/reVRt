"""Handler for file containing GeoTIFF layers"""

import logging
from pathlib import Path
from functools import cached_property

import rioxarray
import xarray as xr

from revrt.exceptions import revrtFileNotFoundError


logger = logging.getLogger(__name__)


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
        self._template_file = Path(template_file or fp)
        self._block_size = block_size

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
            if self.template_file.suffix == ".zarr"
            else rioxarray.open_rasterio
        )
        with open_method(self.template_file) as ds:
            return {
                "width": ds.rio.width,
                "height": ds.rio.height,
                "crs": ds.rio.crs,
        with xr.open_dataset(self.fp) as ds:
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

    def _layer_profile_from_open_ds(self, layer, ds):
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
