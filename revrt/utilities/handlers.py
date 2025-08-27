"""Handler for file containing GeoTIFF layers"""

import shutil
import logging
from pathlib import Path
from warnings import warn
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
from revrt.warn import revrtWarning


logger = logging.getLogger(__name__)
_ZARR_COMPRESSORS = zarr.codecs.BloscCodec(
    cname="zstd", clevel=9, shuffle=zarr.codecs.BloscShuffle.shuffle
)
_NUM_GEOTIFF_DIMS = 3  # (band, y, x)


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

    TRANSFORM_ATOL = 0.01
    """Tolerance in transform comparison when checking GeoTIFFs"""

    def __init__(self, fp):
        """

        Parameters
        ----------
        fp : path-like
            Path to layered file on disk. If this file is to be created,
            a `template_file` must be provided (and must exist on disk).
            Otherwise, the `template_file` input can be ignored and this
            input will be used as the template file.
        """
        self.fp = Path(fp)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.fp})"

    def __str__(self):
        num_layers = len(self.data_layers)
        if num_layers == 1:  # pragma: no cover
            return f"{self.__class__.__name__} with 1 layer"
        return f"{self.__class__.__name__} with {num_layers:,d} layers"

    def __getitem__(self, layer):
        # This method is ported for backward compatibility, but it's
        # unlikely to be useful in practice since it loads the entire
        # layer data all at once
        if layer not in self.layers:
            msg = f"{layer!r} is not present in {self.fp}"
            raise revrtKeyError(msg)

        logger.debug("\t- Extracting %s from %s", layer, self.fp)
        with xr.open_dataset(self.fp, consolidated=False) as ds:
            profile = _layer_profile_from_open_ds(layer, ds)
            values = ds[layer].values

        return profile, values

    @cached_property
    def profile(self):
        """dict: Template layer profile"""
        with xr.open_dataset(self.fp, consolidated=False) as ds:
            return {
                "width": ds.rio.width,
                "height": ds.rio.height,
                "crs": ds.rio.crs,
                "transform": ds.rio.transform(),
            }

    @property
    def shape(self):
        """tuple: Template layer shape"""
        return self.profile["height"], self.profile["width"]

    @property
    def layers(self):
        """list: All available layers in file"""
        if not self.fp.exists():
            msg = f"File {self.fp} not found"
            raise revrtFileNotFoundError(msg)

        with xr.open_dataset(self.fp, consolidated=False) as ds:
            return list(ds.variables)

    @property
    def data_layers(self):
        """list: Available data layers in file"""
        return [
            layer_name
            for layer_name in self.layers
            if layer_name
            not in {
                "band",
                "x",
                "y",
                "spatial_ref",
                self.LATITUDE,
                self.LONGITUDE,
            }
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
        with xr.open_dataset(self.fp, consolidated=False) as ds:
            return _layer_profile_from_open_ds(layer, ds)

    # TODO: allow template_file to be another zarr file
    def create_new(
        self, template_file, overwrite=False, chunk_x=2048, chunk_y=2048
    ):
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
        chunk_x, chunk_y : int, default=2048
            Chunk size of x and y dimension for layered file and any
            output created from it GeoTIFFs. By default, ``2048``.
        """
        if self.fp.exists() and not overwrite:
            msg = f"File {self.fp!r} exits and overwrite=False"
            raise revrtFileExistsError(msg)

        _validate_template(template_file)

        logger.debug("\t- Initializing %s from %s", self.fp, template_file)

        try:
            _init_zarr_file_from_tiff_template(
                template_file, self.fp, chunk_x=chunk_x, chunk_y=chunk_y
            )
            logger.info(
                "Layered file %s created from %s!", self.fp, template_file
            )
        except Exception:
            logger.exception("Error initializing %s", self.fp)
            if self.fp.exists():
                delete_data_file(self.fp)

    def write_layer(
        self,
        values,
        layer_name,
        description=None,
        overwrite=False,
        nodata=None,
    ):
        """Write a layer to the file

        Parameters
        ----------
        values : array-like
            Layer data (can be numpy array, xarray.DataArray, or
            dask.array).
        layer_name : str
            Name of layer to be written to file.
        description : str, optional
            Description of layer being added. By default, ``None``.
        overwrite : bool, default=False
            Option to overwrite layer data if layer already exists in
            :class:`LayeredFile`.

            .. IMPORTANT::
              When overwriting data, the encoding (and therefore things
              like data type, nodata value, etc) is not allowed to
              change. If you need to overwrite an existing layer with a
              new type of data, manually remove it from the file first.

            By default, ``False``.
        nodata : int | float, optional
            Optional nodata value for the raster layer. This value will
            be added to the layer's attributes meta dictionary under the
            "nodata" key.

            .. WARNING::
               ``rioxarray`` does not recognize the "nodata" value when
               reading from a zarr file (because zarr uses the
               ``_FillValue`` encoding internally). To get the correct
               "nodata" value back when reading a :class:`LayeredFile`,
               you can either 1) read from ``da.rio.encoded_nodata`` or
               2) check the layer's attributes for the ``"nodata"`` key,
               and if present, use ``da.rio.write_nodata`` to write the
               nodata value so that ``da.rio.nodata`` gives the right
               value.

        Raises
        ------
        revrtFileNotFoundError
            If :class:`LayeredFile` does not exist.
        revrtKeyError
            If layer with the same name already exists and
            ``overwrite=False``.
        """
        if not self.fp.exists():
            msg = (
                f"File {self.fp} not found. Please create the file before "
                "adding layers."
            )
            raise revrtFileNotFoundError(msg)

        self._check_for_existing_layer(layer_name, overwrite)

        if values.ndim < _NUM_GEOTIFF_DIMS:
            values = np.expand_dims(values, 0)

        if values.shape[1:] != self.shape:
            msg = (
                f"Shape of provided data {values.shape[1:]} does "
                f"not match shape of LayeredFile: {self.shape}"
            )
            raise revrtValueError(msg)

        with xr.open_dataset(self.fp, consolidated=False) as ds:
            attrs = ds.attrs
            crs = ds.rio.crs
            transform = ds.rio.transform()
            layer_is_new = layer_name not in ds
            coords = ds.coords

        chunks = (1, attrs["chunks"]["y"], attrs["chunks"]["x"])

        da = xr.DataArray(values, dims=("band", "y", "x"), attrs=attrs)
        da = da.assign_coords(coords)
        da.attrs["count"] = 1
        da.attrs["description"] = description
        if nodata is not None:
            if layer_is_new:
                nodata = da.dtype.type(nodata)
                da = da.rio.write_nodata(nodata)
                da.attrs["nodata"] = nodata
            else:
                msg = (
                    "Attempting to set ``nodata`` value when overwriting "
                    "layer - this is not allowed. ``nodata`` value must be "
                    "set when layer is first created. User-provided "
                    f"``nodata`` value ({nodata}) will be ignored."
                )
                warn(msg, revrtWarning)

        ds_to_add = xr.Dataset({layer_name: da}, attrs=attrs)
        da = da.rio.write_crs(crs)
        da = da.rio.write_transform(transform)
        da = da.rio.write_grid_mapping()

        encoding = None
        if layer_is_new:
            encoding = {layer_name: da.encoding or {}}
            encoding[layer_name].update(
                {
                    "compressors": _ZARR_COMPRESSORS,
                    "dtype": da.dtype,
                    "chunks": chunks,
                }
            )

        ds_to_add.to_zarr(
            self.fp,
            mode="a",
            encoding=encoding,
            zarr_format=3,
            consolidated=False,
            compute=True,
        )

    def _check_for_existing_layer(self, layer_name, overwrite):
        """Warn about existing layers"""
        if layer_name not in self.layers:
            return

        msg = f"{layer_name!r} is already present in {self.fp}"
        if not overwrite:
            msg = f"{msg} and 'overwrite=False'"
            raise revrtKeyError(msg)

        msg = f"{msg} and will be replaced"
        logger.info(msg)

    def write_geotiff_to_file(
        self,
        geotiff,
        layer_name,
        check_tiff=True,
        description=None,
        overwrite=True,
        nodata=None,
    ):
        """Transfer GeoTIFF to layered file

        Parameters
        ----------
        geotiff : path-like
            Path to GeoTIFF file.
        layer_name : str
            Name of layer to be written to file.
        check_tiff : bool, optional
            Option to check GeoTIFF profile, CRS, and shape against
            layered file profile, CRS, and shape. By default, ``True``.
        description : str, optional
            Description of layer being added. By default, ``None``.
        overwrite : bool, default=False
            Option to overwrite layer data if layer already exists in
            :class:`LayeredFile`.

            .. IMPORTANT::
              When overwriting data, the encoding (and therefore things
              like data type, nodata value, etc) is not allowed to
              change. If you need to overwrite an existing layer with a
              new type of data, manually remove it from the file first.

            By default, ``False``.
        nodata : int | float, optional
            Optional nodata value for the raster layer. This value will
            be added to the layer's attributes meta dictionary under the
            "nodata" key.

            .. WARNING::
               ``rioxarray`` does not recognize the "nodata" value when
               reading from a zarr file (because zarr uses the
               ``_FillValue`` encoding internally). To get the correct
               "nodata" value back when reading a :class:`LayeredFile`,
               you can either 1) read from ``da.rio.encoded_nodata`` or
               2) check the layer's attributes for the ``"nodata"`` key,
               and if present, use ``da.rio.write_nodata`` to write the
               nodata value so that ``da.rio.nodata`` gives the right
               value.

        """
        if not self.fp.exists():
            logger.info("%s not found - creating from %s...", self.fp, geotiff)
            self.create_new(geotiff)

        logger.info(
            "%s being extracted from %s and added to %s",
            layer_name,
            geotiff,
            self.fp,
        )

        if check_tiff:
            logger.debug("\t- Checking %s against %s", geotiff, self.fp)
            check_geotiff(self.fp, geotiff, transform_atol=self.TRANSFORM_ATOL)

        with rioxarray.open_rasterio(geotiff, chunks="auto") as tif:
            logger.debug("\t- Writing data from %s to %s", geotiff, self.fp)
            self.write_layer(
                tif,
                layer_name,
                description=description,
                overwrite=overwrite,
                nodata=nodata,
            )

    def layer_to_geotiff(self, layer, geotiff, **profile_kwargs):
        """Extract layer from file and write to GeoTIFF file

        Parameters
        ----------
        layer : str
            Layer to extract,
        geotiff : path-like
            Path to output GeoTIFF file.
        **profile_kwargs
            Additional keyword arguments to pass into writing the
            raster. The following attributes ar ignored (they are set
            using properties of the source :class:`LayeredFile`):

                - nodata
                - transform
                - crs
                - count
                - width
                - height

        """
        logger.debug("\t- Writing %s from %s to %s", layer, self.fp, geotiff)
        with xr.open_dataset(self.fp, chunks="auto", consolidated=False) as ds:
            ds[layer].rio.to_raster(geotiff, driver="GTiff", **profile_kwargs)

    def extract_layers(self, layers, **profile_kwargs):
        """Extract layers from file and save to disk as GeoTIFFs

        Parameters
        ----------
        layers : dict
            Dictionary mapping layer names to GeoTIFF files to create.
        **profile_kwargs
            Additional keyword arguments to pass into writing the
            raster. The following attributes ar ignored (they are set
            using properties of the source :class:`LayeredFile`):

                - nodata
                - transform
                - crs
                - count
                - width
                - height
        """
        logger.info("Extracting layers from %s", self.fp)
        for layer_name, geotiff in layers.items():
            logger.info("- Extracting %s", layer_name)
            self.layer_to_geotiff(layer_name, geotiff, **profile_kwargs)

    def extract_all_layers(self, out_dir, **profile_kwargs):
        """Extract all layers from file and save to disk as GeoTIFFs

        Parameters
        ----------
        out_dir : str
            Path to output directory into which layers should be saved
            as GeoTIFFs.
        **profile_kwargs
            Additional keyword arguments to pass into writing the
            raster. The following attributes ar ignored (they are set
            using properties of the source :class:`LayeredFile`):

                - nodata
                - transform
                - crs
                - count
                - width
                - height
        """
        out_dir = Path(out_dir)
        if not out_dir.exists():
            out_dir.mkdir(parents=True)

        layers = {
            layer_name: out_dir / f"{layer_name}.tif"
            for layer_name in self.data_layers
        }
        self.extract_layers(layers, **profile_kwargs)


class LayeredTransmissionFile(LayeredFile):
    """Handle reading and writing H5 files and GeoTiffs"""

    def __init__(self, fp, layer_dir="."):
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
        layer_dir : path-like, optional
            Directory to search for layers in, if not found in current
            directory. By default, ``'.'``.
        """
        super().__init__(fp=fp)
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
        xr.open_dataset(layer_file_fp, consolidated=False) as ds,
        rioxarray.open_rasterio(geotiff) as tif,
    ):
        if tif.band > 1:
            msg = f"{geotiff} contains more than one band!"
            raise revrtProfileCheckError(msg)

        layered_file_shape = ds.sizes["band"], ds.sizes["y"], ds.sizes["x"]
        if layered_file_shape != tif.shape:
            msg = (
                f"Shape of exclusion data in {geotiff} and {layer_file_fp} "
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


def _layer_profile_from_open_ds(layer, ds):
    """Get layer profile from open dataset"""
    return {
        "nodata": ds[layer].attrs.get("nodata", ds[layer].rio.encoded_nodata),
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


def _init_zarr_file_from_tiff_template(
    template_file, out_fp, chunk_x, chunk_y
):
    """Initialize Zarr file from GeoTIFF template"""
    with rioxarray.open_rasterio(template_file) as geo:
        transform = geo.rio.transform()
        src_crs = geo.rio.crs

        x, y, lat, lon = _compute_lat_lon(
            geo.sizes["y"],
            geo.sizes["x"],
            src_crs,
            transform,
            chunk_x=chunk_x,
            chunk_y=chunk_y,
        )

        out_ds = _compile_ds(
            x, y, lat, lon, transform, src_crs, chunk_x, chunk_y
        )
        _save_ds_as_zarr_with_encodings(
            out_ds, chunk_x=chunk_x, chunk_y=chunk_y, out_fp=out_fp
        )


def _compute_lat_lon(ny, nx, src_crs, transform, chunk_x=2048, chunk_y=2048):
    """Compute latitude and longitude arrays from transform and CRS"""
    xx = dask.array.arange(nx, chunks=chunk_x, dtype="float32") + 0.5
    yy = dask.array.arange(ny, chunks=chunk_y, dtype="float32") + 0.5
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
        src_crs.to_string(),
        dtype="float32",
        new_axis=(0,),  # we add a new leading axis of length 2
        chunks=((2,), *x_mesh_transformed.chunks),  # chunk sizes for [2, y, x]
    )
    logger.debug(
        "Array shapes:\n\t- x=%r\n\t- y=%r\n\t- lon=%r\n\t- lat=%r",
        x.shape,
        y.shape,
        lon.shape,
        lat.shape,
    )
    return x, y, lat, lon


def _compile_ds(x, y, lat, lon, transform, src_crs, chunk_x, chunk_y):
    """Create an xarray Dataset with coordinates and attributes"""
    attrs = {"chunks": {"y": chunk_y, "x": chunk_x}}

    out_ds = xr.Dataset(attrs=attrs)
    out_ds = out_ds.assign_coords(
        band=(("band"), [0]),
        y=(("y"), y.astype(np.float32)),
        x=(("x"), x.astype(np.float32)),
        longitude=(("y", "x"), lon),
        latitude=(("y", "x"), lat),
    )

    out_ds = out_ds.rio.write_crs(src_crs)
    out_ds = out_ds.rio.write_transform(transform)
    return out_ds.rio.write_grid_mapping()


def _save_ds_as_zarr_with_encodings(out_ds, chunk_x, chunk_y, out_fp):
    """Write dataset to Zarr file with encodings"""
    encoding = {
        "y": {"dtype": "float32", "chunks": (chunk_y,)},
        "x": {"dtype": "float32", "chunks": (chunk_x,)},
        "longitude": {
            "compressors": _ZARR_COMPRESSORS,
            "dtype": "float32",
            "chunks": (chunk_y, chunk_x),
        },
        "latitude": {
            "compressors": _ZARR_COMPRESSORS,
            "dtype": "float32",
            "chunks": (chunk_y, chunk_x),
        },
    }
    logger.debug("Writing data to %s with encoding:\n %r", out_fp, encoding)
    out_ds.to_zarr(
        out_fp, mode="w", encoding=encoding, zarr_format=3, consolidated=False
    )


def _proj_to_lon_lat(xx_block, yy_block, src):
    """Block-wise transform to lon/lat; returns array shape [2, y, x]"""
    # create transformer inside the block to avoid pickling issues
    tr = Transformer.from_crs(src, "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(xx_block, yy_block)
    out = np.empty((2, *xx_block.shape), dtype="float32")
    out[0] = lon
    out[1] = lat
    return out
