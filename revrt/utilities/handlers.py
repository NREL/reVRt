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

    def __init__(
        self, fp, chunks=(128, 128), template_file=None, block_size=None
    ):
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
            (``*.zarr``) file containing the profile and transform to be
            used for the layered file. If ``None``, then the `fp`
            input is used as the template. By default, ``None``.
        block_size : int, optional
            Optional block size to use when building lat/lon datasets.
            Setting this value can help reduce memory issues when
            building a ``LayeredFile`` file. If ``None``, the lat/lon
            arrays are processed in full. By default, ``None``.
        """
        self.fp = Path(fp)
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

    @property
    def template_file(self):
        """str: Path to template file"""
        return self._template_file

    @template_file.setter
    def template_file(self, new_template_file):
        self._template_file = Path(new_template_file)
        self._validate_template()

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
