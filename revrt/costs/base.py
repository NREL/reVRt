"""Abstract base class for layer builders"""

from pathlib import Path
from abc import ABC, abstractmethod

from revrt.constants import CELL_SIZE, DEFAULT_DTYPE


class BaseLayerCreator(ABC):
    """Abstract Base Class to create transmission routing layers"""

    def __init__(
        self,
        io_handler,
        input_layer_dir=".",
        output_tiff_dir=".",
        dtype=DEFAULT_DTYPE,
        cell_size=CELL_SIZE,
    ):
        """
        Parameters
        ----------
        io_handler : :class:`LayeredFile`
            Transmission layer IO handler
        input_layer_dir : path-like, optional
            Directory to search for input layers in, if not found in
            current directory. By default, ``'.'``.
        output_tiff_dir : path-like, optional
            Directory where cost layers should be saved as GeoTIFF.
            By default, ``"."``.
        dtype : np.dtype, optional
            Data type for final dataset. By default, ``float32``.
        cell_size : int, optional
            Side length of each cell, in meters. Cells are assumed to be
            square. By default, :obj:`CELL_SIZE`.
        """
        self._io_handler = io_handler
        self.input_layer_dir = Path(input_layer_dir)
        self.output_tiff_dir = Path(output_tiff_dir)
        self._dtype = dtype
        self._cell_size = cell_size

    @property
    def shape(self):
        """tuple: Layer shape"""
        return self._io_handler.shape

    @abstractmethod
    def build(self, *args, **kwargs):
        """Build layer"""
        raise NotImplementedError
