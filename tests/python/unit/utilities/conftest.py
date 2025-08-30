"""Fixtures for use across utilities tests"""

import pytest
import numpy as np
import xarray as xr

from rasterio.transform import from_origin


@pytest.fixture(scope="module")
def sample_tiff_props():
    """Return properties for sample TIFF file used for tests"""
    width, height = 6, 10
    cell_size = 0.5
    x0, y0 = 0, 40
    transform = from_origin(x0, y0, cell_size, cell_size)
    return (x0, y0, width, height, cell_size, transform)


@pytest.fixture(scope="module")
def sample_tiff_fp(sample_tiff_props, tmp_path_factory):
    """Return path to TIFF file used for tests"""

    x0, y0, width, height, cell_size, transform = sample_tiff_props
    data = np.arange(width * height, dtype=np.float32)
    da = xr.DataArray(
        data.reshape((height, width)),
        dims=("y", "x"),
        coords={
            "x": x0 + np.arange(width) * cell_size + cell_size / 2,
            "y": y0 - np.arange(height) * cell_size - cell_size / 2,
        },
        name="test_band",
    )

    da = da.rio.write_crs("EPSG:4326")
    da.rio.write_transform(transform)

    out_fp = tmp_path_factory.mktemp("data") / "sample.tif"
    da.rio.to_raster(out_fp, driver="GTiff")
    return out_fp
