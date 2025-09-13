"""Fixtures for use across costs tests"""

import pytest
import numpy as np
import xarray as xr
from rasterio.transform import from_origin


@pytest.fixture(scope="module")
def sample_tiff_props():
    """Return properties for sample TIFF file used for tests"""
    width, height = 6, 5
    cell_size = 5
    x0, y0 = -10, 10
    transform = from_origin(x0, y0, cell_size, cell_size)
    return (x0, y0, width, height, cell_size, transform)


@pytest.fixture(scope="module")
def sample_iso_fp(sample_tiff_props, tmp_path_factory):
    """Return path to TIFF file used for tests"""
    x0, y0, width, height, cell_size, transform = sample_tiff_props
    da = xr.DataArray(
        np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2, 2],
                [3, 3, 3, 3, 3, 3],
                [4, 4, 4, 4, 4, 4],
            ]
        ),
        dims=("y", "x"),
        coords={
            "x": x0 + np.arange(width) * cell_size + cell_size / 2,
            "y": y0 - np.arange(height) * cell_size - cell_size / 2,
        },
        name="test_band",
    )

    da = da.rio.write_crs("ESRI:102008")
    da.rio.write_transform(transform)

    out_fp = tmp_path_factory.mktemp("data") / "sample_iso.tif"
    da.rio.to_raster(out_fp, driver="GTiff")
    return out_fp


@pytest.fixture(scope="module")
def sample_nlcd_fp(sample_tiff_props, tmp_path_factory):
    """Return path to TIFF file used for tests"""
    x0, y0, width, height, cell_size, transform = sample_tiff_props
    da = xr.DataArray(
        np.array(
            [
                [11, 11, 95, 95, 95, 90],
                [11, 90, 90, 81, 81, 90],
                [11, 90, 21, 80, 41, 90],
                [95, 90, 22, 24, 42, 90],
                [95, 90, 23, 81, 43, 90],
            ]
        ),
        dims=("y", "x"),
        coords={
            "x": x0 + np.arange(width) * cell_size + cell_size / 2,
            "y": y0 - np.arange(height) * cell_size - cell_size / 2,
        },
        name="test_band",
    )

    da = da.rio.write_crs("ESRI:102008")
    da.rio.write_transform(transform)

    out_fp = tmp_path_factory.mktemp("data") / "sample_nlcd.tif"
    da.rio.to_raster(out_fp, driver="GTiff")
    return out_fp


@pytest.fixture(scope="module")
def sample_slope_fp(sample_tiff_props, tmp_path_factory):
    """Return path to TIFF file used for tests"""
    x0, y0, width, height, cell_size, transform = sample_tiff_props
    da = xr.DataArray(
        np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 2, 3, 8, 9, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        ),
        dims=("y", "x"),
        coords={
            "x": x0 + np.arange(width) * cell_size + cell_size / 2,
            "y": y0 - np.arange(height) * cell_size - cell_size / 2,
        },
        name="test_band",
    )

    da = da.rio.write_crs("ESRI:102008")
    da.rio.write_transform(transform)

    out_fp = tmp_path_factory.mktemp("data") / "sample_slope.tif"
    da.rio.to_raster(out_fp, driver="GTiff")
    return out_fp


@pytest.fixture(scope="module")
def sample_extra_fp(sample_tiff_props, tmp_path_factory):
    """Return path to TIFF file used for tests"""
    x0, y0, width, height, cell_size, transform = sample_tiff_props
    da = xr.DataArray(
        np.array(np.arange(width * height).reshape((height, width))),
        dims=("y", "x"),
        coords={
            "x": x0 + np.arange(width) * cell_size + cell_size / 2,
            "y": y0 - np.arange(height) * cell_size - cell_size / 2,
        },
        name="test_band",
    )

    da = da.rio.write_crs("ESRI:102008")
    da.rio.write_transform(transform)

    out_fp = tmp_path_factory.mktemp("data") / "sample_extra_data.tif"
    da.rio.to_raster(out_fp, driver="GTiff")
    return out_fp
