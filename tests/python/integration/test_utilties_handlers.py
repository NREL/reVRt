"""Integration tests for reVRt handlers"""

from pathlib import Path

import pytest
import rioxarray
import xarray as xr
import numpy as np

from revrt.utilities import LayeredFile


@pytest.fixture(scope="module")
def ri_tiff_fp(test_data_dir):
    """Return path to TIFF file used for tests"""
    return test_data_dir / "utilities" / "ri_transmission_barriers.tif"


def test_create_new_file(tmp_path, ri_tiff_fp):
    """Test creating a new file"""

    test_fp = tmp_path / "test.zarr"
    assert not test_fp.exists()

    lf = LayeredFile(test_fp)
    lf.create_new(ri_tiff_fp, overwrite=True)

    assert test_fp.exists()

    with (
        xr.open_dataset(test_fp) as ds,
        rioxarray.open_rasterio(ri_tiff_fp) as xds,
    ):
        lat_lon = xds.rio.reproject("EPSG:4326")
        assert np.isclose(ds["longitude"].min(), lat_lon.x.min())
        assert np.isclose(ds["longitude"].max(), lat_lon.x.max())
        assert np.isclose(ds["latitude"].min(), lat_lon.y.min())
        assert np.isclose(ds["latitude"].max(), lat_lon.y.max())
        assert (*ds["band"].shape, *ds["y"].shape, *ds["x"].shape) == xds.shape
        assert ds["latitude"].shape == xds.shape[1:]
        assert ds["longitude"].shape == xds.shape[1:]


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
