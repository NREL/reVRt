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


def test_extract_layer_to_geotiff():
    """Test extracting layer data from HDF5 file."""

    values, profile = extract_geotiff(ISO_TIFF)

    with tempfile.TemporaryDirectory() as td:
        h5_file = os.path.join(td, "test.h5")
        lh5 = LayeredFile(h5_file, template_file=XMISSION_H5)
        lh5.write_layer_to_h5(
            values, "iso_regions", profile=profile, description="ISO"
        )

        out_fp = os.path.join(td, "test.tiff")
        lh5.layer_to_geotiff("iso_regions", out_fp)

        test_values, test_profile = extract_geotiff(out_fp)

        assert np.allclose(test_values, values)
        assert test_profile == profile


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
