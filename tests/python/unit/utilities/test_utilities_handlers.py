"""Tests for reVRt utilities"""

from pathlib import Path

import pytest
import rioxarray
from pyproj.crs import CRS
from rasterio.transform import Affine

from revrt.utilities import LayeredFile
from revrt.exceptions import (
    revrtFileExistsError,
    revrtFileNotFoundError,
    revrtValueError,
)


@pytest.fixture(scope="module")
def test_exclusion_fp(test_utility_data_dir):
    """Return path to exclusion test file"""
    return test_utility_data_dir / "ri_exclusions.zarr"


@pytest.fixture(scope="module")
def test_tl_fp(test_utility_data_dir):
    """Return path to transmission layers test file"""
    return test_utility_data_dir / "transmission_layers.zarr"


def extract_geotiff(geotiff):
    """Test helper function to extract data from GeoTiff"""
    with rioxarray.open_rasterio(geotiff, chunks=(128, 128)) as tif:
        values, profile = tif.values, tif.profile

    return values, profile


def test_bad_file_format():
    """Test init with bad file format"""

    lh5 = LayeredFile("test_file.h5")
    with pytest.raises(revrtValueError) as error:
        lh5.template_file = "test_file.txt"

    assert "format is not supported" in str(error)

    with pytest.raises(revrtFileNotFoundError) as error:
        lh5.template_file = "test_file.h5"

    assert "not found on disk" in str(error)


def test_not_overwrite_when_create_new_file(tmp_path):
    """Test not overwriting when creating a new file"""

    test_fp = tmp_path / "test.h5"
    test_fp.touch()
    lh5 = LayeredFile(test_fp)
    with pytest.raises(revrtFileExistsError) as error:
        lh5.create_new(overwrite=False)
    assert "exits and overwrite=False" in str(error)

    with pytest.raises(revrtValueError) as error:
        lh5.create_new(overwrite=True)
    assert "Must provide template file" in str(error)


def test_layered_file_handler_props(test_tl_fp):
    """Test LayeredFile property attributes"""

    lf = LayeredFile(test_tl_fp)

    assert repr(lf) == f"LayeredFile({test_tl_fp})"
    assert str(lf) == "LayeredFile with 8 layers"
    assert lf.shape == (1434, 972)

    expected_profile = {
        "width": 972,
        "height": 1434,
        "crs": CRS(
            "+proj=tmerc +lat_0=41.0833333333333 "
            "+lon_0=-71.5 +k=0.99999375 +x_0=100000 +y_0=0 "
            "+ellps=GRS80 +units=m +no_defs=True"
        ),
        "transform": Affine(
            90.0, 0.0, 65848.6171875, 0.0, -90.0, 103948.140625
        ),
    }
    assert lf.profile == expected_profile

    expected_layers = {
        "band",
        "x",
        "y",
        "latitude",
        "longitude",
        "spatial_ref",
        "ISO_regions",
        "tie_line_costs_1500MW",
        "tie_line_costs_400MW",
        "tie_line_multipliers",
        "tie_line_costs_3000MW",
        "tie_line_costs_205MW",
        "tie_line_costs_102MW",
        "transmission_barrier",
    }
    assert set(lf.layers) == expected_layers

    expected_data_layers = expected_layers - {
        "band",
        "x",
        "y",
        "latitude",
        "longitude",
        "spatial_ref",
    }
    assert set(lf.data_layers) == expected_data_layers


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
