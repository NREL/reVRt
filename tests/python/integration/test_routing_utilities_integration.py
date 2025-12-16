"""reVrt tests for routing utilities"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from rasterio.transform import from_origin

from revrt.routing.utilities import map_to_costs
from revrt.utilities import LayeredFile


@pytest.fixture(scope="module")
def sample_layered_data(tmp_path_factory):
    """Sample layered data files to use across tests"""
    data_dir = tmp_path_factory.mktemp("routing_data")

    layered_fp = data_dir / "test_layered.zarr"
    layer_file = LayeredFile(layered_fp)

    height, width = (7, 8)
    cell_size = 1.0
    x0, y0 = 0.0, float(height)
    transform = from_origin(x0, y0, cell_size, cell_size)
    x_coords = (
        x0 + np.arange(width, dtype=np.float32) * cell_size + cell_size / 2
    )
    y_coords = (
        y0 - np.arange(height, dtype=np.float32) * cell_size - cell_size / 2
    )

    layer_1 = np.array(
        [
            [
                [7, 7, 8, 0, 9, 9, 9, 0],
                [8, 1, 2, 2, 9, 9, 9, 0],
                [9, 1, 3, 3, 9, 1, 2, 3],
                [9, 1, 2, 1, 9, 1, 9, 0],
                [9, 9, 9, 1, 9, 1, 9, 0],
                [9, 9, 9, 1, 1, 1, 9, 0],
                [9, 9, 9, 9, 9, 9, 9, 0],
            ]
        ],
        dtype=np.float32,
    )

    da = xr.DataArray(
        layer_1,
        dims=("band", "y", "x"),
        coords={"y": y_coords, "x": x_coords},
    )
    da = da.rio.write_crs("EPSG:4326")
    da = da.rio.write_transform(transform)

    geotiff_fp = data_dir / "layer_1.tif"
    da.rio.to_raster(geotiff_fp, driver="GTiff")

    layer_file.write_geotiff_to_file(geotiff_fp, "layer_1", overwrite=True)
    return layered_fp


def test_basic_map_to_costs_integration(sample_layered_data):
    """Basic integration test for mapping to costs from layered data"""

    route_table = pd.DataFrame(
        {
            "start_lat": 5.5,
            "start_lon": [1.5, 2.5],
            "end_lat": 4.5,
            "end_lon": 6.5,
        }
    )

    with xr.open_dataset(
        sample_layered_data, consolidated=False, engine="zarr"
    ) as ds:
        route_table = map_to_costs(
            route_table,
            crs=ds.rio.crs,
            transform=ds.rio.transform(),
            shape=ds.rio.shape,
        )

    np.testing.assert_array_equal(
        route_table["start_row"].to_numpy(), np.array([1, 1])
    )
    np.testing.assert_array_equal(
        route_table["start_col"].to_numpy(), np.array([1, 2])
    )
    np.testing.assert_array_equal(
        route_table["end_row"].to_numpy(), np.array([2, 2])
    )
    np.testing.assert_array_equal(
        route_table["end_col"].to_numpy(), np.array([6, 6])
    )


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
