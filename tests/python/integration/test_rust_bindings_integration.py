"""revrt rust binding tests"""

import json
from pathlib import Path

import pytest
import numpy as np
import xarray as xr
from rasterio.transform import from_origin
from skimage.graph import MCP_Geometric

from revrt import find_paths

from revrt.utilities import LayeredFile


def test_basic_single_route_layered_file(tmp_path):
    """Test routing using a LayeredFile-generated cost surface"""

    cost_values = np.array(
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

    height, width = cost_values.shape[1:]
    cell_size = 1.0
    x0, y0 = 0.0, float(height)
    transform = from_origin(x0, y0, cell_size, cell_size)
    x_coords = (
        x0 + np.arange(width, dtype=np.float32) * cell_size + cell_size / 2
    )
    y_coords = (
        y0 - np.arange(height, dtype=np.float32) * cell_size - cell_size / 2
    )

    da = xr.DataArray(
        cost_values,
        dims=("band", "y", "x"),
        coords={"y": y_coords, "x": x_coords},
    )
    da = da.rio.write_crs("EPSG:4326")
    da = da.rio.write_transform(transform)

    geotiff_fp = tmp_path / "costs.tif"
    da.rio.to_raster(geotiff_fp, driver="GTiff")

    layered_fp = tmp_path / "test_layered.zarr"
    layer_file = LayeredFile(layered_fp)
    layer_file.write_geotiff_to_file(
        geotiff_fp,
        "test_costs",
        overwrite=True,
    )
    cost_definition = {
        "cost_layers": [{"layer_name": "test_costs"}],
        "ignore_null_costs": True,
    }
    results = find_paths(
        zarr_fp=layered_fp,
        cost_layers=json.dumps(cost_definition),
        start=[(1, 1)],
        end=[(2, 6)],
    )

    assert len(results) == 1
    test_path, test_cost = results[0]

    mcp = MCP_Geometric(cost_values[0])
    costs, __ = mcp.find_costs(starts=[(1, 1)], ends=[(2, 6)])

    assert test_path == mcp.traceback((2, 6))
    assert np.isclose(test_cost, costs[(2, 6)])


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
