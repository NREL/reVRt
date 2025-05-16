"""reVRt rust binding tests"""

import json
from pathlib import Path

import pytest
import numpy as np
import xarray as xr

from reVRt import find_path


def test_basic_single_route(tmp_path):
    """Test a basic routing invocation"""

    da = xr.DataArray(
        np.array(
            [
                [7, 7, 8, 0, 9, 9, 9, 0],
                [8, 1, 2, 2, 9, 9, 9, 0],
                [9, 1, 3, 3, 9, 1, 2, 3],
                [9, 1, 2, 1, 9, 1, 9, 0],
                [9, 9, 9, 1, 9, 1, 9, 0],
                [9, 9, 9, 1, 1, 1, 9, 0],
            ],
            dtype=np.float32,
        ),
        dims=("y", "x"),
    )

    test_cost_fp = tmp_path / "test.zarr"
    ds = xr.Dataset({"test_costs": da})
    ds["test_costs"].encoding = {"fill_value": 1_000.0, "_FillValue": 1_000.0}
    ds.chunk({"x": 4, "y": 3}).to_zarr(test_cost_fp, mode="w", zarr_format=3)

    cost_definition = {"cost_layers": [{"layer_name": "test_costs"}]}
    results = find_path(
        zarr_fp=test_cost_fp,
        cost_layers=json.dumps(cost_definition),
        start=(1, 1),
        end=[(2, 6)],
    )

    assert len(results) == 1


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
