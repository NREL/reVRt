"""Validate reVRt against scikit-image

Compare the solution given by scikit-image with reVRt, for the
same conditions.
"""

import json

import hypothesis
from hypothesis.extra.numpy import arrays, array_shapes
import numpy as np
from skimage.graph import MCP_Geometric
import xarray as xr

from revrt import find_paths

# Maximum value for input features used to calculate cost
# The test never ends for large values, such as 1e10.
MAX_COST = 1e6


def validate_single_var(data, start, end, tmp_path):
    """Validate reVRt against skimage for a given feature array

    Currently only for a single variable
    """
    da = xr.DataArray(data[None], dims=("band", "y", "x"))

    test_cost_fp = tmp_path / "test.zarr"
    ds = xr.Dataset({"test_costs": da})
    ds["test_costs"].encoding = {"fill_value": 1_000.0, "_FillValue": 1_000.0}
    ds.chunk({"x": 4, "y": 3}).to_zarr(test_cost_fp, mode="w", zarr_format=3)

    cost_definition = {"cost_layers": [{"layer_name": "test_costs"}]}
    results = find_paths(
        zarr_fp=str(test_cost_fp),
        cost_layers=json.dumps(cost_definition),
        start=[start],
        end=[end],
    )

    assert len(results) == 1
    revrt_route, revrt_cost = results[0]

    cost = da.values[0]
    mcp = MCP_Geometric(cost)
    costs, __ = mcp.find_costs(starts=[start], ends=[end])
    skimage_route = mcp.traceback(end)

    # compare route
    assert np.array_equal(skimage_route, revrt_route)
    # compare final cost
    assert np.isclose(revrt_cost, costs[end])


@hypothesis.given(
    arrays(
        np.float32,
        array_shapes(min_dims=2, max_dims=2, min_side=7, max_side=32),
        elements=hypothesis.strategies.integers(
            min_value=1, max_value=MAX_COST
        ),
        unique=True,
    ),
    hypothesis.strategies.tuples(
        hypothesis.strategies.floats(0, 1), hypothesis.strategies.floats(0, 1)
    ),
    hypothesis.strategies.tuples(
        hypothesis.strategies.floats(0, 1), hypothesis.strategies.floats(0, 1)
    ),
)
@hypothesis.settings(deadline=5_000, max_examples=100)
def test_basic(tmp_path_factory, data, start, end):
    """Validate single f32 variable"""
    start = (
        round(start[0] * max(0, data.shape[0] - 1)),
        round(start[1] * max(0, data.shape[1] - 1)),
    )
    end = (
        round(end[0] * max(0, data.shape[0] - 1)),
        round(end[1] * max(0, data.shape[1] - 1)),
    )

    tmpdir = tmp_path_factory.mktemp("skimage_test")
    validate_single_var(data, start, end, tmpdir)
