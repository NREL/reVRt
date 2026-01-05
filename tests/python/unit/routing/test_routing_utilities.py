"""reVrt tests for routing utilities"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest
from rasterio.transform import from_origin, xy

from revrt.routing.utilities import (
    _transform_lat_lon_to_row_col,
    _get_start_end_point_cost_indices,
    filter_points_outside_cost_domain,
    map_to_costs,
)
from revrt.warn import revrtWarning


@pytest.fixture
def cost_grid():
    """Simple grid metadata for routing tests"""

    height, width = (4, 5)
    cell_size = 1.0
    transform = from_origin(0.0, float(height), cell_size, cell_size)
    return "EPSG:4326", transform, (height, width)


def test_transform_lat_lon_to_row_col_expected_indices(cost_grid):
    """Map lat/lon pairs to expected raster indices"""

    crs, transform, _ = cost_grid
    lon_a, lat_a = xy(transform, 0, 0, offset="center")
    lon_b, lat_b = xy(transform, 3, 4, offset="center")

    row, col = _transform_lat_lon_to_row_col(
        transform, crs, np.array([lat_a, lat_b]), np.array([lon_a, lon_b])
    )

    assert isinstance(row, np.ndarray)
    assert isinstance(col, np.ndarray)
    np.testing.assert_array_equal(row, np.array([0, 3]))
    np.testing.assert_array_equal(col, np.array([0, 4]))


def test_get_start_end_point_cost_indices_adds_expected_columns(cost_grid):
    """Populate start/end index columns from coordinates"""

    crs, transform, _ = cost_grid
    lon_start_a, lat_start_a = xy(transform, 0, 0, offset="center")
    lon_start_b, lat_start_b = xy(transform, 1, 2, offset="center")
    lon_end_a, lat_end_a = xy(transform, 2, 3, offset="center")
    lon_end_b, lat_end_b = xy(transform, 3, 4, offset="center")

    route_points = pd.DataFrame(
        {
            "start_lat": [str(lat_start_a), str(lat_start_b)],
            "start_lon": [str(lon_start_a), str(lon_start_b)],
            "end_lat": [str(lat_end_a), str(lat_end_b)],
            "end_lon": [str(lon_end_a), str(lon_end_b)],
        }
    )

    updated = _get_start_end_point_cost_indices(route_points, crs, transform)

    assert updated is route_points
    np.testing.assert_array_equal(
        route_points["start_row"].to_numpy(), np.array([0, 1])
    )
    np.testing.assert_array_equal(
        route_points["start_col"].to_numpy(), np.array([0, 2])
    )
    np.testing.assert_array_equal(
        route_points["end_row"].to_numpy(), np.array([2, 3])
    )
    np.testing.assert_array_equal(
        route_points["end_col"].to_numpy(), np.array([3, 4])
    )


def test_filter_points_outside_cost_domain_no_warning(cost_grid):
    """Do not warn when all routes remain in bounds"""

    _, _, shape = cost_grid
    route_points = pd.DataFrame(
        {
            "start_row": [0, 1],
            "start_col": [0, 2],
            "end_row": [2, 3],
            "end_col": [3, 4],
        }
    )
    expected = route_points.copy(deep=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        filtered = filter_points_outside_cost_domain(route_points, shape)

    assert not caught
    pd.testing.assert_frame_equal(filtered, expected)


def test_filter_points_outside_cost_domain_warns_and_drops(cost_grid):
    """Warn and drop routes that fall outside bounds"""

    _, _, shape = cost_grid
    route_points = pd.DataFrame(
        {
            "start_row": [0, -1],
            "start_col": [0, 1],
            "end_row": [1, 5],
            "end_col": [1, 6],
            "label": ["valid", "invalid"],
        }
    )

    with pytest.warns(revrtWarning) as record:
        filtered = filter_points_outside_cost_domain(route_points, shape)

    assert len(record) == 1
    assert "outside of the cost exclusion domain" in str(record[0].message)
    assert filtered.index.tolist() == [0]
    assert filtered["label"].tolist() == ["valid"]


def test_map_to_costs_maps_and_preserves_valid_routes(cost_grid):
    """End-to-end mapping keeps valid route intact"""

    crs, transform, shape = cost_grid
    lon_start, lat_start = xy(transform, 0, 0, offset="center")
    lon_end, lat_end = xy(transform, 2, 3, offset="center")

    route_points = pd.DataFrame(
        {
            "start_lat": [lat_start],
            "start_lon": [lon_start],
            "end_lat": [lat_end],
            "end_lon": [lon_end],
        }
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mapped = map_to_costs(
            route_points.copy(deep=True), crs, transform, shape
        )

    revrt_warnings = [w for w in caught if isinstance(w.message, revrtWarning)]
    assert not revrt_warnings

    np.testing.assert_array_equal(
        mapped["start_row"].to_numpy(), np.array([0])
    )
    np.testing.assert_array_equal(
        mapped["start_col"].to_numpy(), np.array([0])
    )
    np.testing.assert_array_equal(
        mapped["end_row"].to_numpy(),
        np.array([2]),
    )
    np.testing.assert_array_equal(
        mapped["end_col"].to_numpy(),
        np.array([3]),
    )


def test_map_to_costs_filters_routes_outside_cost_domain(cost_grid):
    """Remove routes that leave the cost domain"""

    crs, transform, shape = cost_grid
    lon_valid_start, lat_valid_start = xy(transform, 0, 0, offset="center")
    lon_valid_end, lat_valid_end = xy(transform, 1, 1, offset="center")

    # Column beyond grid width keeps latitude inside range but forces filtering
    lon_outside = lon_valid_start + shape[1]
    lat_outside = lat_valid_start

    route_points = pd.DataFrame(
        {
            "start_lat": [lat_valid_start, lat_outside],
            "start_lon": [lon_valid_start, lon_outside],
            "end_lat": [lat_valid_end, lat_outside],
            "end_lon": [lon_valid_end, lon_outside],
        }
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mapped = map_to_costs(route_points, crs, transform, shape)

    revrt_warnings = [w for w in caught if isinstance(w.message, revrtWarning)]
    assert len(revrt_warnings) == 1
    np.testing.assert_array_equal(
        mapped["start_row"].to_numpy(), np.array([0])
    )
    np.testing.assert_array_equal(
        mapped["start_col"].to_numpy(), np.array([0])
    )
    np.testing.assert_array_equal(
        mapped["end_row"].to_numpy(),
        np.array([1]),
    )
    np.testing.assert_array_equal(
        mapped["end_col"].to_numpy(),
        np.array([1]),
    )
    assert mapped.index.tolist() == [0]


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
