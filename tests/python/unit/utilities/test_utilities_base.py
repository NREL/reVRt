"""Tests for base reVRt utilities"""

from pathlib import Path

import pytest
import geopandas as gpd
from shapely.geometry import box, LineString

from revrt.utilities.base import buffer_routes
from revrt.exceptions import revrtValueError


@pytest.fixture
def sample_paths():
    """Sample paths for buffering tests"""
    return gpd.GeoDataFrame(
        {
            "id": [1, 2],
            "A": ["a", "b"],
            "voltage": [12.0, 24],
        },
        geometry=[box(-5, -5, 5, 5), LineString([(10, -7), (10, 13)])],
        crs="ESRI:102008",
    )


def test_buffer_no_row_input(sample_paths):
    """Test that no ROW input raises error"""

    with pytest.raises(
        revrtValueError,
        match="Must provide either `row_widths` or `row_width_ranges` input!",
    ):
        buffer_routes(sample_paths)


def test_buffer_routes(sample_paths):
    """Test buffering routes by row width with exact integer value"""

    row_widths = {"12": 10, "24": 20, "36": 30}
    routes = buffer_routes(sample_paths, row_widths)
    assert "geometry" in routes
    assert routes.geometry.is_valid.all()
    assert all(routes.geometry.type == "Polygon")

    # account for rounded corners
    assert 19**2 < routes.iloc[0].geometry.area < 20**2
    assert routes.iloc[1].geometry.area == 20 * 20


def test_buffer_routes_range(sample_paths):
    """Test buffering routes by row width with range of voltages"""

    row_width_ranges = [
        {"min": 0, "max": 18, "width": 10},
        {"min": 18, "max": 30, "width": 20},
    ]
    routes = buffer_routes(sample_paths, row_width_ranges=row_width_ranges)
    assert "geometry" in routes
    assert routes.geometry.is_valid.all()
    assert all(routes.geometry.type == "Polygon")

    # account for rounded corners
    assert 19**2 < routes.iloc[0].geometry.area < 20**2
    assert routes.iloc[1].geometry.area == 20 * 20


def test_buffer_routes_value_takes_precedence_over_range(sample_paths):
    """Test buffering routes by row width with values and ranges"""

    row_width_ranges = [
        {"min": 0, "max": 18, "width": 10},
        {"min": 18, "max": 30, "width": 20},
    ]
    row_widths = {"24": 16, "36": 30}
    routes = buffer_routes(
        sample_paths, row_widths=row_widths, row_width_ranges=row_width_ranges
    )
    assert "geometry" in routes
    assert routes.geometry.is_valid.all()
    assert all(routes.geometry.type == "Polygon")

    # account for rounded corners
    assert 19**2 < routes.iloc[0].geometry.area < 20**2
    assert routes.iloc[1].geometry.area == 20 * 16


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
