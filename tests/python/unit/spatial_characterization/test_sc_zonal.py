"""Test statistic computation functions for spatial characterization"""

from pathlib import Path

import pytest
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import box
from rasterio.transform import Affine

from revrt.spatial_characterization.stats import _PCT_PREFIX
from revrt.spatial_characterization.zonal import ZonalStats
from revrt.exceptions import revrtTypeError


@pytest.fixture
def sample_raster():
    """Sample raster data for testing"""
    return xr.DataArray(
        np.array(
            [
                [1, 1, 5],
                [4, 5, 5],
                [9, 9, 9],
            ],
            dtype=np.float64,
        ),
        dims=("y", "x"),
        attrs={"transform": Affine(10.0, 0.0, -15, 0.0, -10.0, 15)},
    )


@pytest.fixture
def five_sample_zones():
    """GeoDataFrame with 5 zones for testing on sample raster"""
    return gpd.GeoDataFrame(
        {"id": [1, 2, 3, 4, 5], "A": ["a", "b", "c", "d", "e"]},
        geometry=[
            box(-5, -5, 5, 5),
            box(0, 0, 1, 1),
            box(100, 100, 200, 200),
            box(-5, -5, 15, 15),
            box(-10, -10, 10, 10),
        ],
    )


def test_basic_zonal_stats_from_array(sample_raster, five_sample_zones):
    """Test basic execution of `from_array` method"""

    cat_map = {1.0: "CAT_A", 5.0: "CAT_B", 9.0: "CAT_C"}
    nodata = 9

    zs = ZonalStats(
        nodata=nodata, stats="All", category_map=cat_map, all_touched=True
    )
    stats = zs.from_array(
        five_sample_zones,
        sample_raster,
        sample_raster.attrs["transform"],
        prefix="test_",
        copy_properties=["id"],
    )
    stats = list(stats)

    assert len(stats) == 5

    # central pixel zone
    first_expected = {
        "test_count": 1,
        "test_min": 5.0,
        "test_max": 5.0,
        "test_mean": 5.0,
        "test_sum": 5.0,
        "test_std": 0.0,
        "test_median": 5.0,
        "test_majority": "CAT_B",
        "test_minority": "CAT_B",
        "test_unique": 1,
        "test_range": 0.0,
        "test_nodata": 0.0,
        "test_pixel_count": {"CAT_B": 1},
        "test_fractional_pixel_count": {"CAT_B": 1.0},
        "test_fractional_area": {"CAT_B": 100.0},
        "test_value_multiplied_by_fractional_area": 500.0,
        "id": 1,
    }
    assert stats[0] == first_expected
    assert "A" not in stats[0]

    # central pixel zone
    second_expected = {
        "test_count": 1,
        "test_min": 5.0,
        "test_max": 5.0,
        "test_mean": 5.0,
        "test_sum": 5.0,
        "test_std": 0.0,
        "test_median": 5.0,
        "test_majority": "CAT_B",
        "test_minority": "CAT_B",
        "test_unique": 1,
        "test_range": 0.0,
        "test_nodata": 0.0,
        "test_pixel_count": {"CAT_B": 1},
        "test_fractional_pixel_count": {"CAT_B": 0.01},
        "test_fractional_area": {"CAT_B": 1.0},
        "test_value_multiplied_by_fractional_area": 5.0,
        "id": 2,
    }
    assert stats[1] == second_expected
    assert "A" not in stats[1]

    # no overlap with zone
    third_expected = {
        "test_count": 0,
        "test_min": None,
        "test_max": None,
        "test_mean": None,
        "test_sum": None,
        "test_std": None,
        "test_median": None,
        "test_majority": None,
        "test_minority": None,
        "test_unique": None,
        "test_range": None,
        "test_nodata": None,
        "test_pixel_count": 0,
        "test_fractional_pixel_count": None,
        "test_fractional_area": None,
        "test_value_multiplied_by_fractional_area": None,
        "id": 3,
    }
    assert stats[2] == third_expected
    assert "A" not in stats[2]

    # top right corner
    fourth_expected = {
        "test_count": 4,
        "test_min": 1.0,
        "test_max": 5.0,
        "test_mean": 4.0,
        "test_sum": 16.0,
        "test_std": 1.7320508075688772,
        "test_median": 5.0,
        "test_majority": "CAT_B",
        "test_minority": "CAT_A",
        "test_unique": 2.0,
        "test_range": 4.0,
        "test_nodata": 0.0,
        "test_pixel_count": {"CAT_A": 1, "CAT_B": 3},
        "test_fractional_pixel_count": {"CAT_A": 1.0, "CAT_B": 3.0},
        "test_fractional_area": {"CAT_A": 100.0, "CAT_B": 300.0},
        "test_value_multiplied_by_fractional_area": 1600.0,
        "id": 4,
    }
    assert stats[3] == fourth_expected
    assert "A" not in stats[3]

    # centered on middle pixel
    fifth_expected = {
        "test_count": 6,
        "test_min": 1.0,
        "test_max": 5.0,
        "test_mean": 3.5,
        "test_sum": 21.0,
        "test_std": 1.8027756377319946,
        "test_median": 4.5,
        "test_majority": "CAT_B",
        "test_minority": 4.0,
        "test_unique": 3,
        "test_range": 4.0,
        "test_nodata": 3.0,
        "test_pixel_count": {"CAT_A": 2, "CAT_B": 3, 4.0: 1},
        "test_fractional_pixel_count": {
            "CAT_A": 0.75,
            "CAT_B": 1.75,
            4.0: 0.5,
        },
        "test_fractional_area": {
            "CAT_A": 75,
            "CAT_B": 175,
            4.0: 50,
        },
        "test_value_multiplied_by_fractional_area": 1150.0,
        "id": 5,
    }
    assert stats[4] == fifth_expected
    assert "A" not in stats[4]


@pytest.mark.parametrize("prefix", [None, "test_"])
def test_zonal_stats_from_array_extra_params(prefix, sample_raster):
    """Test execution of `from_array` method with extra params"""

    zones = gpd.GeoDataFrame(
        {"id": [1, 2], "A": ["a", "b"]},
        geometry=[box(100, 100, 200, 200), box(-9, -9, 9, 9)],
    )

    def _squared(processed_raster):
        return processed_raster.astype(np.float64) ** 2

    def _count_ones(processed_raster, *__, **___):
        return np.sum(processed_raster == 1)

    def _count_twenty_fives(processed_raster, *__, **___):
        return np.sum(processed_raster == 25)

    zs = ZonalStats(
        stats=f"{_PCT_PREFIX}50 median",
        all_touched=False,
        zone_func=_squared,
        add_stats={"ones_count": _count_ones, "other": _count_twenty_fives},
    )
    stats = zs.from_array(
        zones, sample_raster, sample_raster.attrs["transform"], prefix=prefix
    )
    stats = list(stats)
    prefix = prefix or ""

    # no overlap with zone
    first_expected = {
        f"{prefix}median": None,
        f"{prefix}{_PCT_PREFIX}50": None,
        f"{prefix}ones_count": None,
        f"{prefix}other": None,
        "id": 1,
        "A": "a",
    }
    assert stats[0] == first_expected

    # center pixel
    second_expected = {
        f"{prefix}median": 25.0,
        f"{prefix}{_PCT_PREFIX}50": 25.0,
        f"{prefix}ones_count": 0,
        f"{prefix}other": 1,
        "id": 2,
        "A": "b",
    }
    assert stats[1] == second_expected


def test_bad_callable(sample_raster, five_sample_zones):
    """Test `from_array` method with bad zone function"""

    zs = ZonalStats(zone_func=1)
    stats = zs.from_array(
        five_sample_zones, sample_raster, sample_raster.attrs["transform"]
    )
    with pytest.raises(revrtTypeError) as exc_info:
        stats = list(stats)

    assert (
        str(exc_info.value) == "zone_func must be a callable function "
        "which accepts a single `raster` arg."
    )


def test_parallel_zonal_stats_no_client(sample_raster, five_sample_zones):
    """Test parallel compute of zonal stats without a dask client"""

    cat_map = {1.0: "CAT_A", 5.0: "CAT_B", 9.0: "CAT_C"}
    nodata = 9

    zs = ZonalStats(
        nodata=nodata, stats="All", category_map=cat_map, all_touched=True
    )
    truth_stats = zs.from_array(
        five_sample_zones,
        sample_raster,
        sample_raster.attrs["transform"],
        prefix="test_",
        copy_properties=["id"],
        parallel=False,
    )
    truth_stats = list(truth_stats)

    test_stats = zs.from_array(
        five_sample_zones,
        sample_raster,
        sample_raster.attrs["transform"],
        prefix="test_",
        copy_properties=["id"],
        parallel=True,
    )
    test_stats = list(test_stats)

    assert test_stats == truth_stats


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
