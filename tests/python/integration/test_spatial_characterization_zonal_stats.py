"""Zonal stats integration tests"""

import pytest

from trev.spatial_characterization.zonal_stats import zonal_stats


@pytest.fixture(scope="module")
def sc_dir(test_data_dir):
    """Return Path to test data directory."""
    return test_data_dir / "spatial_characterization"


@pytest.fixture(scope="module")
def zonal_polygon_fp(sc_dir):
    """Return Path to test data directory."""
    return sc_dir / "polygons.shp"


def test_categorization_multi_stat(sc_dir, zonal_polygon_fp):
    """Test `zonal_stats` with categorical data and multiple stats"""
    category_names = {
        11: "Cat 1",
        12: "Cat 2",
        21: "Cat 3",
        22: "Cat 4",
        23: "Cat 5",
        24: "Cat 6",
        31: "Cat 7",
        41: "Cat 8",
        42: "Cat 9",
        43: "Cat 10",
        52: "Cat 11",
        71: "Cat 12",
        81: "Cat 13",
        82: "Cat 14",
        90: "Cat 15",
        95: "Cat 16",
    }

    expected = [
        {
            "fractional_pixel_count": {
                "Cat 9": 6503.84,
                "Cat 8": 19742.58,
                "Cat 12": 71294.13,
                "Cat 16": 1105.0,
                "Cat 11": 859959.59,
                "Cat 15": 10775.42,
                "Cat 7": 295.93,
                "Cat 13": 29370.76,
                "Cat 14": 293.0,
                "Cat 3": 4801.57,
                "Cat 4": 649.13,
                "Cat 6": 2.0,
                "Cat 1": 602.2,
                "Cat 5": 111.22,
                "Cat 10": 53.0,
            },
            "fractional_area": {
                "Cat 9": 52681134.01,
                "Cat 8": 159914926.19,
                "Cat 12": 577482441.59,
                "Cat 16": 8950528.5,
                "Cat 11": 6965672713.94,
                "Cat 15": 87280895.23,
                "Cat 7": 2397010.26,
                "Cat 13": 237903195.86,
                "Cat 14": 2373300.0,
                "Cat 3": 38892701.91,
                "Cat 4": 5257974.34,
                "Cat 6": 16200.0,
                "Cat 1": 4877782.21,
                "Cat 5": 900899.11,
                "Cat 10": 429300.0,
            },
            "id": 1,
        },
        {
            "fractional_pixel_count": {
                "Cat 11": 122.1,
                "Cat 9": 19.2,
            },
            "fractional_area": {
                "Cat 11": 989042.24,
                "Cat 9": 155497.85,
            },
            "id": 2,
        },
        {
            "fractional_pixel_count": {
                "Cat 11": 2.18,
                "Cat 12": 1.0,
                "Cat 9": 2.96,
            },
            "fractional_area": {
                "Cat 11": 17693.11,
                "Cat 12": 8100.0,
                "Cat 9": 23944.66,
            },
            "id": 3,
        },
    ]
    stats = zonal_stats(
        str(zonal_polygon_fp),  # TODO: fix need for str
        sc_dir / "layer_a.tif",
        categorical=True,
        stats=["fractional_area", "fractional_pixel_count"],
        all_touched=True,
        nodata=0,
        category_map=category_names,
        copy_properties=["id"],
    )
    assert stats == expected


def test_categorization_single_stat(sc_dir, zonal_polygon_fp):
    """Test `zonal_stats` with categorical data and a single stat"""
    categories = {1: "low", 2: "medium", 3: "high", 4: "preclusion"}

    expected = [
        {
            "fractional_area": {
                "medium": 6602742612.39,
                "low": 982589212.95,
                "preclusion": 466731581.99,
                "high": 92967595.81,
            },
            "id": 1,
        },
        {"fractional_area": {"medium": 1144540.09}, "id": 2},
        {"fractional_area": {"medium": 49737.77}, "id": 3},
    ]
    stats = zonal_stats(
        str(zonal_polygon_fp),
        sc_dir / "layer_b.tif",
        categorical=True,
        stats=["fractional_area"],
        all_touched=True,
        nodata=0,
        category_map=categories,
        copy_properties=["id"],
    )
    assert stats == expected


def test_fractional_area(sc_dir, zonal_polygon_fp):
    """Test `zonal_stats` for fractional area characterization"""
    expected = [
        {"value_multiplied_by_fractional_area": 53332734934.61, "id": 1},
        {"value_multiplied_by_fractional_area": 3747374.22, "id": 2},
        {"value_multiplied_by_fractional_area": 201921.02, "id": 3},
    ]

    def multiply_m2_to_acres_factor(area_in_m2):
        return area_in_m2 / 4046.85

    stats = zonal_stats(
        str(zonal_polygon_fp),
        sc_dir / "layer_c.tif",
        categorical=False,
        zone_func=multiply_m2_to_acres_factor,
        stats=["value_multiplied_by_fractional_area"],
        all_touched=True,
        nodata=0,
        copy_properties=["id"],
    )
    assert stats == expected
