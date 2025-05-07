"""Zonal stats integration tests"""

from pathlib import Path

import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from dask.distributed import Client
from rasterstats import zonal_stats as rzs
from rasterstats.utils import VALID_STATS, DEFAULT_STATS

from trev.spatial_characterization.zonal import ZonalStats


VALID_STATS.remove("nan")


@pytest.fixture(scope="module")
def sc_dir(test_data_dir):
    """Return Path to test data directory"""
    return test_data_dir / "spatial_characterization"


@pytest.fixture(scope="module")
def zonal_polygon_fp(sc_dir):
    """Return Path to test data directory"""
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
    zs = ZonalStats(
        stats=["fractional_area", "fractional_pixel_count"],
        all_touched=True,
        nodata=0,
        category_map=category_names,
    )
    stats = zs.from_files(
        zonal_polygon_fp, sc_dir / "layer_a.tif", copy_properties=["id"]
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
    zs = ZonalStats(
        stats=["fractional_area"],
        all_touched=True,
        nodata=0,
        category_map=categories,
    )
    stats = zs.from_files(
        zonal_polygon_fp, sc_dir / "layer_b.tif", copy_properties=["id"]
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
        return area_in_m2.astype("float64") / 4046.85

    zs = ZonalStats(
        zone_func=multiply_m2_to_acres_factor,
        stats=["value_multiplied_by_fractional_area"],
        all_touched=True,
        nodata=0,
    )
    stats = zs.from_files(
        zonal_polygon_fp, sc_dir / "layer_c.tif", copy_properties=["id"]
    )
    assert stats == expected


@pytest.mark.parametrize("nodata", [None, 11])
@pytest.mark.parametrize("stats", [None, VALID_STATS])
@pytest.mark.parametrize("all_touched", [True, False])
@pytest.mark.parametrize(
    "zone_func", [None, lambda x: x.astype("float64") ** 2 / 2]
)
def test_against_rasterstats(
    sc_dir, zonal_polygon_fp, nodata, stats, all_touched, zone_func
):
    """Test against the rasterstats zonal_stats function"""
    test_map = {11: "Cat 1", -1: "Unknown"}
    zs = ZonalStats(
        nodata=nodata,
        stats=stats,
        all_touched=all_touched,
        zone_func=zone_func,
        category_map=test_map,
    )
    test_stats = zs.from_files(zonal_polygon_fp, sc_dir / "layer_a.tif")
    truth_stats = rzs(
        str(zonal_polygon_fp),
        sc_dir / "layer_a.tif",
        nodata=nodata,
        stats=stats,
        all_touched=all_touched,
        zone_func=zone_func,
        category_map=test_map,
    )
    for out_test, out_expected in zip(test_stats, truth_stats, strict=True):
        out_test.pop("id", None)
        assert len(out_test) == len(stats) if stats else len(DEFAULT_STATS)
        assert len(out_expected) == len(stats) if stats else len(DEFAULT_STATS)
        for k, v in out_test.items():
            assert np.isclose(v, out_expected[k], rtol=1.0e-6, atol=1.0e-8)


def test_percentile_against_rasterstats(sc_dir, zonal_polygon_fp):
    """Test percentile stats against rasterstats zonal_stats function"""
    stats = [
        "percentile_10",
        "percentile_15.3",
        "percentile_25",
        "percentile_50",
        "percentile_75",
        "percentile_90",
    ]
    zs = ZonalStats(stats=stats, all_touched=True)
    test_stats = zs.from_files(zonal_polygon_fp, sc_dir / "layer_a.tif")
    for stats in test_stats:
        stats.pop("id", None)
    truth_stats = rzs(
        str(zonal_polygon_fp),
        sc_dir / "layer_a.tif",
        stats=stats,
        all_touched=True,
    )
    assert test_stats == truth_stats
    assert all(len(out_stats) == len(stats) for out_stats in test_stats)
    assert all(len(out_stats) == len(stats) for out_stats in truth_stats)


def test_pixel_count_against_rasterstats(sc_dir, zonal_polygon_fp):
    """Test pixel count against rasterstats zonal_stats function"""
    zs = ZonalStats(stats=["count", "pixel_count"], all_touched=True)
    test_stats = zs.from_files(zonal_polygon_fp, sc_dir / "layer_a.tif")
    for stats in test_stats:
        stats.pop("id", None)
        stats.update(stats.pop("pixel_count"))
    truth_stats = rzs(
        str(zonal_polygon_fp),
        sc_dir / "layer_a.tif",
        stats=["count"],
        all_touched=True,
        categorical=True,
    )
    assert test_stats == truth_stats


def test_range_only_against_rasterstats(sc_dir, zonal_polygon_fp):
    """Test range stat against rasterstats zonal_stats function"""
    zs = ZonalStats(stats=["range"], all_touched=True)
    test_stats = zs.from_files(zonal_polygon_fp, sc_dir / "layer_a.tif")
    for stats in test_stats:
        stats.pop("id", None)
    truth_stats = rzs(
        str(zonal_polygon_fp),
        sc_dir / "layer_a.tif",
        stats=["range"],
        all_touched=True,
    )
    assert test_stats == truth_stats
    assert all(len(stats) == 1 for stats in test_stats)
    assert all(len(stats) == 1 for stats in truth_stats)


@pytest.mark.parametrize(
    "in_out",
    [
        (None, [{"count": 0, "min": None, "max": None, "mean": None}]),
        (["range"], [{"range": None}]),
    ],
)
def test_no_intersection(sc_dir, tmp_path, in_out):
    """Test stats for shape with no intersection with raster"""
    stats, expected_out = in_out
    test_fp = tmp_path / "test.gpkg"
    gpd.GeoDataFrame(geometry=[Point(0, 0).buffer(10)]).to_file(
        test_fp, driver="GPKG"
    )
    zs = ZonalStats(stats=stats, all_touched=True)
    test_stats = zs.from_files(test_fp, sc_dir / "layer_a.tif")

    assert test_stats == expected_out


def test_add_stats_against_rasterstats(sc_dir, zonal_polygon_fp):
    """Test range stat against rasterstats zonal_stats function"""
    add_stats = {
        "my_stat": lambda x, *_, **__: float(x.max() - x.min()),
        "my_stat_2": lambda x, *_, **__: float(x.max() * 2),
    }
    zs = ZonalStats(add_stats=add_stats, all_touched=True)
    test_stats = zs.from_files(zonal_polygon_fp, sc_dir / "layer_a.tif")
    for stats in test_stats:
        stats.pop("id", None)
    truth_stats = rzs(
        str(zonal_polygon_fp),
        sc_dir / "layer_a.tif",
        all_touched=True,
        add_stats=add_stats,
    )
    for out_test, out_expected in zip(test_stats, truth_stats, strict=True):
        out_test.pop("id", None)
        assert len(out_test) == len(DEFAULT_STATS) + len(add_stats)
        assert len(out_expected) == len(DEFAULT_STATS) + len(add_stats)
        for k, v in out_test.items():
            assert np.isclose(v, out_expected[k], rtol=1.0e-6, atol=1.0e-8)


def test_prefix_against_rasterstats(sc_dir, zonal_polygon_fp):
    """Test pixel count against rasterstats zonal_stats function"""
    stats = [*VALID_STATS, "percentile_10.5"]
    add_stats = {
        "my_stat": lambda x, *_, **__: float(x.max() - x.min()),
        "my_stat_2": lambda x, *_, **__: float(x.max() * 2),
    }
    zs = ZonalStats(stats=VALID_STATS, all_touched=True, add_stats=add_stats)
    test_stats = zs.from_files(
        zonal_polygon_fp, sc_dir / "layer_a.tif", prefix="test_"
    )
    for stats in test_stats:
        stats.pop("id", None)

    truth_stats = rzs(
        str(zonal_polygon_fp),
        sc_dir / "layer_a.tif",
        stats=VALID_STATS,
        all_touched=True,
        prefix="test_",
        add_stats=add_stats,
    )
    for out_test, out_expected in zip(test_stats, truth_stats, strict=True):
        out_test.pop("id", None)
        assert len(out_test) == len(VALID_STATS) + len(add_stats)
        assert len(out_expected) == len(VALID_STATS) + len(add_stats)
        for k, v in out_test.items():
            assert k.startswith("test_")
            assert np.isclose(v, out_expected[k], rtol=1.0e-6, atol=1.0e-8)


def test_parallel_zonal_stats_with_client(sc_dir, zonal_polygon_fp):
    """Test parallel compute of zonal stats without a dask client"""

    zs = ZonalStats(stats=VALID_STATS, all_touched=True)
    truth_stats = zs.from_files(
        zonal_polygon_fp, sc_dir / "layer_a.tif", parallel=False
    )
    truth_stats = list(truth_stats)

    with Client() as client:
        client.get_task_stream()
        test_stats = zs.from_files(
            zonal_polygon_fp, sc_dir / "layer_a.tif", parallel=True
        )
        test_stats = list(test_stats)
        assert client.get_task_stream()

    assert test_stats == truth_stats


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
