"""Test TreV spatial characterization CLI"""

from pathlib import Path

import pytest
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
from rasterio.transform import Affine
from shapely.geometry import box, LineString

from trev.spatial_characterization.stats import (
    Stat,
    FractionalStat,
    _PCT_PREFIX,  # noqa: PLC2701
)
from trev.spatial_characterization.cli import buffered_lcp_stats


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
        attrs={
            "transform": Affine(10.0, 0.0, -15, 0.0, -10.0, 15),
            "crs": "ESRI:102008",
        },
    )


def test_buffered_lcp_stats(tmp_path, sample_raster):
    """Test running stats through buffered stats function"""
    raster_fp = tmp_path / "test.tif"
    zones_fp = tmp_path / "test.gpkg"
    out_fp = tmp_path / "test.csv"

    zones = gpd.GeoDataFrame(
        {"id": [1, 2], "A": ["a", "b"]},
        geometry=[box(-5, -5, 5, 5), LineString([(10, -7), (10, 13)])],
    )
    zones = zones.set_crs(sample_raster.attrs["crs"])

    sample_raster.rio.to_raster(raster_fp)
    zones.to_file(zones_fp, driver="GPKG")

    assert not out_fp.exists()
    buffered_lcp_stats(
        raster_fp,
        zones_fp,
        row_widths={1: 200, 2: 8},
        out_fp=out_fp,
        row_width_key="id",
        stats="*",
    )

    assert out_fp.exists()

    out_stats = pd.read_csv(out_fp)
    assert len(out_stats) == len(zones)

    sub_arr = sample_raster.isel(x=2)
    assert np.allclose(
        out_stats[Stat.COUNT], [sample_raster.count(), sub_arr.count()]
    )
    assert np.allclose(
        out_stats[Stat.MIN], [sample_raster.min(), sub_arr.min()]
    )
    assert np.allclose(
        out_stats[Stat.MAX], [sample_raster.max(), sub_arr.max()]
    )
    assert np.allclose(
        out_stats[Stat.MEAN], [sample_raster.mean(), sub_arr.mean()]
    )
    assert np.allclose(
        out_stats[Stat.SUM], [sample_raster.sum(), sub_arr.sum()]
    )
    assert np.allclose(
        out_stats[Stat.STD], [sample_raster.std(), sub_arr.std()]
    )
    assert np.allclose(
        out_stats[Stat.MEDIAN], [sample_raster.median(), sub_arr.median()]
    )
    assert np.allclose(out_stats[Stat.MAJORITY], 5)
    assert np.allclose(out_stats[Stat.MINORITY], [4, 9])
    assert np.allclose(out_stats[Stat.UNIQUE], [4, 2])
    assert np.allclose(out_stats[Stat.RANGE], [8, 4])
    assert np.allclose(out_stats[Stat.NODATA], 0)
    assert np.allclose(
        out_stats[f"{Stat.PIXEL_COUNT}_1.0"], [2, np.nan], equal_nan=True
    )
    assert np.allclose(
        out_stats[f"{Stat.PIXEL_COUNT}_4.0"], [1, np.nan], equal_nan=True
    )
    assert np.allclose(out_stats[f"{Stat.PIXEL_COUNT}_5.0"], [3, 2])
    assert np.allclose(out_stats[f"{Stat.PIXEL_COUNT}_9.0"], [3, 1])

    assert np.allclose(
        out_stats[f"{FractionalStat.FRACTIONAL_PIXEL_COUNT}_1.0"],
        [2, np.nan],
        equal_nan=True,
    )
    assert np.allclose(
        out_stats[f"{FractionalStat.FRACTIONAL_PIXEL_COUNT}_4.0"],
        [1, np.nan],
        equal_nan=True,
    )
    assert np.allclose(
        out_stats[f"{FractionalStat.FRACTIONAL_PIXEL_COUNT}_5.0"],
        [3, 0.8 + 0.64],
    )
    assert np.allclose(
        out_stats[f"{FractionalStat.FRACTIONAL_PIXEL_COUNT}_9.0"],
        [3, 0.16],
    )
    assert np.allclose(
        out_stats[f"{FractionalStat.FRACTIONAL_AREA}_1.0"],
        [200, np.nan],
        equal_nan=True,
    )
    assert np.allclose(
        out_stats[f"{FractionalStat.FRACTIONAL_AREA}_4.0"],
        [100, np.nan],
        equal_nan=True,
    )
    assert np.allclose(
        out_stats[f"{FractionalStat.FRACTIONAL_AREA}_5.0"], [300, 80 + 64]
    )
    assert np.allclose(
        out_stats[f"{FractionalStat.FRACTIONAL_AREA}_9.0"], [300, 16]
    )
    assert np.allclose(
        out_stats[FractionalStat.VALUE_MULTIPLIED_BY_FRACTIONAL_AREA],
        [sample_raster.sum() * 100, 5 * (80 + 64) + 9 * 16],
    )


def test_buffered_lcp_stats_percentile(tmp_path, sample_raster):
    """Test running percentile stats through buffered stats function"""
    raster_fp = tmp_path / "test.tif"
    zones_fp = tmp_path / "test.gpkg"
    out_fp = tmp_path / "test.csv"

    zones = gpd.GeoDataFrame(
        {"id": [1, 2], "A": ["a", "b"]},
        geometry=[box(-5, -5, 5, 5), LineString([(10, -7), (10, 13)])],
    )
    zones = zones.set_crs(sample_raster.attrs["crs"])

    sample_raster.rio.to_raster(raster_fp)
    zones.to_file(zones_fp, driver="GPKG")

    assert not out_fp.exists()
    buffered_lcp_stats(
        raster_fp,
        zones_fp,
        row_widths={"a": 200, "b": 8},
        out_fp=out_fp,
        row_width_key="A",
        stats=[f"{_PCT_PREFIX}50", f"{_PCT_PREFIX}95"],
    )

    assert out_fp.exists()

    out_stats = pd.read_csv(out_fp)
    assert len(out_stats) == len(zones)

    sub_arr = sample_raster.isel(x=2)
    assert np.allclose(
        out_stats[f"{_PCT_PREFIX}50"],
        [np.percentile(sample_raster, 50), np.percentile(sub_arr, 50)],
    )
    assert np.allclose(
        out_stats[f"{_PCT_PREFIX}95"],
        [np.percentile(sample_raster, 95), np.percentile(sub_arr, 95)],
    )


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
