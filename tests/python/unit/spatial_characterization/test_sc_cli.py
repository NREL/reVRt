"""Test revrt spatial characterization CLI"""

import os
import json
import platform
from pathlib import Path

import pytest
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
from rasterio.transform import Affine
from shapely.geometry import box, LineString

from revrt.spatial_characterization.stats import (
    Stat,
    FractionalStat,
    _PCT_PREFIX,
)
from revrt.spatial_characterization.cli import buffered_route_characterizations
from revrt._cli import main


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


def test_buffered_route_characterizations(tmp_path, sample_raster):
    """Test running stats through buffered characterizations function"""
    raster_fp = tmp_path / "test.tif"
    zones_fp = tmp_path / "test.gpkg"

    zones = gpd.GeoDataFrame(
        {"id": [1, 2], "A": ["a", "b"]},
        geometry=[box(-5, -5, 5, 5), LineString([(10, -7), (10, 13)])],
    )
    zones = zones.set_crs(sample_raster.attrs["crs"])

    sample_raster.rio.to_raster(raster_fp)
    zones.to_file(zones_fp, driver="GPKG")

    out_stats = buffered_route_characterizations(
        raster_fp,
        zones_fp,
        row_widths={1: 200, 2: 8},
        row_width_key="id",
        stats="*",
    )

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


def test_buffered_route_characterizations_with_multiplier(
    tmp_path, sample_raster
):
    """Test running stats with a scalar multiplier"""
    raster_fp = tmp_path / "test.tif"
    zones_fp = tmp_path / "test.gpkg"

    zones = gpd.GeoDataFrame(
        {"id": [1, 2], "A": ["a", "b"]},
        geometry=[box(-5, -5, 5, 5), LineString([(10, -7), (10, 13)])],
    )
    zones = zones.set_crs(sample_raster.attrs["crs"])

    sample_raster.rio.to_raster(raster_fp)
    zones.to_file(zones_fp, driver="GPKG")

    out_stats = buffered_route_characterizations(
        raster_fp,
        zones_fp,
        row_widths={1: 200, 2: 8},
        multiplier_scalar=3,
        row_width_key="id",
        stats="*",
    )

    assert len(out_stats) == len(zones)

    scaled_raster = sample_raster * 3
    sub_arr = scaled_raster.isel(x=2)
    assert np.allclose(
        out_stats[Stat.COUNT], [scaled_raster.count(), sub_arr.count()]
    )
    assert np.allclose(
        out_stats[Stat.MIN], [scaled_raster.min(), sub_arr.min()]
    )
    assert np.allclose(
        out_stats[Stat.MAX], [scaled_raster.max(), sub_arr.max()]
    )
    assert np.allclose(
        out_stats[Stat.MEAN], [scaled_raster.mean(), sub_arr.mean()]
    )
    assert np.allclose(
        out_stats[Stat.SUM], [scaled_raster.sum(), sub_arr.sum()]
    )
    assert np.allclose(
        out_stats[Stat.STD], [scaled_raster.std(), sub_arr.std()]
    )
    assert np.allclose(
        out_stats[Stat.MEDIAN], [scaled_raster.median(), sub_arr.median()]
    )
    assert np.allclose(out_stats[Stat.MAJORITY], 15)
    assert np.allclose(out_stats[Stat.MINORITY], [12, 27])
    assert np.allclose(out_stats[Stat.UNIQUE], [4, 2])
    assert np.allclose(out_stats[Stat.RANGE], [24, 12])
    assert np.allclose(out_stats[Stat.NODATA], 0)
    assert np.allclose(
        out_stats[f"{Stat.PIXEL_COUNT}_3.0"], [2, np.nan], equal_nan=True
    )
    assert np.allclose(
        out_stats[f"{Stat.PIXEL_COUNT}_12.0"], [1, np.nan], equal_nan=True
    )
    assert np.allclose(out_stats[f"{Stat.PIXEL_COUNT}_15.0"], [3, 2])
    assert np.allclose(out_stats[f"{Stat.PIXEL_COUNT}_27.0"], [3, 1])

    assert np.allclose(
        out_stats[f"{FractionalStat.FRACTIONAL_PIXEL_COUNT}_3.0"],
        [2, np.nan],
        equal_nan=True,
    )
    assert np.allclose(
        out_stats[f"{FractionalStat.FRACTIONAL_PIXEL_COUNT}_12.0"],
        [1, np.nan],
        equal_nan=True,
    )
    assert np.allclose(
        out_stats[f"{FractionalStat.FRACTIONAL_PIXEL_COUNT}_15.0"],
        [3, 0.8 + 0.64],
    )
    assert np.allclose(
        out_stats[f"{FractionalStat.FRACTIONAL_PIXEL_COUNT}_27.0"],
        [3, 0.16],
    )
    assert np.allclose(
        out_stats[f"{FractionalStat.FRACTIONAL_AREA}_3.0"],
        [200, np.nan],
        equal_nan=True,
    )
    assert np.allclose(
        out_stats[f"{FractionalStat.FRACTIONAL_AREA}_12.0"],
        [100, np.nan],
        equal_nan=True,
    )
    assert np.allclose(
        out_stats[f"{FractionalStat.FRACTIONAL_AREA}_15.0"], [300, 80 + 64]
    )
    assert np.allclose(
        out_stats[f"{FractionalStat.FRACTIONAL_AREA}_27.0"], [300, 16]
    )
    assert np.allclose(
        out_stats[FractionalStat.VALUE_MULTIPLIED_BY_FRACTIONAL_AREA],
        [scaled_raster.sum() * 100, 15 * (80 + 64) + 27 * 16],
    )


def test_buffered_route_characterizations_percentile(tmp_path, sample_raster):
    """Test running percentile stats"""
    raster_fp = tmp_path / "test.tif"
    zones_fp = tmp_path / "test.gpkg"

    zones = gpd.GeoDataFrame(
        {"id": [1, 2], "A": [50, 42]},
        geometry=[box(-5, -5, 5, 5), LineString([(10, -7), (10, 13)])],
    )
    zones = zones.set_crs(sample_raster.attrs["crs"])

    sample_raster.rio.to_raster(raster_fp)
    zones.to_file(zones_fp, driver="GPKG")

    out_stats = buffered_route_characterizations(
        raster_fp,
        zones_fp,
        row_widths={"50": 200, 42: 8},
        row_width_key="A",
        stats=[f"{_PCT_PREFIX}50", f"{_PCT_PREFIX}95"],
    )

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


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_cli_command_minimal(tmp_cwd, sample_raster, cli_runner):
    """Test running from config with minimal user inputs"""
    raster_fp = tmp_cwd / "test_raster.tif"
    zones_fp = tmp_cwd / "test_zones.gpkg"

    zones = gpd.GeoDataFrame(
        {"voltage": [1, 2], "A": ["a", "b"]},
        geometry=[box(-5, -5, 5, 5), LineString([(10, -7), (10, 13)])],
    )
    zones = zones.set_crs(sample_raster.attrs["crs"])

    sample_raster.rio.to_raster(raster_fp)
    zones.to_file(zones_fp, driver="GPKG")

    config = {
        "execution_control": {"option": "local"},
        "layers": {
            "geotiff_fp": str(raster_fp),
            "route_fp": str(zones_fp),
        },
        "row_widths": {"1": 200, "2": 8},
    }
    config_fp = tmp_cwd / "config.json"
    with config_fp.open("w", encoding="utf-8") as f:
        json.dump(config, f)

    assert not list(tmp_cwd.glob("*.csv"))
    cli_runner.invoke(
        main, ["route-characterization", "-c", config_fp.as_posix()]
    )

    out_files = list(tmp_cwd.glob("*.csv"))
    assert len(out_files) == 1

    out_fp = Path(out_files[0])
    assert out_fp.name == "characterized_test_raster_test_zones.csv"

    out_stats = pd.read_csv(out_fp)

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
    assert np.allclose(out_stats["voltage"], [1, 2])
    assert out_stats["A"].to_list() == ["a", "b"]


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_cli_command_multiple_rasters(tmp_cwd, sample_raster, cli_runner):
    """Test running from config with multiple raster inputs"""
    raster_fp = tmp_cwd / "raster.tif"
    zones_fp = tmp_cwd / "lcp.gpkg"

    zones = gpd.GeoDataFrame(
        {"voltage": [1, 2], "A": ["a", "b"]},
        geometry=[box(-5, -5, 5, 5), LineString([(10, -7), (10, 13)])],
    )
    zones = zones.set_crs(sample_raster.attrs["crs"])

    sample_raster.rio.to_raster(raster_fp)
    zones.to_file(zones_fp, driver="GPKG")

    row_widths = {"1": 200, "2": 8}
    row_widths_fp = tmp_cwd / "row_widths.json"
    with row_widths_fp.open("w", encoding="utf-8") as f:
        json.dump(row_widths, f)

    config = {
        "execution_control": {"option": "local"},
        "layers": [
            {
                "geotiff_fp": str(raster_fp),
                "route_fp": str(zones_fp),
                "stats": "count min",
            },
            {
                "geotiff_fp": str(raster_fp),
                "route_fp": str(zones_fp),
                "prefix": "test_",
                "stats": "max mean",
                "copy_properties": ["A"],
            },
        ],
        "row_widths": str(row_widths_fp),
    }
    config_fp = tmp_cwd / "config.json"
    with config_fp.open("w", encoding="utf-8") as f:
        json.dump(config, f)

    assert not list(tmp_cwd.glob("*.csv"))
    cli_runner.invoke(
        main, ["route-characterization", "-c", config_fp.as_posix()]
    )

    out_files = sorted(tmp_cwd.glob("*.csv"))
    assert len(out_files) == 2

    out_fp = Path(out_files[0])
    assert out_fp.name == "characterized_raster_lcp_j0.csv"

    out_stats = pd.read_csv(out_fp)
    sub_arr = sample_raster.isel(x=2)

    assert np.allclose(
        out_stats[Stat.COUNT], [sample_raster.count(), sub_arr.count()]
    )
    assert np.allclose(
        out_stats[Stat.MIN], [sample_raster.min(), sub_arr.min()]
    )
    assert np.allclose(out_stats["voltage"], [1, 2])
    assert out_stats["A"].to_list() == ["a", "b"]
    assert not any(c in out_stats for c in [Stat.MAX, Stat.MEAN])

    out_fp = Path(out_files[1])
    assert out_fp.name == "characterized_raster_lcp_j1.csv"

    out_stats = pd.read_csv(out_fp)

    assert np.allclose(
        out_stats[f"test_{Stat.MAX}"], [sample_raster.max(), sub_arr.max()]
    )
    assert np.allclose(
        out_stats[f"test_{Stat.MEAN}"], [sample_raster.mean(), sub_arr.mean()]
    )
    assert out_stats["A"].to_list() == ["a", "b"]
    assert not any(
        c in out_stats
        for c in [
            Stat.COUNT,
            Stat.MIN,
            f"test_{Stat.COUNT}",
            f"test_{Stat.MIN}",
            "voltage",
        ]
    )


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
