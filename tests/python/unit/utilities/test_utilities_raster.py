"""Tests for base reVRt utilities"""

from pathlib import Path

import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union
from rasterio.transform import Affine

from revrt.utilities.raster import rasterize_shape_file


@pytest.mark.parametrize("at", [True, False])
def test_basic_rasterize_shape_file(tmp_path, at):
    """Test basic shapefile rasterization"""
    land_mask_fp = tmp_path / "test_basic_shape_mask.gpkg"
    basic_shape = gpd.GeoDataFrame(
        geometry=[unary_union([box(0, -10, 10, 0), box(5, 0, 10, 5)])],
        crs="ESRI:102008",
    )
    basic_shape.to_file(land_mask_fp, driver="GPKG")

    out = rasterize_shape_file(
        land_mask_fp,
        width=6,
        height=5,
        transform=Affine(5.0, 0.0, -12.5, 0.0, -5.0, 12.5),
        buffer_dist=None,
        all_touched=at,
        dest_crs=None,
        burn_value=1,
        boundary_only=False,
        dtype="uint8",
    )

    assert out.shape == (5, 6)
    assert out.dtype == "uint8"
    assert out.max() == 1
    assert out.min() == 0
    assert out.sum() == 11 if at else 7


def test_basic_rasterize_shape_file_with_opts(tmp_path):
    """Test basic shapefile rasterization with other options"""
    land_mask_fp = tmp_path / "test_basic_shape_mask.gpkg"
    basic_shape = gpd.GeoDataFrame(
        geometry=[unary_union([box(0, -10, 10, 0), box(5, 0, 10, 5)])],
        crs="ESRI:102008",
    )
    basic_shape.to_file(land_mask_fp, driver="GPKG")

    out = rasterize_shape_file(
        land_mask_fp,
        width=6,
        height=5,
        transform=Affine(5.0, 0.0, -12.5, 0.0, -5.0, 12.5),
        buffer_dist=None,
        all_touched=False,
        burn_value=9,
        boundary_only=True,
        dtype="uint8",
    )

    assert np.allclose(
        out,
        np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 9, 9, 0],
                [0, 0, 9, 9, 9, 0],
                [0, 0, 9, 0, 9, 0],
                [0, 0, 9, 9, 9, 0],
            ]
        ),
    )

    out = rasterize_shape_file(
        land_mask_fp,
        width=6,
        height=5,
        transform=Affine(5.0, 0.0, -12.5, 0.0, -5.0, 12.5),
        buffer_dist=5,
        all_touched=False,
        burn_value=9,
        boundary_only=True,
        dtype="uint8",
    )

    assert np.allclose(
        out,
        np.array(
            [
                [0, 0, 9, 9, 9, 9],
                [0, 9, 9, 0, 0, 9],
                [0, 9, 0, 0, 0, 9],
                [0, 9, 0, 0, 0, 9],
                [0, 9, 0, 0, 0, 9],
            ]
        ),
    )

    # Make sure buffer in previous step doesn't affect re-runs
    out = rasterize_shape_file(
        land_mask_fp,
        width=6,
        height=5,
        transform=Affine(5.0, 0.0, -12.5, 0.0, -5.0, 12.5),
        buffer_dist=None,
        all_touched=False,
        burn_value=9,
        boundary_only=True,
        dtype="uint8",
    )

    assert np.allclose(
        out,
        np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 9, 9, 0],
                [0, 0, 9, 9, 9, 0],
                [0, 0, 9, 0, 9, 0],
                [0, 0, 9, 9, 9, 0],
            ]
        ),
    )


def test_rasterize_with_reproject(tmp_path):
    """Test basic shapefile rasterization with reprojecting"""
    land_mask_fp = tmp_path / "test_basic_shape_mask.gpkg"
    basic_shape = gpd.GeoDataFrame(
        geometry=[unary_union([box(0, -10, 10, 0), box(5, 0, 10, 5)])],
        crs="ESRI:102008",
    )
    basic_shape.to_file(land_mask_fp, driver="GPKG")

    out = rasterize_shape_file(
        land_mask_fp,
        width=6,
        height=5,
        transform=Affine(5.0, 0.0, -12.5, 0.0, -5.0, 12.5),
        buffer_dist=None,
        all_touched=False,
        dest_crs="EPSG:4326",
        burn_value=1,
        boundary_only=False,
        dtype="uint8",
    )

    assert out.shape == (5, 6)
    assert out.dtype == "uint8"
    assert out.max() == 0
    assert out.min() == 0
    assert out.sum() == 0


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
