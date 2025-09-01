"""Test masks for cost layer creation"""

from pathlib import Path

import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union
from rasterio.transform import Affine

from revrt.costs.masks import Masks
from revrt.exceptions import revrtAttributeError
from revrt.utilities import LayeredFile


def test_no_masks():
    """Test error when no masks"""
    masks = Masks((3, 3), "EPSG:4326", None, ".")
    with pytest.raises(revrtAttributeError, match="No mask available"):
        _ = masks.dry_mask

    with pytest.raises(revrtAttributeError, match="No mask available"):
        _ = masks.wet_mask

    with pytest.raises(revrtAttributeError, match="No mask available"):
        _ = masks.dry_plus_mask

    with pytest.raises(revrtAttributeError, match="No mask available"):
        _ = masks.wet_plus_mask

    with pytest.raises(revrtAttributeError, match="No mask available"):
        _ = masks.landfall_mask


def test_basic_shapefile_to_masks(tmp_path):
    """Test basic shapefile to masks"""
    land_mask_fp = tmp_path / "test_basic_shape_mask.gpkg"
    basic_shape = gpd.GeoDataFrame(
        geometry=[unary_union([box(0, -10, 10, 0), box(5, 0, 10, 5)])],
        crs="ESRI:102008",
    )
    basic_shape.to_file(land_mask_fp, driver="GPKG")

    masks = Masks(
        shape=(5, 6),
        crs="ESRI:102008",
        transform=Affine(5.0, 0.0, -12.5, 0.0, -5.0, 12.5),
        masks_dir=tmp_path,
    )

    masks.create(land_mask_fp, save_tiff=False, reproject_vector=False)
    assert len(list(tmp_path.glob("*.tif"))) == 0

    assert np.allclose(
        masks.landfall_mask,
        np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 0, 1, 0, 1, 0],
                [0, 0, 1, 1, 1, 0],
            ]
        ),
    )
    assert masks.landfall_mask.dtype == bool

    assert np.allclose(
        masks.wet_mask,
        np.array(
            [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0, 1],
                [1, 1, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 1],
            ]
        ),
    )
    assert masks.wet_mask.dtype == bool

    assert np.allclose(
        masks.dry_mask,
        np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        ),
    )
    assert masks.dry_mask.dtype == bool

    assert np.allclose(
        masks.dry_plus_mask,
        np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0],
            ]
        ),
    )
    assert masks.dry_plus_mask.dtype == bool

    assert np.allclose(
        masks.wet_plus_mask,
        np.array(
            [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 0, 1, 1],
                [1, 1, 1, 1, 1, 1],
            ]
        ),
    )
    assert masks.wet_plus_mask.dtype == bool


def test_loading_basic_masks(tmp_path):
    """Test basic loading of masks"""
    land_mask_fp = tmp_path / "test_basic_shape_mask.gpkg"
    layer_file_fp = tmp_path / "test_masks_layer_file.zarr"
    basic_shape = gpd.GeoDataFrame(
        geometry=[unary_union([box(0, -10, 10, 0), box(5, 0, 10, 5)])],
        crs="ESRI:102008",
    )
    basic_shape.to_file(land_mask_fp, driver="GPKG")

    masks = Masks(
        shape=(5, 6),
        crs="ESRI:102008",
        transform=Affine(5.0, 0.0, -12.5, 0.0, -5.0, 12.5),
        masks_dir=tmp_path,
    )

    masks.create(land_mask_fp, save_tiff=True, reproject_vector=False)
    assert len(list(tmp_path.glob("*.tif"))) == 4
    for fn in [
        Masks.LANDFALL_MASK_FNAME,
        Masks.RAW_LAND_MASK_FNAME,
        Masks.LAND_MASK_FNAME,
        Masks.OFFSHORE_MASK_FNAME,
    ]:
        assert (tmp_path / fn).exists()

    lf = LayeredFile(layer_file_fp)
    lf.create_new(tmp_path / Masks.LANDFALL_MASK_FNAME)

    new_masks = Masks(
        shape=(5, 6),
        crs="ESRI:102008",
        transform=Affine(5.0, 0.0, -12.5, 0.0, -5.0, 12.5),
        masks_dir=tmp_path,
    )
    new_masks.load(layer_file_fp)

    assert np.allclose(
        new_masks.landfall_mask,
        np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 0, 1, 0, 1, 0],
                [0, 0, 1, 1, 1, 0],
            ]
        ),
    )
    assert new_masks.landfall_mask.dtype == bool

    assert np.allclose(
        new_masks.wet_mask,
        np.array(
            [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0, 1],
                [1, 1, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 1],
            ]
        ),
    )
    assert new_masks.wet_mask.dtype == bool

    assert np.allclose(
        new_masks.dry_mask,
        np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        ),
    )
    assert new_masks.dry_mask.dtype == bool

    assert np.allclose(
        new_masks.dry_plus_mask,
        np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0],
            ]
        ),
    )
    assert new_masks.dry_plus_mask.dtype == bool

    assert np.allclose(
        new_masks.wet_plus_mask,
        np.array(
            [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 0, 1, 1],
                [1, 1, 1, 1, 1, 1],
            ]
        ),
    )
    assert new_masks.wet_plus_mask.dtype == bool


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
