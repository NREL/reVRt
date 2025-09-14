"""Test masks for cost layer creation"""

import json
import traceback
from pathlib import Path

import pytest
import rioxarray
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union

from revrt._cli import main
from revrt.constants import BARRIER_H5_LAYER_NAME
from revrt.costs.cli import build_routing_layers
from revrt.costs.masks import Masks
from revrt.exceptions import revrtConfigurationError


@pytest.fixture(scope="module")
def tiff_layers_for_testing(sample_tiff_props, tmp_path_factory):
    """Test TIFF layers for testing LayerCreator"""
    layer_dir = tmp_path_factory.mktemp("layers")

    x0, y0, width, height, cell_size, transform = sample_tiff_props

    layers = {
        "friction_1.tif": np.array([[ind] * width for ind in range(height)]),
        "fi_1.tif": np.array([list(range(width))] * height),
    }

    for layer_fn, data in layers.items():
        da = xr.DataArray(
            data,
            dims=("y", "x"),
            coords={
                "x": x0 + np.arange(width) * cell_size + cell_size / 2,
                "y": y0 - np.arange(height) * cell_size - cell_size / 2,
            },
            name="test_band",
        )

        da = da.rio.write_crs("ESRI:102008")
        da.rio.write_transform(transform)
        da.rio.to_raster(layer_dir / layer_fn, driver="GTiff")

    return layer_dir, layers


@pytest.fixture(scope="module")
def masks_for_testing(sample_tiff_props, tmp_path_factory):
    """Masks for testing build function"""
    *__, width, height, ___, transform = sample_tiff_props

    masks_dir = tmp_path_factory.mktemp("masks")
    land_mask_fp = masks_dir / "test_basic_shape_mask.gpkg"
    basic_shape = gpd.GeoDataFrame(
        geometry=[unary_union([box(0, -10, 10, 0), box(5, 0, 10, 5)])],
        crs="ESRI:102008",
    )
    basic_shape.to_file(land_mask_fp, driver="GPKG")

    masks = Masks(
        shape=(height, width),
        crs="ESRI:102008",
        transform=transform,
        masks_dir=masks_dir,
    )
    masks.create(land_mask_fp, save_tiff=True, reproject_vector=True)
    return masks


def test_build_config_missing_action(tmp_path):
    """Test correct error is raised for config with no actions"""
    tiff_fp = tmp_path / "nonexistent.tif"
    tiff_fp.touch()

    with pytest.raises(
        revrtConfigurationError,
        match=r"At least one of .* must be in the config file",
    ):
        build_routing_layers(
            routing_file=tmp_path / "test.zarr", template_file=tiff_fp
        )


@pytest.mark.parametrize("mw", [None, 1, 2])
def test_build_basic_all(
    tmp_path,
    sample_iso_fp,
    sample_nlcd_fp,
    sample_slope_fp,
    sample_extra_fp,
    tiff_layers_for_testing,
    masks_for_testing,
    mw,
):
    """Test basic building of layers, dry costs, and merging"""
    test_fp = tmp_path / "test.zarr"
    out_tiff_dir = tmp_path / "out_tiffs"
    layer_dir, layers = tiff_layers_for_testing

    assert not test_fp.exists()
    assert not out_tiff_dir.exists()

    config = {
        "routing_file": test_fp,
        "template_file": sample_extra_fp,
        "input_layer_dir": layer_dir,
        "output_tiff_dir": out_tiff_dir,
        "masks_dir": masks_for_testing._masks_dir,
        "layers": [
            {
                "layer_name": "fi_1",
                "include_in_file": False,
                "build": {
                    "fi_1.tif": {"extent": "wet+", "pass_through": True}
                },
            },
            {
                "layer_name": "friction",
                "build": {
                    "friction_1.tif": {
                        "extent": "dry+",
                        "map": {x: x for x in range(20)},
                    }
                },
            },
        ],
        "dry_costs": {
            "iso_region_tiff": sample_iso_fp,
            "nlcd_tiff": sample_nlcd_fp,
            "slope_tiff": sample_slope_fp,
            "extra_tiffs": [sample_extra_fp],
        },
        "merge_friction_and_barriers": {
            "friction_layer": "friction",
            "barrier_layer": "fi_1",
            "barrier_multiplier": 100,
        },
    }

    build_routing_layers(**config, max_workers=mw)

    assert test_fp.exists()
    assert out_tiff_dir.exists()
    assert (out_tiff_dir / "fi_1.tif").exists()
    assert (out_tiff_dir / "friction.tif").exists()
    assert (out_tiff_dir / f"{BARRIER_H5_LAYER_NAME}.tif").exists()

    expected_datasets = [
        "sample_nlcd",
        "sample_iso",
        "sample_slope",
        "sample_extra_data",
        "friction",
        "dry_multipliers",
        "tie_line_costs_102MW",
        "tie_line_costs_205MW",
        "tie_line_costs_400MW",
        "tie_line_costs_1500MW",
        "tie_line_costs_3000MW",
    ]
    with xr.open_dataset(test_fp, consolidated=False, engine="zarr") as ds:
        for ds_name in expected_datasets:
            assert ds_name in ds

        assert "fi_1" not in ds
        assert "friction_1" not in ds
        assert np.allclose(
            ds["friction"],
            layers["friction_1.tif"] * masks_for_testing.dry_plus_mask,
        )

    with rioxarray.open_rasterio(
        out_tiff_dir / f"{BARRIER_H5_LAYER_NAME}.tif", chunks="auto"
    ) as ds:
        assert np.allclose(
            ds,
            layers["friction_1.tif"] * masks_for_testing.dry_plus_mask
            + layers["fi_1.tif"] * masks_for_testing.wet_plus_mask * 100,
        )


def test_build_dry_only(
    tmp_path,
    sample_iso_fp,
    sample_nlcd_fp,
    sample_slope_fp,
    sample_extra_fp,
    masks_for_testing,
):
    """Test building only dry costs"""
    test_fp = tmp_path / "test.zarr"
    out_tiff_dir = tmp_path / "out_tiffs"

    assert not test_fp.exists()
    assert not out_tiff_dir.exists()

    config = {
        "routing_file": str(test_fp),
        "template_file": str(sample_extra_fp),
        "output_tiff_dir": str(out_tiff_dir),
        "masks_dir": str(masks_for_testing._masks_dir),
        "dry_costs": {
            "iso_region_tiff": str(sample_iso_fp),
            "nlcd_tiff": str(sample_nlcd_fp),
            "slope_tiff": str(sample_slope_fp),
            "extra_tiffs": [str(sample_extra_fp)],
        },
    }

    build_routing_layers(**config)

    assert test_fp.exists()
    assert out_tiff_dir.exists()
    assert not (out_tiff_dir / "fi_1.tif").exists()
    assert not (out_tiff_dir / "friction.tif").exists()
    assert not (out_tiff_dir / f"{BARRIER_H5_LAYER_NAME}.tif").exists()

    expected_datasets = [
        "sample_nlcd",
        "sample_iso",
        "sample_slope",
        "sample_extra_data",
        "dry_multipliers",
        "tie_line_costs_102MW",
        "tie_line_costs_205MW",
        "tie_line_costs_400MW",
        "tie_line_costs_1500MW",
        "tie_line_costs_3000MW",
    ]
    with xr.open_dataset(test_fp, consolidated=False, engine="zarr") as ds:
        for ds_name in expected_datasets:
            assert ds_name in ds

        assert "fi_1" not in ds
        assert "friction_1" not in ds
        assert "friction" not in ds


def test_build_layers_only(
    tmp_path, sample_extra_fp, tiff_layers_for_testing, masks_for_testing
):
    """Test building only layers"""
    test_fp = tmp_path / "test.zarr"
    out_tiff_dir = tmp_path / "out_tiffs"
    layer_dir, layers = tiff_layers_for_testing

    assert not test_fp.exists()
    assert not out_tiff_dir.exists()

    config = {
        "routing_file": str(test_fp),
        "template_file": str(sample_extra_fp),
        "input_layer_dir": str(layer_dir),
        "output_tiff_dir": str(out_tiff_dir),
        "masks_dir": str(masks_for_testing._masks_dir),
        "layers": [
            {
                "layer_name": "fi_1",
                "include_in_file": False,
                "build": {
                    "fi_1.tif": {"extent": "wet+", "pass_through": True}
                },
            },
            {
                "layer_name": "friction",
                "build": {
                    "friction_1.tif": {
                        "extent": "dry+",
                        "map": {x: x for x in range(20)},
                    }
                },
            },
        ],
    }

    build_routing_layers(**config)

    assert test_fp.exists()
    assert out_tiff_dir.exists()
    assert (out_tiff_dir / "fi_1.tif").exists()
    assert (out_tiff_dir / "friction.tif").exists()
    assert not (out_tiff_dir / f"{BARRIER_H5_LAYER_NAME}.tif").exists()

    expected_missing_datasets = [
        "sample_nlcd",
        "sample_iso",
        "sample_slope",
        "sample_extra_data",
        "dry_multipliers",
        "tie_line_costs_102MW",
        "tie_line_costs_205MW",
        "tie_line_costs_400MW",
        "tie_line_costs_1500MW",
        "tie_line_costs_3000MW",
    ]
    with xr.open_dataset(test_fp, consolidated=False, engine="zarr") as ds:
        for ds_name in expected_missing_datasets:
            assert ds_name not in ds

        assert "friction" in ds
        assert "fi_1" not in ds
        assert "friction_1" not in ds
        assert np.allclose(
            ds["friction"],
            layers["friction_1.tif"] * masks_for_testing.dry_plus_mask,
        )

    # Test adding one more layer
    config = {
        "routing_file": str(test_fp),
        "template_file": str(sample_extra_fp),
        "input_layer_dir": str(layer_dir),
        "output_tiff_dir": str(out_tiff_dir),
        "masks_dir": str(masks_for_testing._masks_dir),
        "layers": [
            {
                "layer_name": "fi_1",
                "include_in_file": False,
                "build": {
                    "fi_1.tif": {"extent": "wet+", "pass_through": True}
                },
            },
            {
                "layer_name": "friction",
                "build": {
                    "friction_1.tif": {
                        "extent": "dry+",
                        "map": {x: x for x in range(20)},
                    }
                },
            },
            {
                "layer_name": "fi_1",
                "include_in_file": True,
                "build": {"fi_1.tif": {"pass_through": True}},
            },
        ],
    }

    build_routing_layers(**config)

    with xr.open_dataset(test_fp, consolidated=False, engine="zarr") as ds:
        for ds_name in expected_missing_datasets:
            assert ds_name not in ds

        assert "friction" in ds
        assert "fi_1" in ds
        assert "friction_1" not in ds
        assert np.allclose(
            ds["friction"],
            layers["friction_1.tif"] * masks_for_testing.dry_plus_mask,
        )
        assert np.allclose(ds["fi_1"], layers["fi_1.tif"])


def test_build_basic_from_cli(
    tmp_path,
    sample_iso_fp,
    sample_nlcd_fp,
    sample_slope_fp,
    sample_extra_fp,
    tiff_layers_for_testing,
    masks_for_testing,
    cli_runner,
):
    """Test basic building from command line"""
    test_fp = tmp_path / "test.zarr"
    out_tiff_dir = tmp_path / "out_tiffs"
    layer_dir, layers = tiff_layers_for_testing

    assert not test_fp.exists()
    assert not out_tiff_dir.exists()

    config = {
        "execution_control": {"max_workers": 1},
        "routing_file": str(test_fp),
        "template_file": str(sample_extra_fp),
        "input_layer_dir": str(layer_dir),
        "output_tiff_dir": str(out_tiff_dir),
        "masks_dir": str(masks_for_testing._masks_dir),
        "layers": [
            {
                "layer_name": "fi_1",
                "include_in_file": False,
                "build": {
                    "fi_1.tif": {"extent": "wet+", "pass_through": True}
                },
            },
            {
                "layer_name": "friction",
                "build": {
                    "friction_1.tif": {
                        "extent": "dry+",
                        "map": {x: x for x in range(20)},
                    }
                },
            },
        ],
        "dry_costs": {
            "iso_region_tiff": str(sample_iso_fp),
            "nlcd_tiff": str(sample_nlcd_fp),
            "slope_tiff": str(sample_slope_fp),
            "extra_tiffs": [str(sample_extra_fp)],
        },
        "merge_friction_and_barriers": {
            "friction_layer": "friction",
            "barrier_layer": "fi_1",
            "barrier_multiplier": 100,
        },
    }

    config_path = tmp_path / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f)

    result = cli_runner.invoke(
        main, ["build-routing-layers", "-c", str(config_path)]
    )
    msg = f"Failed with error {traceback.print_exception(*result.exc_info)}"
    assert result.exit_code == 0, msg

    assert test_fp.exists()
    assert out_tiff_dir.exists()
    assert (out_tiff_dir / "fi_1.tif").exists()
    assert (out_tiff_dir / "friction.tif").exists()
    assert (out_tiff_dir / f"{BARRIER_H5_LAYER_NAME}.tif").exists()

    expected_datasets = [
        "sample_nlcd",
        "sample_iso",
        "sample_slope",
        "sample_extra_data",
        "friction",
        "dry_multipliers",
        "tie_line_costs_102MW",
        "tie_line_costs_205MW",
        "tie_line_costs_400MW",
        "tie_line_costs_1500MW",
        "tie_line_costs_3000MW",
    ]
    with xr.open_dataset(test_fp, consolidated=False, engine="zarr") as ds:
        for ds_name in expected_datasets:
            assert ds_name in ds

        assert "fi_1" not in ds
        assert "friction_1" not in ds
        assert np.allclose(
            ds["friction"],
            layers["friction_1.tif"] * masks_for_testing.dry_plus_mask,
        )

    with rioxarray.open_rasterio(
        out_tiff_dir / f"{BARRIER_H5_LAYER_NAME}.tif", chunks="auto"
    ) as ds:
        assert np.allclose(
            ds,
            layers["friction_1.tif"] * masks_for_testing.dry_plus_mask
            + layers["fi_1.tif"] * masks_for_testing.wet_plus_mask * 100,
        )


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
