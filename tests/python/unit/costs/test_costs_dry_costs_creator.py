"""Test dry cost layer creation"""

from pathlib import Path

import pytest
import rioxarray
import numpy as np
import xarray as xr

from revrt.constants import METERS_IN_MILE
from revrt.costs.dry_costs_creator import (
    DEFAULT_HILL_MULTIPLIER,
    DEFAULT_MTN_MULTIPLIER,
    DEFAULT_HILL_SLOPE,
    DEFAULT_MTN_SLOPE,
    DryCostsCreator,
    compute_slope_multipliers,
    compute_land_use_multipliers,
)
from revrt.utilities import LayeredFile
from revrt.exceptions import revrtValueError


def test_compute_slope_multipliers_defaults():
    """Test compute_slope_multipliers with default config values"""
    input_slopes = np.array(
        [
            [
                0.9 * DEFAULT_HILL_SLOPE,
                DEFAULT_HILL_SLOPE,
                1.1 * DEFAULT_HILL_SLOPE,
            ],
            [
                0.9 * DEFAULT_MTN_SLOPE,
                DEFAULT_MTN_SLOPE,
                1.1 * DEFAULT_MTN_SLOPE,
            ],
        ]
    )

    slope_multipliers = compute_slope_multipliers(input_slopes, chunks=(1, 3))
    expected_multipliers = np.array(
        [
            [1.0, DEFAULT_HILL_MULTIPLIER, DEFAULT_HILL_MULTIPLIER],
            [1.0, DEFAULT_MTN_MULTIPLIER, DEFAULT_MTN_MULTIPLIER],
        ]
    )
    assert np.allclose(slope_multipliers, expected_multipliers)


def test_compute_slope_multipliers_custom():
    """Test compute_slope_multipliers with custom config values"""
    hill_slope, mountain_slope = 10, 20
    hill_multiplier, mountain_multiplier = 3.0, 4.0
    input_slopes = np.array([[1, 10, 15], [1, 20, 30]])

    config = {
        "hill_slope": hill_slope,
        "mtn_slope": mountain_slope,
        "hill_mult": hill_multiplier,
        "mtn_mult": mountain_multiplier,
    }

    slope_multipliers = compute_slope_multipliers(
        input_slopes, chunks=(1, 3), config=config
    )
    expected_multipliers = np.array(
        [
            [1.0, hill_multiplier, hill_multiplier],
            [1.0, mountain_multiplier, mountain_multiplier],
        ]
    )
    assert np.allclose(slope_multipliers, expected_multipliers)


def test_compute_land_use_multipliers():
    """Test compute_land_use_multipliers with custom config values"""
    input_classes = np.array([[1, 2, 3], [3, 1, 4]])
    land_use_classes = {"TestClassA": [1, 2], "TestClassB": [3]}
    multipliers = {"TestClassA": 1.5, "TestClassB": 2.0}

    land_use_multipliers = compute_land_use_multipliers(
        input_classes, multipliers, land_use_classes, chunks=(1, 3)
    )
    expected_multipliers = np.array(
        [
            [
                multipliers["TestClassA"],
                multipliers["TestClassA"],
                multipliers["TestClassB"],
            ],
            [multipliers["TestClassB"], multipliers["TestClassA"], 1.0],
        ]
    )
    assert np.allclose(land_use_multipliers, expected_multipliers)


def test_compute_land_use_multipliers_missing_class_mapping():
    """Test for error when class mapping is missing"""
    input_classes = np.array([[1, 2, 3], [3, 1, 4]])
    land_use_classes = {"TestClassA": [1, 2], "TestClassB": [3]}
    multipliers = {"TestClassA": 1.5, "TestClassB": 2.0, "TestClassC": 2.5}

    with pytest.raises(
        revrtValueError,
        match="Class TestClassC not in land_use_classes:",
    ):
        compute_land_use_multipliers(
            input_classes, multipliers, land_use_classes, chunks=(1, 3)
        )


def test_compute_land_use_multipliers_bad_class_mapping():
    """Test for error when class mapping is not a list"""
    input_classes = np.array([[1, 2, 3], [3, 1, 4]])
    land_use_classes = {"TestClassA": 1, "TestClassB": 3}
    multipliers = {"TestClassA": 1.5, "TestClassB": 2.0}

    with pytest.raises(
        revrtValueError, match="NLCD values must be in list form"
    ):
        compute_land_use_multipliers(
            input_classes, multipliers, land_use_classes, chunks=(1, 3)
        )


def test_dry_costs_build(
    tmp_path, sample_iso_fp, sample_nlcd_fp, sample_slope_fp
):
    """Test building dry costs layer"""

    expected_datasets = [
        "sample_iso",
        "sample_nlcd",
        "sample_slope",
        "dry_multipliers",
        "tie_line_costs_102MW",
        "tie_line_costs_205MW",
        "tie_line_costs_400MW",
        "tie_line_costs_3000MW",
        "tie_line_costs_1500MW",
    ]

    lf = LayeredFile(tmp_path / "test.zarr")
    lf.create_new(sample_nlcd_fp)

    with xr.open_dataset(lf.fp, consolidated=False, engine="zarr") as ds:
        for ds_name in expected_datasets:
            assert ds_name not in ds

    dcc = DryCostsCreator(
        lf, input_layer_dir=tmp_path, output_tiff_dir=tmp_path
    )
    dcc.build(sample_iso_fp, sample_nlcd_fp, sample_slope_fp)

    with xr.open_dataset(lf.fp, consolidated=False, engine="zarr") as ds:
        for ds_name in expected_datasets:
            assert ds_name in ds

        assert "sample_extra_data" not in ds

        expected_multipliers = np.array(
            [
                [10.0, 10.0, 1.0, 1.0, 1.0, 1.0],
                [10.0, 1.2, 1.2, 1.0, 1.0, 1.2],
                [10.0, 2.0, 2.0, 1.0, 3.0, 2.0],
                [1.8, 1.98, 1.1880001, 1.4520001, 1.4520001, 1.8],
                [1.33, 1.33, 1.75, 1.0, 1.47, 1.33],
            ],
        )
        assert np.allclose(ds["dry_multipliers"], expected_multipliers)

        expected_costs = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [30562.17, 3667.46, 3667.46, 3056.22, 3056.22, 3667.46],
                [47348.48, 9469.70, 9469.70, 4734.85, 14204.55, 9469.70],
                [7017.95, 7719.74, 4631.85, 5661.14, 5661.14, 7017.95],
                [3383.59, 3383.59, 4452.10, 2544.06, 3739.76, 3383.59],
            ]
        )
        assert np.allclose(ds["tie_line_costs_102MW"], expected_costs)

        expected_costs = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [36663.5, 4399.62, 4399.62, 3666.35, 3666.35, 4399.62],
                [64750.67, 12950.13, 12950.13, 6475.07, 19425.20, 12950.13],
                [8089.147, 8898.0625, 5338.8374, 6525.246, 6525.246, 8089.147],
                [4067.35, 4067.35, 5351.78, 3058.16, 4495.49, 4067.35],
            ]
        )
        assert np.allclose(ds["tie_line_costs_205MW"], expected_costs)

        expected_costs = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [49075.02, 5889.00, 5889.00, 4907.50, 4907.50, 5889.00],
                [94262.02, 18852.40, 18852.40, 9426.20, 28278.60, 18852.40],
                [9477.603, 10425.363, 6255.2183, 7645.267, 7645.267, 9477.603],
                [6477.737, 6477.737, 8523.338, 4870.4785, 7159.6035, 6477.737],
            ]
        )
        assert np.allclose(ds["tie_line_costs_400MW"], expected_costs)

        expected_costs = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [78474.49, 9416.939, 9416.939, 7847.449, 7847.449, 9416.939],
                [122794.05, 24558.81, 24558.81, 12279.40, 36838.21, 24558.81],
                [12896.70, 14186.37, 8511.83, 10403.34, 10403.34, 12896.70],
                [13868.06, 13868.06, 18247.44, 10427.11, 15327.85, 13868.06],
            ]
        )
        assert np.allclose(ds["tie_line_costs_3000MW"], expected_costs)

        expected_costs = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [94847.88, 11381.75, 11381.75, 9484.79, 9484.79, 11381.75],
                [148414.52, 29682.90, 29682.90, 14841.45, 44524.35, 29682.90],
                [15587.55, 17146.30, 10287.78, 12573.96, 12573.96, 15587.55],
                [16761.57, 16761.57, 22054.69, 12602.68, 18525.94, 16761.57],
            ]
        )
        assert np.allclose(ds["tie_line_costs_1500MW"], expected_costs)

        with rioxarray.open_rasterio(sample_iso_fp, chunks="auto") as da:
            assert np.allclose(ds["sample_iso"], da.isel(band=0))

        with rioxarray.open_rasterio(sample_nlcd_fp, chunks="auto") as da:
            assert np.allclose(ds["sample_nlcd"], da.isel(band=0))

        with rioxarray.open_rasterio(sample_slope_fp, chunks="auto") as da:
            assert np.allclose(ds["sample_slope"], da.isel(band=0))


@pytest.mark.parametrize("lu", [True, False])
@pytest.mark.parametrize("slope", [True, False])
def test_dry_costs_build_extra_inputs(
    tmp_path,
    sample_iso_fp,
    sample_nlcd_fp,
    sample_slope_fp,
    sample_extra_fp,
    lu,
    slope,
    sample_tiff_props,
):
    """Test building dry costs layer"""
    *__, cell_size, ___ = sample_tiff_props
    expected_datasets = [
        "sample_iso",
        "sample_nlcd",
        "sample_slope",
        "dry_multipliers",
        "tie_line_costs_102MW",
        "tie_line_costs_205MW",
        "tie_line_costs_400MW",
        "tie_line_costs_3000MW",
        "tie_line_costs_1500MW",
    ]

    mask = np.array(
        [
            [1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 0],
        ]
    )

    default_mults = {}
    config = {
        "iso_lookup": {"TEST": 3},
        "iso_multipliers": [{"iso": "TEST"}],
        "base_line_costs": {
            "TEST": {"102": 42, "1500": 72, "205": 105, "3000": 77, "400": 21}
        },
    }
    if lu:
        default_mults["land_use"] = {
            "forest": 2,
            "suburban": 3,
            "urban": 4,
            "wetland": 5,
        }
        config["iso_multipliers"][0]["land_use"] = {
            "cropland": 1,
            "forest": 10,
            "suburban": 20,
            "urban": 30,
            "wetland": 40,
        }
    if slope:
        default_mults["slope"] = {
            "hill_mult": 3,
            "hill_slope": 2,
            "mtn_mult": 4,
            "mtn_slope": 7,
        }
        config["iso_multipliers"][0]["slope"] = {
            "hill_mult": 100,
            "hill_slope": 8,
            "mtn_mult": 200,
            "mtn_slope": 9,
        }

    lf = LayeredFile(tmp_path / "test.zarr")
    lf.create_new(sample_nlcd_fp)

    with xr.open_dataset(lf.fp, consolidated=False, engine="zarr") as ds:
        for ds_name in expected_datasets:
            assert ds_name not in ds

    dcc = DryCostsCreator(
        lf, input_layer_dir=tmp_path, output_tiff_dir=tmp_path
    )
    dcc.build(
        sample_iso_fp,
        sample_nlcd_fp,
        sample_slope_fp,
        mask=mask,
        transmission_config=config,
        default_multipliers=default_mults,
        extra_tiffs=[sample_extra_fp],
    )

    with xr.open_dataset(lf.fp, consolidated=False, engine="zarr") as ds:
        for ds_name in expected_datasets:
            assert ds_name in ds

        assert "sample_extra_data" in ds

        expected_multipliers = np.array(
            [
                [10.0, 10.0, 1, 1, 1, 1],
                [10.0, 1, 1, 1, 1, 1],
                [10.0, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
            ],
        )
        if lu:
            expected_multipliers *= np.array(
                [
                    [1, 1, 5, 5, 5, 5],
                    [1, 5, 5, 1, 1, 5],
                    [1, 5, 3, 1, 2, 5],
                    [40, 40, 20, 30, 10, 40],
                    [5, 5, 3, 1, 2, 5],
                ]
            )

        if slope:
            expected_multipliers *= np.array(
                [
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 100, 200, 1],
                    [1, 1, 1, 1, 1, 1],
                ]
            )

        assert np.allclose(ds["dry_multipliers"], expected_multipliers)

        for rating, base_cost in config["base_line_costs"]["TEST"].items():
            expected_costs = (
                np.array([[0] * 6, [0] * 6, [0] * 6, [base_cost] * 6, [0] * 6])
                * mask
                * expected_multipliers
                / METERS_IN_MILE
                * cell_size
            )
            assert np.allclose(
                ds[f"tie_line_costs_{rating}MW"], expected_costs
            )

        with rioxarray.open_rasterio(sample_iso_fp, chunks="auto") as da:
            assert np.allclose(ds["sample_iso"], da.isel(band=0))

        with rioxarray.open_rasterio(sample_nlcd_fp, chunks="auto") as da:
            assert np.allclose(ds["sample_nlcd"], da.isel(band=0))

        with rioxarray.open_rasterio(sample_slope_fp, chunks="auto") as da:
            assert np.allclose(ds["sample_slope"], da.isel(band=0))

        with rioxarray.open_rasterio(sample_extra_fp, chunks="auto") as da:
            assert np.allclose(ds["sample_extra_data"], da.isel(band=0))


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
