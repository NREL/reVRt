"""Test dry cost layer creation"""

from pathlib import Path

import pytest
import numpy as np

from revrt.costs.dry_costs_creator import (
    DEFAULT_HILL_MULTIPLIER,
    DEFAULT_MTN_MULTIPLIER,
    DEFAULT_HILL_SLOPE,
    DEFAULT_MTN_SLOPE,
    compute_slope_multipliers,
    compute_land_use_multipliers,
)
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

    slope_multipliers = compute_slope_multipliers(input_slopes)
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

    slope_multipliers = compute_slope_multipliers(input_slopes, config)
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
        input_classes, multipliers, land_use_classes
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
            input_classes, multipliers, land_use_classes
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
            input_classes, multipliers, land_use_classes
        )


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
