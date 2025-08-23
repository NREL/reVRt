"""Fixtures for use across utilities tests"""

import pytest


@pytest.fixture(scope="module")
def test_utility_data_dir(test_data_dir):
    """Return Path to test data directory."""
    return test_data_dir / "utilities"
