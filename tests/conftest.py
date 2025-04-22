"""Fixtures for use across all tests"""

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def repo_dir():
    """Return Path to top-level repo directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="module")
def test_dir(repo_dir):
    """Return Path to test directory."""
    return repo_dir / "tests"


@pytest.fixture(scope="module")
def test_data_dir(test_dir):
    """Return Path to test data directory."""
    return test_dir / "data"
