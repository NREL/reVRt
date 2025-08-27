"""Fixtures for use across all tests"""

import os
from pathlib import Path

import pytest
from click.testing import CliRunner


LOGGING_META_FILES = {"log.py", "exceptions.py", "warnings.py"}


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


@pytest.fixture(scope="module")
def test_utility_data_dir(test_data_dir):
    """Return Path to test data directory."""
    return test_data_dir / "utilities"


@pytest.fixture
def assert_message_was_logged(caplog):
    """Assert that a particular (partial) message was logged."""
    caplog.clear()

    def assert_message(msg, log_level=None, clear_records=False):
        """Assert that a message was logged."""
        assert caplog.records

        for record in caplog.records:
            if msg in record.message:
                break
        else:
            msg = f"{msg!r} not found in log records"
            raise AssertionError(msg)

        # record guaranteed to be defined b/c of "assert caplog.records"
        if log_level:
            assert record.levelname == log_level  # cspell:disable-line
        assert record.filename not in LOGGING_META_FILES
        assert record.funcName != "__init__"
        assert "revrt" in record.name

        if clear_records:
            caplog.clear()

    return assert_message


@pytest.fixture
def tmp_cwd(tmp_path):
    """Change working dir to temporary dir"""
    original_directory = Path.cwd()
    try:
        os.chdir(tmp_path)
        yield tmp_path
    finally:
        os.chdir(original_directory)


@pytest.fixture(scope="session")
def cli_runner():
    """Cli runner helper utility"""
    return CliRunner()
