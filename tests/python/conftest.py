"""Fixtures for use across all tests"""

import os
import json
import traceback
from pathlib import Path

import pytest
from click.testing import CliRunner


from revrt._cli import main


LOGGING_META_FILES = {"log.py", "exceptions.py", "warnings.py"}


@pytest.fixture(scope="module")
def repo_dir():
    """Return Path to top-level repo directory"""
    return Path(__file__).parent.parent.parent


@pytest.fixture(scope="module")
def test_dir(repo_dir):
    """Return Path to test directory"""
    return repo_dir / "tests"


@pytest.fixture(scope="module")
def test_data_dir(test_dir):
    """Return Path to test data directory"""
    return test_dir / "data"


@pytest.fixture(scope="module")
def test_utility_data_dir(test_data_dir):
    """Return Path to test data directory"""
    return test_data_dir / "utilities"


@pytest.fixture(scope="module")
def test_routing_data_dir(test_data_dir):
    """Return Path to routing test data directory"""
    return test_data_dir / "routing"


@pytest.fixture(scope="module")
def revx_transmission_layers(test_utility_data_dir):
    """Return Path to test data directory"""
    return test_utility_data_dir / "transmission_layers.zarr"


@pytest.fixture
def assert_message_was_logged(caplog):
    """Assert that a particular (partial) message was logged"""
    caplog.clear()

    def assert_message(msg, log_level=None, clear_records=False):
        """Assert that a message was logged"""
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
def cli_error_message():
    """Return CLI error message for assertion context"""

    def _build_message(result):
        """Return CLI error message for assertion context"""
        if not result.exc_info:
            return ""
        return "".join(traceback.format_exception(*result.exc_info))

    return _build_message


@pytest.fixture(scope="session")
def cli_runner():
    """Cli runner helper utility"""
    return CliRunner()


@pytest.fixture(scope="session")
def run_gaps_cli_with_expected_file(cli_runner, cli_error_message):
    """Run a CLI command and check for expected output file"""

    def _run_cli(cli_command, config, run_dir):
        """Run a CLI command and check for expected output file"""
        out_pattern = cli_command.replace("-", "_")
        config_fp = run_dir / f"test_{out_pattern}_config.json"
        config_fp.write_text(json.dumps(config))

        assert not list(run_dir.glob(f"*_{out_pattern}.*"))
        result = cli_runner.invoke(
            main, [cli_command, "-c", config_fp.as_posix()]
        )
        msg = f"Failed with error {cli_error_message(result)}"
        assert result.exit_code == 0, msg

        out_path = list(run_dir.glob(f"*_{out_pattern}.*"))
        assert len(out_path) == 1
        return out_path[0]

    return _run_cli
