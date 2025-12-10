"""revrt exception tests"""

from pathlib import Path

import pytest

from revrt.exceptions import (
    revrtError,
    revrtAttributeError,
    revrtConfigurationError,
    revrtFileExistsError,
    revrtFileNotFoundError,
    revrtInvalidStartCostError,
    revrtKeyError,
    revrtLeastCostPathNotFoundError,
    revrtNotImplementedError,
    revrtProfileCheckError,
    revrtRuntimeError,
    revrtTypeError,
    revrtValueError,
)


BASIC_ERROR_MESSAGE = "An error message"


def test_exceptions_log_error(caplog, assert_message_was_logged):
    """Test that a raised exception logs message, if any."""

    try:
        raise revrtError  # noqa: TRY301
    except revrtError:
        pass

    assert not caplog.records

    try:
        raise revrtError(BASIC_ERROR_MESSAGE)  # noqa: TRY301
    except revrtError:
        pass

    assert_message_was_logged(BASIC_ERROR_MESSAGE, "ERROR")


def test_exceptions_log_uncaught_error(assert_message_was_logged):
    """Test that a raised exception logs message if uncaught."""

    with pytest.raises(revrtError):
        raise revrtError(BASIC_ERROR_MESSAGE)

    assert_message_was_logged(BASIC_ERROR_MESSAGE, "ERROR")


@pytest.mark.parametrize(
    "raise_type, catch_types",
    [
        (
            revrtAttributeError,
            [revrtError, revrtAttributeError, AttributeError],
        ),
        (
            revrtConfigurationError,
            [revrtError, revrtConfigurationError, ValueError],
        ),
        (
            revrtFileExistsError,
            [revrtError, revrtFileExistsError, FileExistsError],
        ),
        (
            revrtFileNotFoundError,
            [revrtError, revrtFileNotFoundError, FileNotFoundError],
        ),
        (
            revrtInvalidStartCostError,
            [
                revrtError,
                revrtInvalidStartCostError,
                revrtValueError,
                ValueError,
            ],
        ),
        (
            revrtLeastCostPathNotFoundError,
            [
                revrtError,
                revrtLeastCostPathNotFoundError,
                revrtRuntimeError,
                RuntimeError,
            ],
        ),
        (
            revrtNotImplementedError,
            [revrtError, revrtNotImplementedError, NotImplementedError],
        ),
        (
            revrtProfileCheckError,
            [revrtError, revrtProfileCheckError, ValueError],
        ),
        (revrtKeyError, [revrtError, revrtKeyError, KeyError]),
        (revrtRuntimeError, [revrtError, revrtRuntimeError, RuntimeError]),
        (revrtTypeError, [revrtError, revrtTypeError, TypeError]),
        (revrtValueError, [revrtError, revrtValueError, ValueError]),
    ],
)
def test_catching_error_by_type(raise_type, catch_types, assert_message_was_logged):
    """Test that gaps exceptions are caught correctly."""
    for catch_type in catch_types:
        with pytest.raises(catch_type) as exc_info:
            raise raise_type(BASIC_ERROR_MESSAGE)

        assert BASIC_ERROR_MESSAGE in str(exc_info.value)
        assert_message_was_logged(raise_type.__name__, "ERROR")
        assert_message_was_logged(BASIC_ERROR_MESSAGE, "ERROR")


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
