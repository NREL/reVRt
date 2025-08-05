"""Custom Exceptions and Errors for revrt"""

import logging


logger = logging.getLogger("revrt")


class revrtError(Exception):  # noqa: N801
    """Generic revrt Error"""

    def __init__(self, *args, **kwargs):
        """Init exception and broadcast message to logger"""
        super().__init__(*args, **kwargs)
        if args:
            logger.error(str(args[0]), stacklevel=2)


class revrtFileNotFoundError(revrtError, FileNotFoundError):  # noqa: N801
    """revrt FileNotFoundError"""


class revrtKeyError(revrtError, KeyError):  # noqa: N801
    """revrt KeyError"""


class revrtNotImplementedError(revrtError, NotImplementedError):  # noqa: N801
    """revrt NotImplementedError"""


class revrtRuntimeError(revrtError, RuntimeError):  # noqa: N801
    """revrt RuntimeError"""


class revrtTypeError(revrtError, TypeError):  # noqa: N801
    """revrt TypeError"""


class revrtValueError(revrtError, ValueError):  # noqa: N801
    """revrt ValueError"""
