# ruff: noqa: N801
"""Custom Exceptions and Errors for revrt"""

import logging


logger = logging.getLogger("revrt")


class revrtError(Exception):
    """Generic revrt Error"""

    def __init__(self, *args, **kwargs):
        """Init exception and broadcast message to logger"""
        super().__init__(*args, **kwargs)
        if args:
            logger.error(str(args[0]), stacklevel=2)


class revrtAttributeError(revrtError, AttributeError):
    """revrt AttributeError"""


class revrtConfigurationError(revrtError, ValueError):
    """revrt ConfigurationError"""


class revrtFileExistsError(revrtError, FileExistsError):
    """revrt FileExistsError"""


class revrtFileNotFoundError(revrtError, FileNotFoundError):
    """revrt FileNotFoundError"""


class revrtKeyError(revrtError, KeyError):
    """revrt KeyError"""


class revrtNotImplementedError(revrtError, NotImplementedError):
    """revrt NotImplementedError"""


class revrtProfileCheckError(revrtError, ValueError):
    """revrt Geotiff Profile Check Error"""


class revrtRuntimeError(revrtError, RuntimeError):
    """revrt RuntimeError"""


class revrtTypeError(revrtError, TypeError):
    """revrt TypeError"""


class revrtValueError(revrtError, ValueError):
    """revrt ValueError"""
