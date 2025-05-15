"""Custom Exceptions and Errors for reVRt"""

import logging


logger = logging.getLogger("reVRt")


class reVRtError(Exception):  # noqa: N801
    """Generic reVRt Error"""

    def __init__(self, *args, **kwargs):
        """Init exception and broadcast message to logger"""
        super().__init__(*args, **kwargs)
        if args:
            logger.error(str(args[0]), stacklevel=2)


class reVRtKeyError(reVRtError, KeyError):  # noqa: N801
    """reVRt KeyError"""


class reVRtNotImplementedError(reVRtError, NotImplementedError):  # noqa: N801
    """reVRt NotImplementedError"""


class reVRtRuntimeError(reVRtError, RuntimeError):  # noqa: N801
    """reVRt RuntimeError"""


class reVRtTypeError(reVRtError, TypeError):  # noqa: N801
    """reVRt TypeError"""


class reVRtValueError(reVRtError, ValueError):  # noqa: N801
    """reVRt ValueError"""
