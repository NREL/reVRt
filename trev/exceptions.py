"""Custom Exceptions and Errors for TreV"""

import logging


logger = logging.getLogger("trev")


class TreVError(Exception):
    """Generic TreV Error"""

    def __init__(self, *args, **kwargs):
        """Init exception and broadcast message to logger"""
        super().__init__(*args, **kwargs)
        if args:
            logger.error(str(args[0]), stacklevel=2)


class TreVKeyError(TreVError, KeyError):
    """TreV KeyError"""


class TreVNotImplementedError(TreVError, NotImplementedError):
    """TreV NotImplementedError"""


class TreVRuntimeError(TreVError, RuntimeError):
    """TreV RuntimeError"""


class TreVTypeError(TreVError, TypeError):
    """TreV TypeError"""


class TreVValueError(TreVError, ValueError):
    """TreV ValueError"""
