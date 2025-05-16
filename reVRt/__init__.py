"""Routing analysis library for the reV model"""

import importlib.metadata

from ._rust import find_paths  # noqa: F401 type: ignore


__version__ = version = importlib.metadata.version("reVRt")
