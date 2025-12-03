"""Routing analysis library for the reV model"""

import importlib.metadata

from ._rust import find_paths

find_paths.__module__ = "revrt"
__version__ = version = importlib.metadata.version("NREL-reVRt")
