"""Routing analysis library"""

import importlib.metadata

from ._rust import find_paths


__version__ = version = importlib.metadata.version("NREL-reVRt")
