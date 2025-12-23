"""Routing analysis library"""

import importlib.metadata

from ._rust import find_paths, RouteFinder

# Needed for inclusion in Sphinx autodoc documentation generation
find_paths.__module__ = "revrt"
RouteFinder.__module__ = "revrt"

__version__ = version = importlib.metadata.version("NREL-reVRt")
