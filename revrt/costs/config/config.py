"""Default configuration for tie-line cost determination"""

import logging
import numpy as np
from pathlib import Path
from contextlib import suppress
from collections import UserDict

from gaps.config import load_config

from revrt.exceptions import revrtValueError


CONFIG_DIR = Path(__file__).resolve().parent
CONFIG = {
    "base_line_costs": CONFIG_DIR / "base_line_costs.json",
    "iso_lookup": CONFIG_DIR / "iso_lookup.json",
    "iso_multipliers": CONFIG_DIR / "iso_multipliers.json",
    "land_use_classes": CONFIG_DIR / "land_use_classes.json",
    "new_substation_costs": CONFIG_DIR / "new_substation_costs.json",
    "power_classes": CONFIG_DIR / "power_classes.json",
    "power_to_voltage": CONFIG_DIR / "power_to_voltage.json",
    "transformer_costs": CONFIG_DIR / "transformer_costs.json",
    "upgrade_substation_costs": CONFIG_DIR / "upgrade_substation_costs.json",
}
logger = logging.getLogger(__name__)


class TransmissionConfig(UserDict):
    """Load default transmission cost configuration as dictionary

    This configuration dict has the following keys:

        - base_line_costs
        - iso_lookup
        - iso_multipliers
        - land_use_classes
        - new_substation_costs
        - power_classes
        - power_to_voltage
        - transformer_costs
        - upgrade_substation_costs
        - reverse_iso (dynamically computed)
        - voltage_to_power (dynamically computed)
        - line_power_to_classes (dynamically computed)

    """

    def __init__(self, config=None):
        """

        Parameters
        ----------
        config : str | dict, optional
            Dictionary of transmission cost configuration values, or
            path to JSON/JSON5 file containing this dictionary. The
            dictionary should have the following keys:

                - base_line_costs
                - iso_lookup
                - iso_multipliers
                - land_use_classes
                - new_substation_costs
                - power_classes
                - power_to_voltage
                - transformer_costs
                - upgrade_substation_costs

            Each of these keys should point to another dictionary or
            path to JSON/JSON5 file containing a dictionary of
            configurations for each section. For the expected contents
            of each dictionary, see the default config. If ``None``,
            values from the default config are used.
            By default, ``None``.
        """
        super().__init__()

        self._load_config(CONFIG)
        self._load_config(config)

    def __getitem__(self, k):
        if k == "reverse_iso":
            out = {v: k for k, v in self["iso_lookup"].items()}
        elif k == "voltage_to_power":
            out = {v: k for k, v in self["power_to_voltage"].items()}
        elif k == "line_power_to_classes":
            out = {v: k for k, v in self["power_classes"].items()}
        else:
            out = super().__getitem__(k)

        return out

    def _load_config(self, config):
        """Load config from user input"""
        if config is None:
            return

        config = _try_load_as_config(config)

        if not isinstance(config, dict):
            msg = (
                "Transmission config must be a path to a json file or a "
                f"dictionary, not: {config}"
            )
            raise revrtValueError(msg)

        for key, config_value in config.items():
            self[key] = _try_load_as_config(config_value)

    def capacity_to_kv(self, capacity):
        """Convert capacity class to line voltage

        Parameters
        ----------
        capacity : int
            Capacity class in MW.

        Returns
        -------
        kV : int
            Tie-line voltage in kV.
        """
        cap_class = parse_cap_class(capacity)
        line_capacity = self["power_classes"][cap_class]
        kv = self["power_to_voltage"][str(line_capacity)]

        return int(kv)

    def kv_to_capacity(self, kv):
        """Convert line voltage to capacity class

        Parameters
        ----------
        kv : int
            Tie-line voltage in kV.

        Returns
        -------
        capacity : int
            Capacity class in MW.
        """
        line_capacity = self["voltage_to_power"][kv]
        capacity = self["line_power_to_classes"][int(line_capacity)]

        return int(capacity.strip("MW"))

    def sub_upgrade_cost(self, region, tie_line_voltage):
        """Extract substation upgrade costs

        Costs are given in $ based on region and tie-line voltage
        rating.

        Parameters
        ----------
        region : int
            Region code used to extract ISO.
        tie_line_voltage : int | str
            Tie-line voltage class in kV.

        Returns
        -------
        int
            Substation upgrade cost.
        """
        if not isinstance(tie_line_voltage, str):
            tie_line_voltage = str(tie_line_voltage)

        region = self["reverse_iso"][region]

        return self["upgrade_substation_costs"][region][tie_line_voltage]

    def new_sub_cost(self, region, tie_line_voltage):
        """Extract new substation costs

        Costs are given in $ based on region and tie-line voltage
        rating.

        Parameters
        ----------
        region : int
            Region code used to extract ISO.
        tie_line_voltage : int | str
            Tie-line voltage class in kV.

        Returns
        -------
        int
            New substation cost.
        """
        if not isinstance(tie_line_voltage, str):
            tie_line_voltage = str(tie_line_voltage)

        region = self["reverse_iso"][region]

        return self["new_substation_costs"][region][tie_line_voltage]

    def transformer_cost(self, feature_voltage, tie_line_voltage):
        """Extract transformer costs

        Costs are given in $ based on region and tie-line voltage
        rating.

        Parameters
        ----------
        feature_voltage : int
            Voltage of feature that tie-line is connecting to.
        tie_line_voltage : int | str
            Tie-line voltage class in kV.

        Returns
        -------
        int
            Transformer cost as $/MW.
        """
        if not isinstance(tie_line_voltage, str):
            tie_line_voltage = str(tie_line_voltage)

        costs = self["transformer_costs"][tie_line_voltage]

        classes = np.array(sorted(map(int, costs)))
        valid_idx = np.where(classes >= feature_voltage)[0]
        v_class = classes[valid_idx[0]] if valid_idx.size else classes[-1]

        return costs[str(v_class)]


def parse_config(config=None):
    """Load TransmissionConfig config if needed

    Parameters
    ----------
    config : str | dict | TransmissionConfig, optional
        Path to transmission config JSON, dictionary of transmission
        config JSONs, or preloaded TransmissionConfig objects.
        By default, ``None``.

    Returns
    -------
    TransmissionConfig
        Transmission config instance.
    """
    if not isinstance(config, TransmissionConfig):
        config = TransmissionConfig(config=config)

    logger.debug("Transmission config:\n%s", config)
    return config


def parse_cap_class(capacity):
    """Parse capacity class from input capacity

    Parameters
    ----------
    capacity : int | float | str
        Capacity to convert to "capacity class".

    Returns
    -------
    cap_class : str
        Capacity class in format "{capacity}MW".
    """
    if not isinstance(capacity, str):
        return f"{int(capacity)}MW"

    if not capacity.endswith("MW"):
        return f"{capacity}MW"

    return capacity


def _try_load_as_config(possibly_config_fp):
    """Try to load input as config, otherwise return input"""
    with suppress(TypeError):
        if Path(possibly_config_fp).suffix in {".json", ".json5"}:
            return load_config(possibly_config_fp)

    return possibly_config_fp
