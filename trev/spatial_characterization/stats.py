"""TreV statistic computation functions"""

from itertools import chain
from functools import partial
from enum import StrEnum, auto

import numpy as np

from trev.exceptions import TreVValueError, TreVNotImplementedError


_PCT_PREFIX = "percentile_"


def _not_implemented(stat_name, *__, **___):
    """Default function for stats which may not have a computation"""
    msg = f"Default computation unavailable for {stat_name!r}"
    raise TreVNotImplementedError(msg)


def _calc_count(processed_raster, **__):
    """Compute number of non-NaN elements"""
    try:
        return processed_raster.count()
    except (AttributeError, TypeError):
        return np.count_nonzero(~np.isnan(processed_raster))


def _calc_min(processed_raster, **__):
    """Compute minimum value in raster"""
    return float(np.nanmin(processed_raster))


def _calc_max(processed_raster, **__):
    """Compute maximum value in raster"""
    return float(np.nanmax(processed_raster))


def _calc_mean(processed_raster, out_dtype, **__):
    """Compute mean value of raster"""
    return float(np.nanmean(processed_raster, dtype=out_dtype))


def _calc_sum(processed_raster, out_dtype, **__):
    """Compute sum of raster"""
    return float(np.nansum(processed_raster, dtype=out_dtype))


def _calc_std(processed_raster, **__):
    """Compute std of raster"""
    return float(np.nanstd(processed_raster))


def _calc_median(processed_raster, **__):
    """Compute median of raster"""
    return float(np.nanmedian(processed_raster))


def _calc_majority(pixel_count, **__):
    """Compute value that makes up majority of raster"""
    return max(pixel_count, key=pixel_count.get)


def _calc_minority(pixel_count, **__):
    """Compute value that makes up minority of raster"""
    return min(pixel_count, key=pixel_count.get)


def _calc_unique(pixel_count, **__):
    """Compute number of unique values in raster"""
    return len(pixel_count)


def _calc_range(processed_raster, feature_stats, **__):
    """Get the range from stats using min/max, compute if not present"""
    try:
        r_min = feature_stats["min"]
    except KeyError:
        r_min = float(np.nanmin(processed_raster))
    try:
        r_max = feature_stats["max"]
    except KeyError:
        r_max = float(np.nanmax(processed_raster))
    return r_max - r_min


def _calc_nodata(masked_raster, nodata, **__):
    """Compute number of nodata pixels in raster"""
    try:
        return float(
            np.sum(
                (masked_raster == nodata)
                | (masked_raster == masked_raster.attrs.get("nodata"))
            )
        )
    except AttributeError:
        return float(np.sum(masked_raster == nodata))


class Stat(StrEnum):
    """Enum of basic computable statistics"""

    COUNT = auto()
    """Compute number of non-NaN elements"""

    MIN = auto()
    """Compute minimum value in zone"""

    MAX = auto()
    """Compute maximum value in zone"""

    MEAN = auto()
    """Compute mean value of zone"""

    SUM = auto()
    """Compute sum of zone"""

    STD = auto()
    """Compute std of zone"""

    MEDIAN = auto()
    """Compute median of zone"""

    MAJORITY = auto()
    """Compute pixel value that makes up majority of zone"""

    MINORITY = auto()
    """Compute pixel value that makes up minority of zone"""

    UNIQUE = auto()
    """Compute number of unique pixel values in zone"""

    RANGE = auto()
    """Compute range of pixel values in zone"""

    NODATA = auto()
    """Compute number of nodata pixels in zone"""

    PIXEL_COUNT = auto()
    """Compute count for each pixel value in zone"""

    def __new__(cls, value):  # noqa: PLR0912, C901
        """Create new enum member"""

        obj = str.__new__(cls, value)
        obj._value_ = value
        match value:
            case "count":
                obj.compute = _calc_count
                obj.requires_pixel_count = True
            case "min":
                obj.compute = _calc_min
                obj.requires_pixel_count = False
            case "max":
                obj.compute = _calc_max
                obj.requires_pixel_count = False
            case "mean":
                obj.compute = _calc_mean
                obj.requires_pixel_count = False
            case "sum":
                obj.compute = _calc_sum
                obj.requires_pixel_count = False
            case "std":
                obj.compute = _calc_std
                obj.requires_pixel_count = False
            case "median":
                obj.compute = _calc_median
                obj.requires_pixel_count = False
            case "majority":
                obj.compute = _calc_majority
                obj.requires_pixel_count = True
            case "minority":
                obj.compute = _calc_minority
                obj.requires_pixel_count = True
            case "unique":
                obj.compute = _calc_unique
                obj.requires_pixel_count = True
            case "range":
                obj.compute = _calc_range
                obj.requires_pixel_count = False
            case "nodata":
                obj.compute = _calc_nodata
                obj.requires_pixel_count = False
            case "pixel_count":
                obj.compute = partial(_not_implemented, value)
                obj.requires_pixel_count = True
            case _:  # pragma: no cover
                obj.compute = partial(_not_implemented, value)
                obj.requires_pixel_count = False

        return obj


class FractionalStat(StrEnum):
    """Enum of fractional pixel statistics"""

    FRACTIONAL_PIXEL_COUNT = auto()
    """Compute fractional pixel count for each pixel value in zone"""

    FRACTIONAL_AREA = auto()
    """Compute fractional area for each pixel value in zone"""

    VALUE_MULTIPLIED_BY_FRACTIONAL_AREA = auto()
    """Compute fractional pixel * area for each pixel value in zone"""


class ComputableStats:
    """A class to represent computable statistics for zonal stats"""

    def __init__(
        self, base_stats=None, percentiles=None, fractional_stats=None
    ):
        self.base_stats = base_stats or []
        self.percentiles = percentiles or {}
        self.fractional_stats = fractional_stats or []
        self.all = {
            *self.base_stats,
            *self.percentiles,
            *self.fractional_stats,
        }

    def __contains__(self, key):
        return key in self.all

    @property
    def empty(self):
        """dict: Dict with empty stat values"""
        out = {
            k: (0 if k in {Stat.COUNT, Stat.PIXEL_COUNT} else None)
            for k in chain(self.base_stats, self.fractional_stats)
        }
        for pct in self.percentiles:
            out[pct] = None
        return out

    def lazy_pixel_count(self, processed_raster):
        """Compute pixel counts from a raster

        The output dictionary will be empty if this stats collection
        does not contain any stats that require a pixel count.

        Parameters
        ----------
        processed_raster : array-like
            List or array of data to compute pixel counts from. This
            collection can contain NaN values, which will be ignored in
            the final output.

        Returns
        -------
        dict
            Dictionary where keys are the unique array entries and the
            values are the counts for each entry. This dictionary will
            be empty if this stats collection does not contain any stats
            that require a pixel count.
        """
        if not any(stat.requires_pixel_count for stat in self.base_stats):
            return {}

        keys, counts = np.unique(processed_raster, return_counts=True)
        return {
            k.item(): c.item()
            for k, c in zip(keys, counts, strict=False)
            if not np.isnan(k)
        }

    @classmethod
    def from_iter(cls, stats=None):
        """Create a ComputableStats object from an iterable of stats

        Parameters
        ----------
        stats : str | iterable of str, optional
            Names of all statistics to compute. Statistics must be one
            of the members of :class`Stat` or :class:`FractionalStat`,
            or must start with the "percentile_" prefix and ent with an
            int or float representing the percentile to compute (e.g.
            ``percentile_10.5``). If only one statistic is to be
            computed, you can provide it directly as a string.
            Otherwise, provide a list of statistic names or a string
            with the names separated by a space. You can also provide
            the string "ALL" or "*" to specify that all statistics
            should be computed. If no input, empty input, or ``None`` is
            provided, then only the base stats ("count", "min", "max",
            "mean") are configured. To summarize, all of the following
            are valid inputs:

                - ``stats="*"`` or ``stats="ALL"`` or ``stats="All"``
                - ``stats="min"``
                - ``stats="min max"``
                - ``stats=["min"]``
                - ``stats=["min", "max", "percentile_10.5"]``

            By default, ``None``.

        Returns
        -------
        ComputableStats
            An initialized :class:`ComputableStats` object.

        Raises
        ------
        TreVValueError
            If one or more input stats are not known.
        """
        if not stats:
            return cls(base_stats=[Stat.COUNT, Stat.MIN, Stat.MAX, Stat.MEAN])

        if isinstance(stats, str):
            if stats.casefold() in {"*", "all"}:
                return cls(
                    base_stats=list(Stat),
                    fractional_stats=list(FractionalStat),
                )

            stats = stats.split()

        allowed_base_stats = {*Stat}
        allowed_fractional_stats = {*FractionalStat}
        base_stats = []
        percentiles = {}
        fractional_stats = []
        for stat in stats:
            stat_name = stat.casefold()
            if stat_name.startswith(_PCT_PREFIX):
                percentiles[stat_name] = _get_percentile(stat_name)
            elif stat_name in allowed_fractional_stats:
                fractional_stats.append(stat_name)
            elif stat_name in allowed_base_stats:
                base_stats.append(Stat(stat_name))
            else:
                valid_stats = {str(s) for s in Stat}
                valid_stats |= {str(s) for s in FractionalStat}
                msg = (
                    f"Stat {stat!r} not valid; must be one of:\n"
                    f"{valid_stats!r}"
                )
                raise TreVValueError(msg)

        return cls(base_stats, percentiles, fractional_stats)


def _get_percentile(stat):
    """Get the percentile value from user string input"""
    q = float(stat.replace(_PCT_PREFIX, ""))
    if not (0 <= q <= 100):  # noqa: PLR2004
        msg = f"Percentiles must be between 0 and 100 (inclusive). Got: {q}"
        raise TreVValueError(msg)
    return q
