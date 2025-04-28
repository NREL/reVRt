"""Test statistic computation functions for spatial characterization"""

from pathlib import Path

import pytest
import numpy as np
import xarray as xr

from trev.spatial_characterization.stats import (
    Stat,
    FractionalStat,
    ComputableStats,
    _PCT_PREFIX,  # noqa: PLC2701
)
from trev.exceptions import TreVNotImplementedError, TreVValueError


def test_ni_error_message():
    """Test error message for not implemented stat"""

    with pytest.raises(TreVNotImplementedError) as exc_info:
        Stat.PIXEL_COUNT.compute(1, 2, 3, "a", kw="another")

    assert (
        str(exc_info.value)
        == "Default computation unavailable for 'pixel_count'"
    )


@pytest.mark.parametrize(
    "in_obj",
    [[np.nan, 2, 3], np.array([4, np.nan, 6]), xr.DataArray([7, 8, np.nan])],
)
def test_stat_count(in_obj):
    """Test the count stat"""

    assert Stat.COUNT.compute(in_obj) == 2


@pytest.mark.parametrize(
    "in_obj",
    [[np.nan, 2, 3], np.array([2, np.nan, 3]), xr.DataArray([2, 3, np.nan])],
)
def test_stat_min(in_obj):
    """Test the min stat"""

    assert Stat.MIN.compute(in_obj) == 2.0


@pytest.mark.parametrize(
    "in_obj",
    [[np.nan, 2, 3], np.array([2, np.nan, 3]), xr.DataArray([2, 3, np.nan])],
)
def test_stat_max(in_obj):
    """Test the max stat"""

    assert Stat.MAX.compute(in_obj) == 3.0


@pytest.mark.parametrize(
    "in_obj",
    [[np.nan, 2, 3], np.array([2, np.nan, 3]), xr.DataArray([2, 3, np.nan])],
)
def test_stat_mean(in_obj):
    """Test the mean stat"""

    assert Stat.MEAN.compute(in_obj, out_dtype="float32") == 2.5


@pytest.mark.parametrize(
    "in_obj",
    [[np.nan, 2, 3], np.array([2, np.nan, 3]), xr.DataArray([2, 3, np.nan])],
)
def test_stat_sum(in_obj):
    """Test the sum stat"""

    assert Stat.SUM.compute(in_obj, out_dtype="float32") == 5


@pytest.mark.parametrize(
    "in_obj",
    [[np.nan, 2, 3], np.array([2, np.nan, 3]), xr.DataArray([2, 3, np.nan])],
)
def test_stat_std(in_obj):
    """Test the std stat"""

    assert Stat.STD.compute(in_obj) == 0.5


@pytest.mark.parametrize(
    "in_obj",
    [[np.nan, 2, 3], np.array([2, np.nan, 3]), xr.DataArray([2, 3, np.nan])],
)
def test_stat_median(in_obj):
    """Test the median stat"""

    assert Stat.MEDIAN.compute(in_obj) == 2.5


@pytest.mark.parametrize(
    "in_obj",
    [{"A": 0, "B": 5, "C": 3, "D": 2}, {"A": 0, "B": 5, "C": 3, "D": 5}],
)
def test_stat_majority(in_obj):
    """Test the majority stat"""

    assert Stat.MAJORITY.compute(in_obj) == "B"


@pytest.mark.parametrize(
    "in_obj",
    [{"A": 0, "B": 5, "C": 3, "D": 2}, {"A": 0, "B": 5, "C": 3, "D": 0}],
)
def test_stat_minority(in_obj):
    """Test the minority stat"""

    assert Stat.MINORITY.compute(in_obj) == "A"


def test_stat_unique():
    """Test the unique stat"""

    assert Stat.UNIQUE.compute({}) == 0
    assert Stat.UNIQUE.compute({"A": 0}) == 1
    assert Stat.UNIQUE.compute({"A": 0, "B": 0, "C": 0}) == 3


def test_range_bounds_exist():
    """Test the range stat if min and max are given"""

    feature_stats = {Stat.MIN: 0, Stat.MAX: 10}
    processed_raster = [10, 30]

    assert Stat.RANGE.compute(processed_raster, feature_stats) == 10


def test_range_bounds_do_not_exist():
    """Test the range stat if min and max are not given"""

    feature_stats = {}
    processed_raster = [10, 30, np.nan]

    assert Stat.RANGE.compute(processed_raster, feature_stats) == 20


@pytest.mark.parametrize(
    "in_obj",
    [np.array([2, np.nan, 3, 3]), xr.DataArray([2, 3, np.nan, 3])],
)
def test_stat_nodata(in_obj):
    """Test the nodata stat"""

    assert Stat.NODATA.compute(in_obj, nodata=3) == 2


def test_stat_nodata_with_rater_nodata():
    """Test the nodata stat when array has it's own nodata"""

    da = xr.DataArray([2, 3, np.nan, 3, 4, 2])
    da.attrs["nodata"] = 4
    assert Stat.NODATA.compute(da, nodata=3) == 3


def test_empty_computable_stats():
    """Test creating an empty `ComputableStats` object"""
    cs = ComputableStats()
    assert not cs.base_stats
    assert not cs.percentiles
    assert not cs.fractional_stats
    assert not cs.all
    assert not cs.empty
    assert not cs.lazy_pixel_count([1, 1, 3, 4, 1, 4])


def test_basic_computable_stats():
    """Test creating a basic `ComputableStats` object"""
    cs = ComputableStats.from_iter()
    assert cs.base_stats
    assert not cs.percentiles
    assert not cs.fractional_stats
    assert cs.all == set(cs.base_stats)
    assert cs.empty == {
        Stat.COUNT: 0,
        Stat.MIN: None,
        Stat.MAX: None,
        Stat.MEAN: None,
    }
    assert not cs.lazy_pixel_count([])
    assert cs.lazy_pixel_count([1, 1, 3, 4, 1, 4]) == {1: 3, 3: 1, 4: 2}


@pytest.mark.parametrize("in_str", ["*", "ALL", "All", "all"])
def test_all_computable_stats(in_str):
    """Test creating a full `ComputableStats` object"""
    cs = ComputableStats.from_iter(in_str)
    assert cs.base_stats
    assert not cs.percentiles
    assert cs.fractional_stats
    assert len(cs.all) == len(cs.base_stats) + len(cs.percentiles) + len(
        cs.fractional_stats
    )
    empty_dict = cs.empty
    assert empty_dict[Stat.COUNT] == 0
    assert empty_dict[Stat.PIXEL_COUNT] == 0
    assert all(
        v is None
        for stat, v in empty_dict.items()
        if stat not in {Stat.COUNT, Stat.PIXEL_COUNT}
    )
    assert not cs.lazy_pixel_count([])
    assert cs.lazy_pixel_count([1, 1, 3, 4, 1, 4]) == {1: 3, 3: 1, 4: 2}


@pytest.mark.parametrize(
    "in_iter",
    [
        f"{Stat.MIN} {_PCT_PREFIX}10.5 {FractionalStat.FRACTIONAL_AREA}",
        [Stat.MIN, f"{_PCT_PREFIX}10.5", FractionalStat.FRACTIONAL_AREA],
    ],
)
def test_computable_stats_from_other_iterable(in_iter):
    """Test creating a `ComputableStats` object for an iter of stats"""
    cs = ComputableStats.from_iter(in_iter)
    assert cs.base_stats == [Stat.MIN]
    assert cs.percentiles == {f"{_PCT_PREFIX}10.5": 10.5}
    assert cs.fractional_stats == [FractionalStat.FRACTIONAL_AREA]
    assert len(cs.all) == 3

    assert cs.empty == {
        Stat.MIN: None,
        f"{_PCT_PREFIX}10.5": None,
        f"{FractionalStat.FRACTIONAL_AREA}": None,
    }

    assert Stat.MIN in cs
    assert f"{_PCT_PREFIX}10.5" in cs
    assert f"{FractionalStat.FRACTIONAL_AREA}" in cs


def test_computable_stats_unknown_stat():
    """Test that an unknown stat raises error"""
    with pytest.raises(TreVValueError) as exc_info:
        ComputableStats.from_iter("DNE")

    exc_info = str(exc_info.value)
    assert "Stat 'DNE' not valid; must be one of:" in exc_info
    assert "{" in exc_info
    assert "}" in exc_info


def test_computable_stats_bad_percentile():
    """Test that a bad percentile raises error"""
    with pytest.raises(TreVValueError) as exc_info:
        ComputableStats.from_iter(f"{_PCT_PREFIX}100.5")

    assert (
        str(exc_info.value)
        == "Percentiles must be between 0 and 100 (inclusive). Got: 100.5"
    )

    with pytest.raises(TreVValueError) as exc_info:
        ComputableStats.from_iter(f"{_PCT_PREFIX}-10")

    assert (
        str(exc_info.value)
        == "Percentiles must be between 0 and 100 (inclusive). Got: -10.0"
    )


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
