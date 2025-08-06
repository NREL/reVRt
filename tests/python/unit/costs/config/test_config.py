"""reVRt transmission config tests"""

import json
from pathlib import Path

import pytest
from gaps.config import load_config

from revrt.costs.config.config import CONFIG, TransmissionConfig, parse_config


@pytest.mark.parametrize("use_pc", [True, False])
def test_default_config(use_pc):
    """Test that defaults for `TransmissionConfig` are loaded properly"""
    tc = parse_config() if use_pc else TransmissionConfig()

    assert list(tc) == list(CONFIG)
    for key, fp in CONFIG.items():
        assert tc[key] == load_config(fp)


def test_parse_config_on_tc_instance():
    """Test calling parse_config with a `TransmissionConfig` instance"""
    tc = parse_config(TransmissionConfig())

    assert list(tc) == list(CONFIG)
    for key, fp in CONFIG.items():
        assert tc[key] == load_config(fp)


@pytest.mark.parametrize("swap_key", CONFIG)
@pytest.mark.parametrize("as_file", [True, False])
@pytest.mark.parametrize("use_pc", [True, False])
def test_custom_config_one_part(tmp_path, swap_key, as_file, use_pc):
    """Test that user can supply pieces of config"""
    custom_vals = {"test_region": {"102": 99}}

    if as_file:
        fp = tmp_path / "test.json"
        with fp.open("w", encoding="utf-8") as fh:
            json.dump(custom_vals, fh)
        tc = (
            parse_config({swap_key: str(fp)})
            if use_pc
            else TransmissionConfig({swap_key: str(fp)})
        )
    else:
        tc = (
            parse_config({swap_key: custom_vals})
            if use_pc
            else TransmissionConfig({swap_key: custom_vals})
        )

    assert list(tc) == list(CONFIG)
    for key, fp in CONFIG.items():
        if key == swap_key:
            assert tc[key] != load_config(fp)
            assert tc[key] == custom_vals
        else:
            assert tc[key] == load_config(fp)


@pytest.mark.parametrize("use_pc", [True, False])
def test_custom_config_one_part_as_path(tmp_path, use_pc):
    """Test that user can supply a piece of config as pathlib.Path"""

    swap_key = next(iter(CONFIG))
    custom_vals = {"test_region": {"102": 99}}

    fp = tmp_path / "test.json"
    with fp.open("w", encoding="utf-8") as fh:
        json.dump(custom_vals, fh)

    tc = (
        parse_config({swap_key: fp})
        if use_pc
        else TransmissionConfig({swap_key: fp})
    )

    assert list(tc) == list(CONFIG)
    for key, fp in CONFIG.items():
        if key == swap_key:
            assert tc[key] != load_config(fp)
            assert tc[key] == custom_vals
        else:
            assert tc[key] == load_config(fp)


@pytest.mark.parametrize("use_pc", [True, False])
def test_custom_config_relative_path(tmp_path, use_pc):
    """Test that user can supply relative paths in config"""

    config_keys = iter(CONFIG)
    swap_key_1 = next(config_keys)
    custom_vals_1 = {"test_region": {"102": 99}}

    fp = tmp_path / "test.json"
    with fp.open("w", encoding="utf-8") as fh:
        json.dump(custom_vals_1, fh)

    sub_dir = tmp_path / "test"
    sub_dir.mkdir()

    swap_key_2 = next(config_keys)
    custom_vals_2 = {"another_ley": 500}

    fp = sub_dir / "same_dir_test.json"
    with fp.open("w", encoding="utf-8") as fh:
        json.dump(custom_vals_2, fh)

    sub_sub_dir = sub_dir / "test_sub"
    sub_sub_dir.mkdir()

    swap_key_3 = next(config_keys)
    custom_vals_3 = {"final_key": ["1000", "10"]}

    fp = sub_sub_dir / "sub_dir_test.json"
    with fp.open("w", encoding="utf-8") as fh:
        json.dump(custom_vals_3, fh)

    swap_key_4 = next(config_keys)
    custom_vals_4 = {"test": {"a": 1, "b": 100}}
    tc_config = {
        swap_key_1: "../test.json",
        swap_key_2: "./same_dir_test.json",
        swap_key_3: "./test_sub/sub_dir_test.json",
        swap_key_4: custom_vals_4,
    }
    tc_config_fp = sub_dir / "tc_config.json"
    with tc_config_fp.open("w", encoding="utf-8") as fh:
        json.dump(tc_config, fh)

    tc = (
        parse_config(tc_config_fp)
        if use_pc
        else TransmissionConfig(tc_config_fp)
    )

    assert list(tc) == list(CONFIG)
    for key, fp in CONFIG.items():
        if key == swap_key_1:
            assert tc[key] != load_config(fp)
            assert tc[key] == custom_vals_1
        elif key == swap_key_2:
            assert tc[key] != load_config(fp)
            assert tc[key] == custom_vals_2
        elif key == swap_key_3:
            assert tc[key] != load_config(fp)
            assert tc[key] == custom_vals_3
        elif key == swap_key_4:
            assert tc[key] != load_config(fp)
            assert tc[key] == custom_vals_4
        else:
            assert tc[key] == load_config(fp)


def test_reverse_iso():
    """Test the special key `reverse_iso`"""

    tc = TransmissionConfig({"iso_lookup": {"iso_1": 5, "iso2": 42}})

    assert tc["iso_lookup"] == {"iso_1": 5, "iso2": 42}
    assert tc["reverse_iso"] == {5: "iso_1", 42: "iso2"}


def test_voltage_to_power():
    """Test the special key `voltage_to_power`"""

    tc = TransmissionConfig({"power_to_voltage": {"102": 100, "750": 345}})

    assert tc["power_to_voltage"] == {"102": 100, "750": 345}
    assert tc["voltage_to_power"] == {100: "102", 345: "750"}


def test_line_power_to_classes():
    """Test the special key `line_power_to_classes`"""

    tc = TransmissionConfig({"power_classes": {"100MW": 102, "750MW": 345}})

    assert tc["power_classes"] == {"100MW": 102, "750MW": 345}
    assert tc["line_power_to_classes"] == {102: "100MW", 345: "750MW"}


def test_kv_capacity_conversion():
    """Test that converting kv to capacity performs as expected"""

    tc = TransmissionConfig(
        {
            "power_classes": {"100MW": 102, "750MW": 750, "500MW": 400},
            "power_to_voltage": {"102": 135, "750": 345, "400": 400},
        }
    )
    assert tc.capacity_to_kv("100") == 135
    assert tc.capacity_to_kv("750MW") == 345
    assert tc.capacity_to_kv(500) == 400

    assert tc.kv_to_capacity(135) == 100
    assert tc.kv_to_capacity(345) == 750
    assert tc.kv_to_capacity(400) == 500


def test_substation_upgrade_costs():
    """Test that substation upgrade costs are extracted correctly"""

    tc = TransmissionConfig(
        {
            "iso_lookup": {"iso_1": 5, "iso2": 42},
            "upgrade_substation_costs": {
                "iso_1": {"138": 100},
                "iso2": {"138": 99_999},
            },
        }
    )

    assert tc.sub_upgrade_cost(region=5, tie_line_voltage=138) == 100
    assert tc.sub_upgrade_cost(region=42, tie_line_voltage="138") == 99_999


def test_new_substation_costs():
    """Test that new substation costs are extracted correctly"""

    tc = TransmissionConfig(
        {
            "iso_lookup": {"iso_1": 5, "iso2": 42},
            "new_substation_costs": {
                "iso_1": {"138": 100},
                "iso2": {"138": 99_999},
            },
        }
    )

    assert tc.new_sub_cost(region=5, tie_line_voltage=138) == 100
    assert tc.new_sub_cost(region=42, tie_line_voltage="138") == 99_999


def test_transformer_cost():
    """Test that transformer costs are extracted correctly"""

    tc = TransmissionConfig(
        {
            "transformer_costs": {
                "115": {"115": 10, "138": 100},
                "138": {"115": 1000, "138": 99_999},
            }
        }
    )

    assert tc.transformer_cost(feature_voltage=10, tie_line_voltage=115) == 10
    assert tc.transformer_cost(feature_voltage=115, tie_line_voltage=115) == 10
    assert (
        tc.transformer_cost(feature_voltage=120, tie_line_voltage=115) == 100
    )
    assert (
        tc.transformer_cost(feature_voltage=138, tie_line_voltage=115) == 100
    )
    assert (
        tc.transformer_cost(feature_voltage=200, tie_line_voltage=115) == 100
    )

    assert (
        tc.transformer_cost(feature_voltage=10, tie_line_voltage="138") == 1000
    )
    assert (
        tc.transformer_cost(feature_voltage=115, tie_line_voltage="138")
        == 1000
    )
    assert (
        tc.transformer_cost(feature_voltage=120, tie_line_voltage="138")
        == 99_999
    )
    assert (
        tc.transformer_cost(feature_voltage=138, tie_line_voltage="138")
        == 99_999
    )
    assert (
        tc.transformer_cost(feature_voltage=200, tie_line_voltage="138")
        == 99_999
    )


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
