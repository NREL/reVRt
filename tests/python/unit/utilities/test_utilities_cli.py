"""Tests for CLI utility commands"""

import json
from pathlib import Path

import pytest
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point, Polygon

from revrt.exceptions import revrtValueError
from revrt.utilities import cli
from revrt.utilities.handlers import LayeredFile
from revrt.warn import revrtWarning


def _write_template_raster(path):
    """Helper function to write a template raster for tests"""
    transform = from_origin(0, 1, 1, 1)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=1,
        width=1,
        count=1,
        dtype=np.uint8,
        crs="EPSG:4326",
        transform=transform,
    ) as dataset:
        dataset.write(np.ones((1, 1), dtype=np.uint8), 1)


def _write_template_zarr(path):
    """Helper function to write a template Zarr file for tests"""
    template_tif = Path(path).with_suffix(".tif")
    _write_template_raster(template_tif)
    layered = LayeredFile(path)
    layered.create_new(
        template_tif,
        overwrite=True,
        chunk_x=1,
        chunk_y=1,
    )
    height = layered.profile["height"]
    width = layered.profile["width"]
    values = np.zeros((1, height, width), dtype=np.uint8)
    layered.write_layer(values, "layer", overwrite=True)
    template_tif.unlink(missing_ok=True)


def test_layers_from_file_selects_specific_layers(monkeypatch, tmp_path):
    """Ensure layers_from_file extracts chosen layers"""

    recorded = {}

    class DummyLayeredFile:
        def __init__(self, path):
            recorded["path"] = path

        def extract_layers(self, layer_map, **kwargs):
            recorded["layers"] = layer_map
            recorded["kwargs"] = kwargs

    monkeypatch.setattr(cli, "LayeredFile", DummyLayeredFile)
    result = cli.layers_from_file(
        "input.zarr",
        tmp_path,
        layers=["alpha", "beta"],
        profile_kwargs={"compress": "LZW"},
    )

    assert recorded["path"] == "input.zarr"
    assert recorded["layers"]["alpha"] == tmp_path / "alpha.tif"
    assert recorded["kwargs"] == {"compress": "LZW"}

    assert result == [str(tmp_path / "alpha.tif"), str(tmp_path / "beta.tif")]


def test_layers_from_file_extracts_all_layers(monkeypatch, tmp_path):
    """Ensure layers_from_file calls extract_all_layers when needed"""

    expected = {"one": tmp_path / "one.tif"}

    class DummyLayeredFile:
        def __init__(self, path):
            self.path = path

        def extract_all_layers(self, out_dir, **kwargs):
            assert out_dir == tmp_path
            assert not kwargs
            return expected

    monkeypatch.setattr(cli, "LayeredFile", DummyLayeredFile)
    result = cli.layers_from_file("input.zarr", tmp_path)
    assert result == [str(tmp_path / "one.tif")]


def test_preprocess_layers_from_file_config(tmp_path):
    """Ensure preprocess injects layer dir"""

    config = {"foo": 1}
    out_dir = tmp_path / "layers"

    processed = cli._preprocess_layers_from_file_config(
        config, tmp_path, out_dir
    )

    assert processed["_out_layer_dir"] == str(out_dir)


def test_convert_pois_to_lines_creates_fake_trans_line(tmp_path):
    """Validate POI conversion includes fake transmission entry"""

    template = tmp_path / "template.tif"
    _write_template_raster(template)

    poi_csv = tmp_path / "pois.csv"
    pd.DataFrame(
        {
            "POI Name": ["Test"],
            "State": ["CO"],
            "Voltage (kV)": [115],
            "Lat": [40.0],
            "Long": [-105.0],
        }
    ).to_csv(poi_csv, index=False)

    out_file = tmp_path / "out.gpkg"
    cli.convert_pois_to_lines(poi_csv, template, out_file)

    frame = gpd.read_file(out_file)
    assert set(frame["category"]) == {"Substation", "TransLine"}
    assert 9999 in frame["gid"].tolist()


def test_map_ss_to_rr_filters_low_voltage(tmp_path):
    """Ensure low voltage substations are dropped with warning"""

    features = gpd.GeoDataFrame(
        {
            "category": ["Substation", "Substation", "Line", "Line"],
            "gid": [1, 2, 10, 11],
            "trans_gids": [json.dumps([10]), json.dumps([11]), None, None],
            "voltage": [0, 0, 115, 34],
        },
        geometry=[
            Point(0.0, 0.0),
            Point(5.0, 0.0),
            LineString([(0.0, 0.0), (0.0, 1.0)]),
            LineString([(5.0, 0.0), (5.0, 1.0)]),
        ],
        crs="EPSG:4326",
    )
    regions = gpd.GeoDataFrame(
        {"region_id": ["A", "B"]},
        geometry=[
            Polygon([(-1, -1), (-1, 1), (1, 1), (1, -1)]),
            Polygon([(4, -1), (4, 1), (6, 1), (6, -1)]),
        ],
        crs="EPSG:4326",
    )

    features_path = tmp_path / "features.gpkg"
    regions_path = tmp_path / "regions.gpkg"
    features.to_file(features_path, driver="GPKG")
    regions.to_file(regions_path, driver="GPKG")

    out_file = tmp_path / "subs.gpkg"
    with pytest.warns(revrtWarning):
        cli.map_ss_to_rr(
            features_path,
            regions_path,
            "region_id",
            out_file=out_file,
        )

    frame = gpd.read_file(out_file)
    assert frame["gid"].tolist() == [1]
    assert frame["region_id"].tolist() == ["A"]
    assert frame.loc[0, "min_volts"] == 115
    assert frame.loc[0, "max_volts"] == 115


def test_map_ss_to_rr_without_drops(tmp_path):
    """Ensure substations remain when voltage threshold is met"""

    features = gpd.GeoDataFrame(
        {
            "category": ["Substation", "Substation", "Line", "Line"],
            "gid": [1, 2, 10, 11],
            "trans_gids": [json.dumps([10]), json.dumps([11]), None, None],
            "voltage": [0, 0, 115, 230],
        },
        geometry=[
            Point(0.0, 0.0),
            Point(5.0, 0.0),
            LineString([(0.0, 0.0), (0.0, 1.0)]),
            LineString([(5.0, 0.0), (5.0, 1.0)]),
        ],
        crs="EPSG:4326",
    )
    regions = gpd.GeoDataFrame(
        {"region_id": ["A", "B"]},
        geometry=[
            Polygon([(-1, -1), (-1, 1), (1, 1), (1, -1)]),
            Polygon([(4, -1), (4, 1), (6, 1), (6, -1)]),
        ],
        crs="EPSG:4326",
    )

    features_path = tmp_path / "features.gpkg"
    regions_path = tmp_path / "regions.gpkg"
    out_file = tmp_path / "remaining_subs.gpkg"

    features.to_file(features_path, driver="GPKG")
    regions.to_file(regions_path, driver="GPKG")

    cli.map_ss_to_rr(
        features_path,
        regions_path,
        "region_id",
        out_file=out_file,
    )

    frame = gpd.read_file(out_file)
    assert frame["gid"].tolist() == [1, 2]
    assert frame["region_id"].tolist() == ["A", "B"]


def test_ss_from_conn_csv(tmp_path):
    """Validate CSV extraction filters null records"""
    csv_path = tmp_path / "connections.csv"
    pd.DataFrame(
        {
            "poi_gid": [1, 1, np.nan],
            "poi_lat": [10.0, 10.0, 40.0],
            "poi_lon": [100.0, 100.0, 120.0],
            "region_id": ["A", "A", "B"],
        }
    ).to_csv(csv_path, index=False)

    out_file = tmp_path / "out.gpkg"
    cli.ss_from_conn(str(csv_path), "region_id", str(out_file))

    frame = gpd.read_file(out_file)
    assert len(frame) == 1
    assert pd.api.types.is_integer_dtype(frame["poi_gid"])


def test_ss_from_conn_gpkg(tmp_path):
    """Validate GeoPackage extraction matches CSV logic"""

    connections = gpd.GeoDataFrame(
        {
            "poi_gid": [1],
            "poi_lat": [10.0],
            "poi_lon": [100.0],
            "region_id": ["A"],
        },
        geometry=[Point(100.0, 10.0)],
        crs="EPSG:4326",
    )

    connections_path = tmp_path / "connections.gpkg"
    out_file = tmp_path / "out.gpkg"
    connections.to_file(connections_path, driver="GPKG")

    cli.ss_from_conn(str(connections_path), "region_id", str(out_file))

    frame = gpd.read_file(out_file)
    assert len(frame) == 1


def test_ss_from_conn_invalid_extension():
    """Ensure invalid connection file extensions raise error"""

    with pytest.raises(revrtValueError):
        cli.ss_from_conn("connections.txt", "region_id", "out.gpkg")


def test_add_rr_to_nn_with_zarr_template(tmp_path):
    """Ensure add_rr_to_nn reads CRS from Zarr template"""

    network = gpd.GeoDataFrame(
        {"value": [1, 2]},
        geometry=[Point(0.0, 0.0), Point(10.0, 0.0)],
        crs="EPSG:4326",
    )
    regions = gpd.GeoDataFrame(
        {"region": ["west", "east"]},
        geometry=[
            Polygon([(-1, -1), (-1, 1), (1, 1), (1, -1)]),
            Polygon([(9, -1), (9, 1), (11, 1), (11, -1)]),
        ],
        crs="EPSG:4326",
    )

    network_path = tmp_path / "network.gpkg"
    regions_path = tmp_path / "regions.gpkg"
    template_path = tmp_path / "template.zarr"
    out_file = tmp_path / "out.gpkg"

    network.to_file(network_path, driver="GPKG")
    regions.to_file(regions_path, driver="GPKG")
    _write_template_zarr(template_path)

    cli.add_rr_to_nn(
        network_path,
        regions_path,
        "region",
        crs_template_file=template_path,
        out_file=out_file,
    )

    frame = gpd.read_file(out_file)
    assert frame["region"].tolist() == ["west", "east"]


def test_add_rr_to_nn_with_existing_region_column(tmp_path):
    """Validate add_rr_to_nn warns when region column exists"""

    network = gpd.GeoDataFrame(
        {"region": ["existing", "existing"]},
        geometry=[Point(0.0, 0.0), Point(1.0, 1.0)],
        crs="EPSG:4326",
    )
    regions = gpd.GeoDataFrame(
        {"region": ["west"]},
        geometry=[Polygon([(-1, -1), (-1, 1), (1, 1), (1, -1)])],
        crs="EPSG:4326",
    )

    network_path = tmp_path / "network.gpkg"
    regions_path = tmp_path / "regions.gpkg"
    template_path = tmp_path / "template.zarr"
    out_file = tmp_path / "out.gpkg"

    network.to_file(network_path, driver="GPKG")
    regions.to_file(regions_path, driver="GPKG")
    _write_template_zarr(template_path)

    with pytest.warns(revrtWarning):
        cli.add_rr_to_nn(
            network_path,
            regions_path,
            "region",
            crs_template_file=template_path,
            out_file=out_file,
        )

    assert not out_file.exists()


def test_add_rr_to_nn_with_tif_template(tmp_path):
    """Ensure add_rr_to_nn reads CRS from GeoTIFF template"""

    network = gpd.GeoDataFrame(
        {"value": [1]},
        geometry=[Point(0.0, 0.0)],
        crs="EPSG:4326",
    )
    regions = gpd.GeoDataFrame(
        {"region": ["west"]},
        geometry=[Polygon([(-1, -1), (-1, 1), (1, 1), (1, -1)])],
        crs="EPSG:4326",
    )

    network_path = tmp_path / "network.gpkg"
    regions_path = tmp_path / "regions.gpkg"
    template_path = tmp_path / "template.tif"
    out_file = tmp_path / "out.gpkg"

    network.to_file(network_path, driver="GPKG")
    regions.to_file(regions_path, driver="GPKG")
    _write_template_raster(template_path)

    cli.add_rr_to_nn(
        network_path,
        regions_path,
        "region",
        crs_template_file=template_path,
        out_file=out_file,
    )

    frame = gpd.read_file(out_file)
    assert frame["region"].tolist() == ["west"]


def test_add_rr_to_nn_default_output_path(tmp_path):
    """Ensure add_rr_to_nn uses default output path when omitted"""

    network = gpd.GeoDataFrame(
        {"value": [1]},
        geometry=[Point(0.0, 0.0)],
        crs="EPSG:4326",
    )
    regions = gpd.GeoDataFrame(
        {"region": ["west"]},
        geometry=[Polygon([(-1, -1), (-1, 1), (1, 1), (1, -1)])],
        crs="EPSG:4326",
    )

    network_path = tmp_path / "network.gpkg"
    regions_path = tmp_path / "regions.gpkg"

    network.to_file(network_path, driver="GPKG")
    regions.to_file(regions_path, driver="GPKG")

    cli.add_rr_to_nn(network_path, regions_path, "region")

    frame = gpd.read_file(network_path)
    assert frame["region"].tolist() == ["west"]


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
