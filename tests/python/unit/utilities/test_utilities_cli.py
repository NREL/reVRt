"""Tests for CLI utility commands"""

from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point, Polygon

from revrt.exceptions import revrtValueError
from revrt.utilities import cli
from revrt.warn import revrtWarning


def _capture_to_file(monkeypatch):
    capture = {}

    def _writer(self, path, driver=None, index=True, **kwargs):
        capture["frame"] = self.copy()
        capture["path"] = Path(path)
        capture["driver"] = driver
        capture["index"] = index
        capture["kwargs"] = kwargs

    monkeypatch.setattr(gpd.GeoDataFrame, "to_file", _writer, raising=False)
    return capture


@pytest.fixture
def noop_to_crs(monkeypatch):
    """Provide noop CRS conversion for GeoDataFrames"""

    def _noop(self, crs):
        self.crs = crs
        return self

    monkeypatch.setattr(gpd.GeoDataFrame, "to_crs", _noop, raising=False)


def _write_template_raster(path):
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


def test_convert_pois_to_lines_creates_fake_trans_line(monkeypatch, tmp_path):
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

    capture = _capture_to_file(monkeypatch)
    cli.convert_pois_to_lines(poi_csv, template, tmp_path / "out.gpkg")

    frame = capture["frame"]
    assert capture["driver"] == "GPKG"
    assert set(frame["category"]) == {"Substation", "TransLine"}
    assert 9999 in frame["gid"].tolist()


def test_map_ss_to_rr_filters_low_voltage(monkeypatch, tmp_path, noop_to_crs):
    """Ensure low voltage substations are dropped with warning"""
    features_path = tmp_path / "features.gpkg"
    regions_path = tmp_path / "regions.gpkg"
    out_file = tmp_path / "subs.gpkg"

    features = gpd.GeoDataFrame(
        {
            "category": ["Substation", "Substation", "Line", "Line"],
            "gid": [1, 2, 10, 11],
            "trans_gids": ["[10]", [11], None, None],
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

    def _reader(path):
        if Path(path) == features_path:
            return features.copy()
        if Path(path) == regions_path:
            return regions.copy()
        msg = "Unexpected read path"
        raise AssertionError(msg)

    monkeypatch.setattr(cli.gpd, "read_file", _reader)
    capture = _capture_to_file(monkeypatch)

    with pytest.warns(revrtWarning):
        cli.map_ss_to_rr(
            features_path,
            regions_path,
            "region_id",
            out_file=out_file,
        )

    frame = capture["frame"]
    assert frame["gid"].tolist() == [1]
    assert frame["region_id"].tolist() == ["A"]
    assert frame.loc[0, "min_volts"] == 115
    assert frame.loc[0, "max_volts"] == 115


def test_map_ss_to_rr_without_drops(monkeypatch, tmp_path, noop_to_crs):
    """Ensure substations remain when voltage threshold is met"""
    features_path = tmp_path / "features.gpkg"
    regions_path = tmp_path / "regions.gpkg"

    features = gpd.GeoDataFrame(
        {
            "category": ["Substation", "Substation", "Line", "Line"],
            "gid": [1, 2, 10, 11],
            "trans_gids": ["[10]", [11], None, None],
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

    def _reader(path):
        if Path(path) == features_path:
            return features.copy()
        if Path(path) == regions_path:
            return regions.copy()
        msg = "Unexpected read path"
        raise AssertionError(msg)

    monkeypatch.setattr(cli.gpd, "read_file", _reader)
    capture = _capture_to_file(monkeypatch)

    cli.map_ss_to_rr(
        features_path,
        regions_path,
        "region_id",
        out_file=tmp_path / "remaining_subs.gpkg",
    )

    frame = capture["frame"]
    assert frame["gid"].tolist() == [1, 2]
    assert frame["region_id"].tolist() == ["A", "B"]


def test_ss_from_conn_csv(monkeypatch, tmp_path):
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

    capture = _capture_to_file(monkeypatch)
    cli.ss_from_conn(str(csv_path), "region_id", str(tmp_path / "out.gpkg"))

    frame = capture["frame"]
    assert capture["driver"] == "GPKG"
    assert len(frame) == 1
    assert str(frame["poi_gid"].dtype) == "Int64"


def test_ss_from_conn_gpkg(monkeypatch):
    """Validate GeoPackage extraction matches CSV logic"""

    data = gpd.GeoDataFrame(
        {
            "poi_gid": [1],
            "poi_lat": [10.0],
            "poi_lon": [100.0],
            "region_id": ["A"],
        }
    )

    def _reader(path):
        return data.copy()

    monkeypatch.setattr(cli.gpd, "read_file", _reader)
    capture = _capture_to_file(monkeypatch)
    cli.ss_from_conn("connections.gpkg", "region_id", "out.gpkg")

    assert len(capture["frame"]) == 1


def test_ss_from_conn_invalid_extension():
    """Ensure invalid connection file extensions raise error"""

    with pytest.raises(revrtValueError):
        cli.ss_from_conn("connections.txt", "region_id", "out.gpkg")


def test_add_rr_to_nn_with_zarr_template(monkeypatch, tmp_path, noop_to_crs):
    """Ensure add_rr_to_nn reads CRS from Zarr template"""

    network_path = tmp_path / "network.gpkg"
    regions_path = tmp_path / "regions.gpkg"
    template_path = tmp_path / "template.zarr"

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

    def _reader(path):
        if Path(path) == network_path:
            return network.copy()
        if Path(path) == regions_path:
            return regions.copy()
        msg = "Unexpected read path"
        raise AssertionError(msg)

    class _Dataset:
        def __init__(self, crs):
            self.rio = SimpleNamespace(crs=crs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(cli.gpd, "read_file", _reader)
    monkeypatch.setattr(
        cli.xr, "open_dataset", lambda *_, **__: _Dataset("EPSG:4326")
    )

    capture = _capture_to_file(monkeypatch)
    cli.add_rr_to_nn(
        network_path,
        regions_path,
        "region",
        crs_template_file=template_path,
        out_file=tmp_path / "out.gpkg",
    )

    frame = capture["frame"]
    assert frame["region"].tolist() == ["west", "east"]


def test_add_rr_to_nn_with_existing_region_column(
    monkeypatch, tmp_path, noop_to_crs
):
    """Validate add_rr_to_nn warns when region column exists"""

    network_path = tmp_path / "network.gpkg"
    regions_path = tmp_path / "regions.gpkg"
    template_path = tmp_path / "template.zarr"

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

    def _reader(path):
        if Path(path) == network_path:
            return network.copy()
        if Path(path) == regions_path:
            return regions.copy()
        msg = "Unexpected read path"
        raise AssertionError(msg)

    class _Dataset:
        def __init__(self, crs):
            self.rio = SimpleNamespace(crs=crs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(cli.gpd, "read_file", _reader)
    monkeypatch.setattr(
        cli.xr, "open_dataset", lambda *_, **__: _Dataset("EPSG:4326")
    )

    written = []

    def _writer(self, path, driver=None, index=True, **kwargs):
        written.append(path)

    monkeypatch.setattr(gpd.GeoDataFrame, "to_file", _writer, raising=False)

    with pytest.warns(revrtWarning):
        cli.add_rr_to_nn(
            network_path,
            regions_path,
            "region",
            crs_template_file=template_path,
            out_file=tmp_path / "out.gpkg",
        )

    assert not written


def test_add_rr_to_nn_with_tif_template(monkeypatch, tmp_path, noop_to_crs):
    """Ensure add_rr_to_nn reads CRS from GeoTIFF template"""

    network_path = tmp_path / "network.gpkg"
    regions_path = tmp_path / "regions.gpkg"
    template_path = tmp_path / "template.tif"

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

    def _reader(path):
        if Path(path) == network_path:
            return network.copy()
        if Path(path) == regions_path:
            return regions.copy()
        msg = "Unexpected read path"
        raise AssertionError(msg)

    class _Geo:
        def __init__(self, crs):
            self.rio = SimpleNamespace(crs=crs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(cli.gpd, "read_file", _reader)
    monkeypatch.setattr(
        cli.rioxarray, "open_rasterio", lambda *_, **__: _Geo("EPSG:4326")
    )

    capture = _capture_to_file(monkeypatch)
    cli.add_rr_to_nn(
        network_path,
        regions_path,
        "region",
        crs_template_file=template_path,
        out_file=tmp_path / "out.gpkg",
    )

    assert capture["frame"]["region"].tolist() == ["west"]


def test_add_rr_to_nn_default_output_path(monkeypatch, tmp_path, noop_to_crs):
    """Ensure add_rr_to_nn uses default output path when omitted"""

    network_path = tmp_path / "network.gpkg"
    regions_path = tmp_path / "regions.gpkg"

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

    def _reader(path):
        if Path(path) == network_path:
            return network.copy()
        if Path(path) == regions_path:
            return regions.copy()
        msg = "Unexpected read path"
        raise AssertionError(msg)

    monkeypatch.setattr(cli.gpd, "read_file", _reader)

    capture = _capture_to_file(monkeypatch)
    cli.add_rr_to_nn(network_path, regions_path, "region")

    assert capture["path"] == tmp_path / "network.gpkg"


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
