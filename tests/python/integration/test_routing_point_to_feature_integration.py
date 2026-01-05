"""Integration tests for point to feature routing"""

import os
import json
import math
import platform
import traceback
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
import xarray as xr

from revrt._cli import main
from revrt.exceptions import revrtValueError
from revrt.routing.cli_point_to_features import (
    build_point_to_feature_route_table_command,
    point_to_feature_route_table,
)
from revrt.routing.utilities import PointToFeatureMapper, make_rev_sc_points
from revrt.warn import revrtWarning


@pytest.fixture(scope="module")
def routing_test_inputs(tmp_path_factory, test_data_dir):
    """Prepare reusable inputs for mapping tests"""

    work_dir = tmp_path_factory.mktemp("routing_point_to_feature")
    features_src = test_data_dir / "routing" / "ri_transmission_features.gpkg"
    regions_src = test_data_dir / "routing" / "ri_regions.gpkg"

    features = gpd.read_file(features_src).copy(deep=True)
    if "category" in features.columns:
        features.loc[features.index[0], "category"] = None
    features_fp = work_dir / "features_subset.gpkg"
    features.to_file(features_fp, driver="GPKG")

    regions = gpd.read_file(regions_src).copy(deep=True)
    regions_fp = work_dir / "regions_subset.gpkg"
    regions.to_file(regions_fp, driver="GPKG")

    return {
        "features_fp": features_fp,
        "regions_fp": regions_fp,
        "features": features,
        "regions": regions,
    }


@pytest.fixture(scope="module")
def cost_metadata(revx_transmission_layers):
    """Return metadata describing the test routing cost grid"""

    with xr.open_dataset(
        revx_transmission_layers, consolidated=False, engine="zarr"
    ) as ds:
        return {
            "crs": ds.rio.crs,
            "transform": ds.rio.transform(),
            "shape": (ds.rio.height, ds.rio.width),
        }


def _determine_sparse_resolution(shape):
    """Pick a resolution that keeps the supply curve grid small"""

    height, width = shape
    resolution = max(height, width)
    while math.ceil(height / resolution) * math.ceil(width / resolution) < 3:
        if resolution == 1:
            break
        resolution = max(1, resolution // 2)
    return resolution


def test_point_to_feature_route_table_generates_expected_outputs(
    tmp_path, routing_test_inputs, cost_metadata, revx_transmission_layers
):
    """Validate that the integration helper writes consistent outputs"""

    resolution = _determine_sparse_resolution(cost_metadata["shape"])
    out_dir = tmp_path / "integration"
    out_dir.mkdir()

    with pytest.warns(revrtWarning) as warn_records:
        outputs = point_to_feature_route_table(
            revx_transmission_layers,
            routing_test_inputs["features_fp"],
            out_dir,
            regions_fpath=routing_test_inputs["regions_fp"],
            resolution=resolution,
            radius=50_000,
            expand_radius=True,
            feature_out_fp="mapped_features",
            route_table_out_fp="route_table",
            region_identifier_column="region_key",
            feature_identifier_column="custom_feat_id",
            batch_size=2,
        )

    warn_messages = [str(w.message) for w in warn_records]
    assert any("'.gpkg' extension" in msg for msg in warn_messages)
    assert any("'.csv' extension" in msg for msg in warn_messages)
    assert any(
        "Dropping" in msg or "will be dropped" in msg for msg in warn_messages
    )

    route_table_path = Path(outputs[0])
    mapped_features_path = Path(outputs[1])
    assert route_table_path.exists()
    assert mapped_features_path.exists()

    route_table = pd.read_csv(route_table_path)
    mapped_features = gpd.read_file(mapped_features_path)

    height, width = cost_metadata["shape"]
    assert {"start_row", "start_col", "custom_feat_id"}.issubset(
        route_table.columns
    )
    assert route_table["start_row"].between(0, height - 1).all()
    assert route_table["start_col"].between(0, width - 1).all()
    assert route_table["custom_feat_id"].tolist() == list(
        range(len(route_table))
    )
    assert "region_key" in route_table.columns
    assert route_table["region_key"].notna().all()
    assert {"custom_feat_id", "region_key"}.issubset(mapped_features.columns)
    assert not mapped_features["category"].isna().any()
    assert set(mapped_features["custom_feat_id"].unique()) == set(
        route_table["custom_feat_id"]
    )


def test_cli_build_feature_route_table(
    tmp_path, routing_test_inputs, cost_metadata, revx_transmission_layers
):
    """Ensure the CLI command mirrors the integration helper"""

    resolution = _determine_sparse_resolution(cost_metadata["shape"])
    out_dir = tmp_path / "cli"
    out_dir.mkdir()

    outputs = build_point_to_feature_route_table_command.runner(
        cost_fpath=revx_transmission_layers,
        features_fpath=routing_test_inputs["features_fp"],
        out_dir=out_dir,
        regions_fpath=routing_test_inputs["regions_fp"],
        resolution=resolution,
        radius=75_000,
        feature_out_fp="cli_features.gpkg",
        route_table_out_fp="cli_routes.csv",
        region_identifier_column="rid",
        feature_identifier_column="cli_feat_id",
        batch_size=2,
    )

    route_table_path = Path(outputs[0])
    mapped_features_path = Path(outputs[1])
    assert route_table_path.exists()
    assert mapped_features_path.exists()

    route_table = pd.read_csv(route_table_path)
    mapped_features = gpd.read_file(mapped_features_path)
    assert "cli_feat_id" in route_table.columns
    assert "cli_feat_id" in mapped_features.columns
    assert set(mapped_features["cli_feat_id"].unique()) == set(
        route_table["cli_feat_id"]
    )


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_cli_build_feature_route_table_from_config(
    tmp_path,
    routing_test_inputs,
    cost_metadata,
    revx_transmission_layers,
    cli_runner,
):
    """Run CLI main entry via config file to build feature route table"""

    resolution = _determine_sparse_resolution(cost_metadata["shape"])
    out_dir = tmp_path / "cli_config"
    out_dir.mkdir()

    config = {
        "cost_fpath": str(revx_transmission_layers),
        "features_fpath": str(routing_test_inputs["features_fp"]),
        "out_dir": str(out_dir),
        "regions_fpath": str(routing_test_inputs["regions_fp"]),
        "resolution": resolution,
        "radius": 75_000,
        "feature_out_fp": "config_features.gpkg",
        "route_table_out_fp": "config_routes.csv",
        "region_identifier_column": "rid",
        "feature_identifier_column": "cfg_feat_id",
        "batch_size": 2,
        "expand_radius": True,
    }

    config_path = out_dir / "build_feature_route_table.json"
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(config, fh)

    result = cli_runner.invoke(
        main, ["build-feature-route-table", "-c", str(config_path)]
    )
    err_msg = None
    if result.exc_info is not None:
        err_msg = "".join(traceback.format_exception(*result.exc_info))
    assert result.exit_code == 0, err_msg

    route_table_path = out_dir / "config_routes.csv"
    mapped_features_path = out_dir / "config_features.gpkg"
    assert route_table_path.exists()
    assert mapped_features_path.exists()

    route_table = pd.read_csv(route_table_path)
    mapped_features = gpd.read_file(mapped_features_path)

    height, width = cost_metadata["shape"]
    assert {"start_row", "start_col", "cfg_feat_id"}.issubset(
        route_table.columns
    )
    assert route_table["start_row"].between(0, height - 1).all()
    assert route_table["start_col"].between(0, width - 1).all()
    assert route_table["cfg_feat_id"].notna().all()
    assert "rid" in route_table.columns
    assert route_table["rid"].notna().all()
    assert "cfg_feat_id" in mapped_features.columns
    assert "rid" in mapped_features.columns
    assert set(mapped_features["cfg_feat_id"].unique()) == set(
        route_table["cfg_feat_id"]
    )


def test_point_to_feature_mapper_requires_radius_or_regions(
    tmp_path, routing_test_inputs, cost_metadata
):
    """Ensure an explicit radius or region constraint is required"""

    mapper = PointToFeatureMapper(
        cost_metadata["crs"], routing_test_inputs["features_fp"]
    )
    points = make_rev_sc_points(
        cost_metadata["shape"][0],
        cost_metadata["shape"][1],
        cost_metadata["crs"],
        cost_metadata["transform"],
        resolution=_determine_sparse_resolution(cost_metadata["shape"]),
    )
    with pytest.raises(
        revrtValueError,
        match=(
            "Must provide either `regions` or a radius to map points "
            "to features!"
        ),
    ):
        mapper.map_points(points, tmp_path / "unreachable.gpkg")


def test_point_to_feature_mapper_extension_warning_and_radius_column(
    tmp_path, routing_test_inputs, cost_metadata, monkeypatch
):
    """Radius column inputs trigger streaming writer extension warnings"""

    mapper = PointToFeatureMapper(
        cost_metadata["crs"], routing_test_inputs["features_fp"]
    )
    points = make_rev_sc_points(
        cost_metadata["shape"][0],
        cost_metadata["shape"][1],
        cost_metadata["crs"],
        cost_metadata["transform"],
        resolution=_determine_sparse_resolution(cost_metadata["shape"]),
    ).head(1)
    points["radius_m"] = 10.0

    original = PointToFeatureMapper._clipped_features

    def _clipped_features_once(self, region, features=None):
        if not getattr(self, "_triggered_once", False):
            self._triggered_once = True
            return gpd.GeoDataFrame(geometry=[], crs=self._crs)
        return original(self, region, features)

    monkeypatch.setattr(
        PointToFeatureMapper, "_clipped_features", _clipped_features_once
    )

    with pytest.warns(revrtWarning) as warn_records:
        mapped = mapper.map_points(
            points,
            tmp_path / "features_no_extension",
            radius="radius_m",
            expand_radius=True,
            batch_size=1,
        )

    warn_messages = [str(w.message) for w in warn_records]
    assert any("'.gpkg' extension" in msg for msg in warn_messages)
    assert mapped["end_feat_id"].tolist() == [0]


def test_clip_to_radius_returns_empty_input(
    routing_test_inputs, cost_metadata
):
    """Do not modify empty feature inputs when clipping by radius"""

    mapper = PointToFeatureMapper(
        cost_metadata["crs"], routing_test_inputs["features_fp"]
    )
    points = make_rev_sc_points(
        cost_metadata["shape"][0],
        cost_metadata["shape"][1],
        cost_metadata["crs"],
        cost_metadata["transform"],
        resolution=_determine_sparse_resolution(cost_metadata["shape"]),
    ).head(1)
    empty = gpd.GeoDataFrame(geometry=[], crs=mapper._crs)

    clipped = mapper._clip_to_radius(points.iloc[0], 100.0, empty, True)
    assert clipped is empty


def test_point_to_feature_mapper_accepts_region_path(
    tmp_path, routing_test_inputs, cost_metadata
):
    """Paths are valid inputs for region clipping"""

    mapper = PointToFeatureMapper(
        cost_metadata["crs"],
        routing_test_inputs["features_fp"],
        regions=routing_test_inputs["regions_fp"],
    )
    points = make_rev_sc_points(
        cost_metadata["shape"][0],
        cost_metadata["shape"][1],
        cost_metadata["crs"],
        cost_metadata["transform"],
        resolution=_determine_sparse_resolution(cost_metadata["shape"]),
    ).head(2)

    mapped = mapper.map_points(
        points,
        tmp_path / "path_regions",
        radius=100_000,
        expand_radius=False,
        batch_size=2,
    )

    assert mapped["end_feat_id"].tolist() == list(range(len(mapped)))


def test_point_to_feature_route_table_without_regions(
    tmp_path, routing_test_inputs, cost_metadata, revx_transmission_layers
):
    """Route table helper runs without region file when radius set"""

    resolution = _determine_sparse_resolution(cost_metadata["shape"])
    out_dir = tmp_path / "no_regions"
    out_dir.mkdir()

    outputs = point_to_feature_route_table(
        revx_transmission_layers,
        routing_test_inputs["features_fp"],
        out_dir,
        resolution=resolution,
        radius=40_000,
        expand_radius=False,
        feature_out_fp="no_regions_features.gpkg",
        route_table_out_fp="no_regions_routes.csv",
        batch_size=1,
    )

    assert Path(outputs[0]).exists()
    assert Path(outputs[1]).exists()


def test_point_to_feature_mapper_preserves_existing_region_id(
    routing_test_inputs, cost_metadata
):
    """Existing region identifiers are preserved without reassignment"""

    regions = (
        routing_test_inputs["regions"].to_crs(cost_metadata["crs"]).copy()
    )
    regions["rid"] = range(len(regions))

    mapper = PointToFeatureMapper(
        cost_metadata["crs"],
        routing_test_inputs["features_fp"],
        regions=regions,
        region_identifier_column="rid",
    )

    assert mapper._regions["rid"].tolist() == regions["rid"].tolist()


def test_map_points_flushes_remaining_batch(
    tmp_path, routing_test_inputs, cost_metadata, monkeypatch
):
    """map_points writes any trailing batch after iteration completes"""

    mapper = PointToFeatureMapper(
        cost_metadata["crs"], routing_test_inputs["features_fp"]
    )
    points = make_rev_sc_points(
        cost_metadata["shape"][0],
        cost_metadata["shape"][1],
        cost_metadata["crs"],
        cost_metadata["transform"],
        resolution=_determine_sparse_resolution(cost_metadata["shape"]),
    ).head(2)

    class DummyWriter:
        def __init__(self):
            self.saved = []

        def save(self, data):
            self.saved.append(data.copy())

    dummy_writer = DummyWriter()
    monkeypatch.setattr(
        "revrt.routing.utilities._init_streaming_writer",
        lambda __: dummy_writer,
    )

    fake_features = (
        routing_test_inputs["features"].to_crs(cost_metadata["crs"]).head(1)
    ).reset_index(drop=True)

    def _fake_clip_to_point(self, point, radius, expand_radius):
        return fake_features.copy()

    monkeypatch.setattr(
        PointToFeatureMapper, "_clip_to_point", _fake_clip_to_point
    )

    mapped = mapper.map_points(
        points,
        tmp_path / "flush_batch",
        radius=10_000,
        batch_size=5,
    )

    assert len(dummy_writer.saved) == 1
    assert len(mapped) == len(points)


def test_clip_to_point_uses_radius_branch(
    routing_test_inputs, cost_metadata, monkeypatch
):
    """_clip_to_point applies both region and radius clipping when needed"""

    mapper = PointToFeatureMapper(
        cost_metadata["crs"],
        routing_test_inputs["features_fp"],
        regions=routing_test_inputs["regions"],
    )

    region_features = (
        routing_test_inputs["features"].to_crs(cost_metadata["crs"]).head(1)
    )

    def _fake_clip_region(self, point):
        return region_features.copy()

    def _fake_clip_radius(self, point, radius, features, expand_radius):
        expanded = features.copy()
        expanded["expanded_radius"] = radius
        return expanded

    monkeypatch.setattr(
        PointToFeatureMapper, "_clip_to_region", _fake_clip_region
    )
    monkeypatch.setattr(
        PointToFeatureMapper, "_clip_to_radius", _fake_clip_radius
    )

    points = make_rev_sc_points(
        cost_metadata["shape"][0],
        cost_metadata["shape"][1],
        cost_metadata["crs"],
        cost_metadata["transform"],
        resolution=_determine_sparse_resolution(cost_metadata["shape"]),
    )

    result = mapper._clip_to_point(
        points.iloc[0], radius=5_000, expand_radius=False
    )

    assert "expanded_radius" in result.columns


def test_clip_to_radius_returns_input_when_radius_missing(
    routing_test_inputs, cost_metadata
):
    """_clip_to_radius returns the input features when radius is missing"""

    mapper = PointToFeatureMapper(
        cost_metadata["crs"], routing_test_inputs["features_fp"]
    )
    points = make_rev_sc_points(
        cost_metadata["shape"][0],
        cost_metadata["shape"][1],
        cost_metadata["crs"],
        cost_metadata["transform"],
        resolution=_determine_sparse_resolution(cost_metadata["shape"]),
    )
    features = (
        routing_test_inputs["features"].to_crs(cost_metadata["crs"]).head(1)
    )

    result = mapper._clip_to_radius(points.iloc[0], None, features, True)

    assert result is features


def test_clip_to_point_without_radius_uses_region_only(
    routing_test_inputs, cost_metadata, monkeypatch
):
    """_clip_to_point should bypass radius clipping when radius missing"""

    mapper = PointToFeatureMapper(
        cost_metadata["crs"],
        routing_test_inputs["features_fp"],
        regions=routing_test_inputs["regions"],
    )

    region_features = (
        routing_test_inputs["features"].to_crs(cost_metadata["crs"]).head(1)
    )

    def _fake_clip_region(self, point):
        return region_features.copy()

    called = {"radius": False}

    def _fake_clip_radius(self, point, radius, features, expand_radius):
        called["radius"] = True
        return features

    monkeypatch.setattr(
        PointToFeatureMapper, "_clip_to_region", _fake_clip_region
    )
    monkeypatch.setattr(
        PointToFeatureMapper, "_clip_to_radius", _fake_clip_radius
    )

    points = make_rev_sc_points(
        cost_metadata["shape"][0],
        cost_metadata["shape"][1],
        cost_metadata["crs"],
        cost_metadata["transform"],
        resolution=_determine_sparse_resolution(cost_metadata["shape"]),
    )

    result = mapper._clip_to_point(points.iloc[0], radius=None)

    assert not called["radius"]
    assert isinstance(result, gpd.GeoDataFrame)


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
