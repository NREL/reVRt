"""Integration tests for point to feature routing"""

import json
import math
import warnings
import traceback
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import LineString, Point, Polygon

from revrt._cli import main
from revrt.exceptions import revrtValueError
from revrt.routing.cli_point_to_features import (
    build_point_to_feature_route_table_command,
    point_to_feature_route_table,
    _check_output_filepaths,
    _make_points,
)
from revrt.routing.utilities import (
    PointToFeatureMapper,
    make_rev_sc_points,
    map_to_costs,
)
from revrt.routing import utilities as routing_utils
from revrt.warn import revrtWarning


@pytest.fixture(scope="module")
def routing_test_inputs(tmp_path_factory, test_data_dir):
    """Prepare reusable inputs for mapping tests"""

    work_dir = tmp_path_factory.mktemp("routing_point_to_feature")
    features_src = Path(test_data_dir / "routing/ri_allconns.gpkg")
    regions_src = Path(test_data_dir / "routing/ri_regions.gpkg")

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


def test_make_points_requires_input(cost_metadata):
    """Verify the helper enforces mutually exclusive inputs"""

    with pytest.raises(
        revrtValueError,
        match=(
            "Either `points_fpath` or `resolution` must be provided to "
            "create route table!"
        ),
    ):
        _make_points(
            cost_metadata["crs"],
            cost_metadata["transform"],
            cost_metadata["shape"],
        )


def test_make_points_from_csv(tmp_path, cost_metadata):
    """Build supply-curve points from a CSV input"""

    height, width = cost_metadata["shape"]
    resolution = _determine_sparse_resolution(cost_metadata["shape"])
    sc_points = make_rev_sc_points(
        height,
        width,
        cost_metadata["crs"],
        cost_metadata["transform"],
        resolution=resolution,
    ).head(5)
    csv_path = tmp_path / "points.csv"
    sc_points.drop(columns="geometry").to_csv(csv_path, index=False)

    points = _make_points(
        cost_metadata["crs"],
        cost_metadata["transform"],
        cost_metadata["shape"],
        points_fpath=csv_path,
    )

    assert {"start_row", "start_col"}.issubset(points.columns)


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


def test_point_to_feature_mapper_clips_features_to_region_boundary(tmp_path):
    """Features are trimmed to region boundary when regions provided"""

    crs = "EPSG:3857"
    region_geom = Polygon([(0, 0), (0, 1_000), (1_000, 1_000), (1_000, 0)])
    regions = gpd.GeoDataFrame({"rid": [7]}, geometry=[region_geom], crs=crs)
    regions_fp = tmp_path / "clip_regions.gpkg"
    regions.to_file(regions_fp, driver="GPKG")

    original = LineString([(-250, 500), (1_250, 500)])
    features = gpd.GeoDataFrame(
        {"category": ["keep"], "gid": [101]}, geometry=[original], crs=crs
    )
    features_fp = tmp_path / "clip_features.gpkg"
    features.to_file(features_fp, driver="GPKG")

    mapper = PointToFeatureMapper(
        crs,
        features_fp,
        regions=regions_fp,
        region_identifier_column="rid",
    )

    points = gpd.GeoDataFrame(
        {"start_row": [0], "start_col": [0]},
        geometry=[Point(500, 500)],
        crs=crs,
    )

    out_fp = tmp_path / "clipped_outputs.gpkg"
    mapper.map_points(
        points,
        out_fp,
        radius=800,
        expand_radius=False,
        batch_size=1,
    )

    clipped = gpd.read_file(out_fp)
    assert len(clipped) == 1
    clipped_geom = clipped.geometry.iloc[0]
    assert clipped_geom.geom_type == "LineString"
    assert clipped_geom.length < original.length
    assert clipped["rid"].tolist() == [7]

    expected = LineString([(0, 500), (1_000, 500)])
    assert clipped_geom.equals_exact(expected, tolerance=1e-6)

    region_difference = clipped_geom.difference(region_geom)
    assert region_difference.is_empty


def test_point_to_feature_mapper_clips_features_to_radius(tmp_path):
    """Features are trimmed to circular radius when radius provided"""

    crs = "EPSG:3857"
    radius = 400.0
    point_geom = Point(0, 0)
    original = LineString([(-1_000, 0), (1_000, 0)])
    features = gpd.GeoDataFrame({"gid": [5]}, geometry=[original], crs=crs)
    features_fp = tmp_path / "radius_features.gpkg"
    features.to_file(features_fp, driver="GPKG")

    mapper = PointToFeatureMapper(crs, features_fp)
    points = gpd.GeoDataFrame(
        {"start_row": [0], "start_col": [0]}, geometry=[point_geom], crs=crs
    )

    out_fp = tmp_path / "radius_outputs.gpkg"
    mapper.map_points(
        points,
        out_fp,
        radius=radius,
        expand_radius=False,
        batch_size=1,
    )

    clipped = gpd.read_file(out_fp)
    assert len(clipped) == 1
    clipped_geom = clipped.geometry.iloc[0]
    expected = original.intersection(point_geom.buffer(radius))
    assert clipped_geom.equals_exact(expected, tolerance=1e-6)
    difference = clipped_geom.difference(point_geom.buffer(radius))
    assert difference.is_empty
    assert clipped_geom.length < original.length


def test_map_to_costs_filters_out_of_bounds(cost_metadata):
    """map_to_costs converts coordinates and drops routes out of domain"""

    base_points = make_rev_sc_points(
        cost_metadata["shape"][0],
        cost_metadata["shape"][1],
        cost_metadata["crs"],
        cost_metadata["transform"],
        resolution=_determine_sparse_resolution(cost_metadata["shape"]),
    )

    start = base_points.iloc[0]
    end = base_points.iloc[len(base_points) // 2]
    routes = pd.DataFrame(
        {
            "start_lat": [start.latitude, 90.0],
            "start_lon": [start.longitude, 0.0],
            "end_lat": [end.latitude, end.latitude],
            "end_lon": [end.longitude, end.longitude],
        }
    )

    mapped = map_to_costs(
        routes,
        cost_metadata["crs"],
        cost_metadata["transform"],
        cost_metadata["shape"],
    )

    assert {"start_row", "start_col", "end_row", "end_col"}.issubset(
        mapped.columns
    )
    assert len(mapped) == 1


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


def test_check_output_filepaths_preserves_valid_extensions(tmp_path):
    """Ensure helper respects valid output extensions without warnings"""

    feature_path = tmp_path / "features.gpkg"
    route_path = tmp_path / "routes.csv"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        checked_feature, checked_route = _check_output_filepaths(
            tmp_path, feature_path.name, route_path.name
        )

    assert checked_feature == feature_path
    assert checked_route == route_path
    assert not caught


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


def test_filter_points_outside_cost_domain_warns_and_drops():
    """filter_points_outside_cost_domain drops out-of-bounds routes"""

    table = pd.DataFrame(
        {
            "start_row": [0, -1],
            "start_col": [0, 15],
            "end_row": [1, 4],
            "end_col": [1, -3],
        }
    )

    with pytest.warns(revrtWarning):
        filtered = routing_utils.filter_points_outside_cost_domain(
            table, (5, 5)
        )

    assert len(filtered) == 1


def test_filter_points_outside_cost_domain_no_warning():
    """filter_points_outside_cost_domain passes rows within bounds"""

    table = pd.DataFrame(
        {
            "start_row": [0, 1],
            "start_col": [0, 1],
            "end_row": [2, 3],
            "end_col": [2, 3],
        }
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", revrtWarning)
        result = routing_utils.filter_points_outside_cost_domain(table, (5, 5))

    assert len(result) == len(table)
    assert not caught


def test_filter_transmission_features_drops_empty_categories(
    routing_test_inputs,
):
    """_filter_transmission_features removes empty category records"""

    features = routing_test_inputs["features"].head(2)
    features = features.reset_index(drop=True).copy()
    features["bgid"] = [1, 2]
    features["egid"] = [3, 4]
    features["cap_left"] = [0.0, 0.0]
    features["gid"] = [11, 12]
    features.loc[0, "category"] = math.nan
    features.loc[1, "category"] = "keep"

    with pytest.warns(revrtWarning):
        cleaned = routing_utils._filter_transmission_features(features)

    assert not any(c in cleaned.columns for c in ["bgid", "egid", "cap_left"])
    assert "trans_gid" in cleaned.columns
    assert cleaned["category"].tolist() == ["keep"]


def test_filter_transmission_features_without_category_column(
    routing_test_inputs,
):
    """_filter_transmission_features tolerates missing category column"""

    features = routing_test_inputs["features"].head(1)
    features = features.drop(columns="category", errors="ignore")

    cleaned = routing_utils._filter_transmission_features(features)

    assert "category" not in cleaned.columns


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
