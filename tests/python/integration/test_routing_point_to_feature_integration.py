"""Integration tests for point to feature routing"""

import math
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
import xarray as xr

from revrt.exceptions import revrtValueError
from revrt.routing.cli_point_to_features import (
    build_point_to_feature_route_table_command,
    point_to_feature_route_table,
    _make_points,
)
from revrt.routing.utilities import (
    PointToFeatureMapper,
    make_rev_sc_points,
    map_to_costs,
)
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


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
