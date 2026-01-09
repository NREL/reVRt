"""reVRt finalize CLI command unit tests"""

import os
import platform
from pathlib import Path

import pandas as pd
import pytest
import geopandas as gpd
from shapely.geometry import LineString

from revrt.routing.cli.finalize import finalize_routes
from revrt.exceptions import revrtFileNotFoundError


def test_merge_routes_no_files(tmp_path):
    """finalize_routes should raise when no files match collect pattern"""
    with pytest.raises(
        revrtFileNotFoundError, match="No files found using collect pattern:"
    ):
        finalize_routes(
            collect_pattern="dne*.csv",
            project_dir=tmp_path,
            out_dir=tmp_path,
            job_name="test",
        )


@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_cli_finalize_routes_merges_csv(
    run_gaps_cli_with_expected_file, tmp_path
):
    """finalize-routes CLI should merge CSV chunk outputs"""

    chunk_dir = tmp_path / "outputs"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    chunk_frames = [
        pd.DataFrame(
            {
                "route_id": ["chunk0_route0", "chunk0_route1"],
                "start_row": [0, 1],
                "start_col": [0, 1],
                "end_row": [2, 3],
                "end_col": [2, 3],
                "cost": [10.0, 20.0],
            }
        ),
        pd.DataFrame(
            {
                "route_id": ["chunk1_route0", "chunk1_route1"],
                "start_row": [4, 5],
                "start_col": [4, 5],
                "end_row": [6, 7],
                "end_col": [6, 7],
                "cost": [30.0, 40.0],
            }
        ),
    ]

    for idx, frame in enumerate(chunk_frames):
        frame.to_csv(chunk_dir / f"routes_part_{idx}.csv", index=False)

    chunk_fp = chunk_dir / "routes_part_999.csv"
    pd.DataFrame(columns=["route_id"]).to_csv(chunk_fp, index=False)

    config = {
        "collect_pattern": "outputs/routes_part_*.csv",
        "chunk_size": 1,
    }

    merged_fp = run_gaps_cli_with_expected_file(
        "finalize-routes", config, tmp_path
    )

    merged_df = pd.read_csv(merged_fp)
    expected_df = pd.concat(chunk_frames, ignore_index=True)
    pd.testing.assert_frame_equal(
        merged_df.sort_values("route_id").reset_index(drop=True),
        expected_df.sort_values("route_id").reset_index(drop=True),
    )

    chunk_files_dir = tmp_path / "chunk_files"
    assert chunk_files_dir.exists()

    for idx in range(len(chunk_frames)):
        original_fp = chunk_dir / f"routes_part_{idx}.csv"
        relocated_fp = chunk_files_dir / f"routes_part_{idx}.csv"
        assert not original_fp.exists()
        assert relocated_fp.exists()

    original_fp = chunk_dir / "routes_part_999.csv"
    relocated_fp = chunk_files_dir / "routes_part_999.csv"
    assert not original_fp.exists()
    assert relocated_fp.exists()


@pytest.mark.parametrize("tol", [None, 0.01])
@pytest.mark.skipif(
    (os.environ.get("TOX_RUNNING") == "True")
    and (platform.system() == "Windows"),
    reason="CLI does not work under tox env on windows",
)
def test_cli_finalize_routes_merges_gpkg(
    run_gaps_cli_with_expected_file, tmp_path, tol
):
    """finalize-routes CLI should merge GeoPackage chunk outputs"""

    chunk_dir = tmp_path / "geoms"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    chunk_geometries = []
    for idx in range(2):
        gdf = gpd.GeoDataFrame(
            {
                "route_id": [
                    f"chunk{idx}_route0",
                    f"chunk{idx}_route1",
                ],
                "start_row": [idx, idx + 10],
                "start_col": [idx, idx + 10],
                "end_row": [idx + 20, idx + 30],
                "end_col": [idx + 20, idx + 30],
            },
            geometry=[
                LineString(
                    [
                        (idx, idx),
                        (idx + 0.05, idx + 0.02),
                        (idx + 1.0, idx + 0.8),
                    ]
                ),
                LineString(
                    [
                        (idx + 1.0, idx),
                        (idx + 1.2, idx + 0.3),
                        (idx + 2.0, idx + 1.0),
                    ]
                ),
            ],
            crs="EPSG:4326",
        )

        chunk_fp = chunk_dir / f"segment_{idx}.gpkg"
        gdf.to_file(chunk_fp, driver="GPKG")
        chunk_geometries.append(gdf)

    chunk_fp = chunk_dir / "segment_999.gpkg"
    gpd.GeoDataFrame(columns=["route_id"]).to_file(chunk_fp, driver="GPKG")

    config = {
        "collect_pattern": "geoms/segment_*.gpkg",
        "chunk_size": 1,
        "simplify_geo_tolerance": tol,
        "out_fp": str(tmp_path / "merged_routes.gpkg"),
        "purge_chunks": True,
    }

    merged_fp = run_gaps_cli_with_expected_file(
        "finalize-routes", config, tmp_path
    )

    merged_gdf = gpd.read_file(merged_fp)
    expected_count = sum(len(gdf) for gdf in chunk_geometries)
    assert len(merged_gdf) == expected_count
    assert set(merged_gdf["route_id"]) == {
        "chunk0_route0",
        "chunk0_route1",
        "chunk1_route0",
        "chunk1_route1",
    }
    assert all(geom.geom_type == "LineString" for geom in merged_gdf.geometry)

    for idx in range(len(chunk_geometries)):
        assert not (chunk_dir / f"segment_{idx}.gpkg").exists()
    assert not (chunk_dir / "segment_999.gpkg").exists()
    assert not (chunk_dir / "chunk_files").exists()


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
