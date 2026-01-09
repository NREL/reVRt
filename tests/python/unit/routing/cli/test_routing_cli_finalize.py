"""reVRt finalize CLI command unit tests"""

import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import geopandas as gpd
from shapely.geometry import LineString
from rasterio.transform import from_origin

from revrt.routing.cli.collect import finalize_routes
from revrt.utilities import LayeredFile
from revrt.exceptions import revrtFileNotFoundError


@pytest.fixture(scope="module")
def sample_layered_data(tmp_path_factory):
    """Create layered routing data mimicking point_to_point tests"""

    data_dir = tmp_path_factory.mktemp("routing_cli_data")

    layered_fp = data_dir / "test_layered.zarr"
    layer_file = LayeredFile(layered_fp)

    height, width = (7, 8)
    cell_size = 1.0
    x0, y0 = 0.0, float(height)
    transform = from_origin(x0, y0, cell_size, cell_size)
    x_coords = (
        x0 + np.arange(width, dtype=np.float32) * cell_size + cell_size / 2
    )
    y_coords = (
        y0 - np.arange(height, dtype=np.float32) * cell_size - cell_size / 2
    )

    layer_values = [
        np.array(
            [
                [
                    [7, 7, 8, 0, 9, 9, 9, 0],
                    [8, 1, 2, 2, 9, 9, 9, 0],
                    [9, 1, 3, 3, 9, 1, 2, 3],
                    [9, 1, 2, 1, 9, 1, 9, 0],
                    [9, 9, 9, 1, 9, 1, 9, 0],
                    [9, 9, 9, 1, 1, 1, 9, 0],
                    [9, 9, 9, 9, 9, 9, 9, 0],
                ]
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [
                    [8, 7, 6, 5, 5, 6, 7, 9],
                    [7, 1, 1, 2, 3, 3, 2, 8],
                    [6, 2, 9, 6, 5, 2, 1, 7],
                    [7, 3, 8, 1, 2, 3, 2, 6],
                    [8, 4, 7, 2, 8, 4, 3, 5],
                    [9, 5, 6, 3, 4, 4, 3, 4],
                    [9, 6, 7, 4, 5, 5, 4, 3],
                ]
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [
                    [6, 6, 6, 6, 6, 7, 8, 9],
                    [5, 2, 2, 3, 4, 5, 6, 8],
                    [4, 3, 7, 7, 6, 4, 5, 7],
                    [5, 4, 6, 2, 3, 4, 4, 6],
                    [6, 5, 5, 3, 7, 5, 5, 5],
                    [7, 6, 6, 4, 5, 5, 4, 4],
                    [8, 7, 7, 5, 6, 5, 4, 3],
                ]
            ],
            dtype=np.float32,
        ),
    ]

    for ind, routing_layer in enumerate(layer_values, start=1):
        da = xr.DataArray(
            routing_layer,
            dims=("band", "y", "x"),
            coords={"y": y_coords, "x": x_coords},
        )
        da = da.rio.write_crs("EPSG:4326")
        da = da.rio.write_transform(transform)

        geotiff_fp = data_dir / f"layer_{ind}.tif"
        da.rio.to_raster(geotiff_fp, driver="GTiff")

        layer_file.write_geotiff_to_file(
            geotiff_fp, f"layer_{ind}", overwrite=True
        )

    return layered_fp


def _build_route_table(layered_fp, rows_cols):
    """Helper to construct route tables with CRS-aligned coordinates"""

    with xr.open_dataset(layered_fp, consolidated=False, engine="zarr") as ds:
        latitudes = ds["latitude"].to_numpy()
        longitudes = ds["longitude"].to_numpy()

    records = []
    for idx, (start, end) in enumerate(rows_cols):
        s_row, s_col = start
        e_row, e_col = end
        records.append(
            {
                "route_id": f"route_{idx}",
                "start_lat": float(latitudes[s_row, s_col]),
                "start_lon": float(longitudes[s_row, s_col]),
                "end_lat": float(latitudes[e_row, e_col]),
                "end_lon": float(longitudes[e_row, e_col]),
                "voltage": 138,
                "polarity": "ac",
            }
        )

    return pd.DataFrame.from_records(records)


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
