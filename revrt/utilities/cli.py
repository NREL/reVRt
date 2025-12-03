"""revrt utilities command line interface (CLI)"""

import logging
from pathlib import Path

import rasterio
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from gaps.cli import CLICommandFromClass, CLICommandFromFunction

from revrt.utilities.handlers import LayeredFile


logger = logging.getLogger(__name__)


def layers_from_file(fp, _out_layer_dir, layers=None, profile_kwargs=None):
    """Extract layers from a layered file on disk

    Parameters
    ----------
    fp : path-like
        Path to layered file on disk.
    layers : list, optional
        List of layer names to extract. Layer names must match layers in
        the `fp`, otherwise an error will be raised. If ``None``,
        extracts all layers from the
        :class:`~revrt.utilities.handlers.LayeredFile`.
        By default, ``None``.
    profile_kwargs : dict, optional
        Additional keyword arguments to pass into writing each raster.
        The following attributes ar ignored (they are set using
        properties of the source
        :class:`~revrt.utilities.handlers.LayeredFile`):

                - nodata
                - transform
                - crs
                - count
                - width
                - height

        By default, ``None``.

    Returns
    -------
    list
        List of paths to the GeoTIFF files that were created.
    """
    # TODO: Add dask client here??
    out_layer_dir = Path(_out_layer_dir)
    out_layer_dir.mkdir(parents=True, exist_ok=True)

    profile_kwargs = profile_kwargs or {}

    if layers is not None:
        layers = {layer: out_layer_dir / f"{layer}.tif" for layer in layers}
        LayeredFile(fp).extract_layers(layers, **profile_kwargs)
    else:
        layers = LayeredFile(fp).extract_all_layers(
            out_layer_dir, **profile_kwargs
        )

    return [str(layer_fp) for layer_fp in layers.values()]


def _preprocess_layers_from_file_config(config, out_dir, out_layer_dir=None):
    """Preprocess user config

    Parameters
    ----------
    config : dict
        User configuration parsed as (nested) dict.
    out_dir : path-like
        Output directory as suggested by GAPs (typically the config
        directory).
    out_layer_dir : path-like, optional
        Path to output directory into which layers should be saved as
        GeoTIFFs. This directory will be created if it does not already
        exist. If not provided, will use the config directory as output.
        By default, ``None``.
    """
    config["_out_layer_dir"] = str(out_layer_dir or out_dir)
    return config


def convert_pois_to_lines(poi_csv_f, template_f, out_f):
    """Convert POIs in CSV to lines and save in a GPKG as substations

    This functions also creates a fake transmission line to connect to
    the substations to satisfy LCP code requirements.

    Parameters
    ----------
    poi_csv_f : path-like
        Path to CSV file with POIs in it.
    template_f : path-like
        Path to template raster with CRS to use for GeoPackage.
    out_f : path-like
        Path and file name for GeoPackage.
    """
    logger.debug("Converting POIs in %s to lines in %s", poi_csv_f, out_f)
    with rasterio.open(template_f) as tif:
        crs = tif.crs

    df = pd.read_csv(poi_csv_f)[
        ["POI Name", "State", "Voltage (kV)", "Lat", "Long"]
    ]

    pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Long, df.Lat))
    pts = pts.set_crs("EPSG:4326")
    pts = pts.to_crs(crs)

    # Convert points to short lines
    new_geom = []
    for pt in pts.geometry:
        end = Point(pt.x + 50, pt.y + 50)
        line = LineString([pt, end])
        new_geom.append(line)
    lines = pts.set_geometry(new_geom, crs=crs)

    # Append some fake values to make the LCP code happy
    lines["ac_cap"] = 9999999
    lines["category"] = "Substation"
    lines["voltage"] = 500  # kV
    lines["trans_gids"] = "[9999]"

    # add a fake trans line for the subs to connect to
    trans_line = pd.DataFrame(
        {
            "POI Name": "fake",
            "ac_cap": 9999999,
            "category": "TransLine",
            "voltage": 500,  # kV
            "trans_gids": None,
        },
        index=[9999],
    )

    trans_line = gpd.GeoDataFrame(trans_line)
    geo = LineString([Point(0, 0), Point(100000, 100000)])
    trans_line = trans_line.set_geometry([geo], crs=crs)

    pois = pd.concat([lines, trans_line])
    pois["gid"] = pois.index

    pois.to_file(out_f, driver="GPKG")
    logger.debug("Complete")


layers_to_file_command = CLICommandFromClass(
    LayeredFile, method="layers_to_file", add_collect=False
)
layers_from_file_command = CLICommandFromFunction(
    function=layers_from_file,
    add_collect=False,
    config_preprocessor=_preprocess_layers_from_file_config,
)
convert_pois_to_lines_command = CLICommandFromFunction(
    convert_pois_to_lines,
    name="convert-pois-to-lines",
    add_collect=False,
    split_keys=None,
)
