"""revrt utilities command line interface (CLI)"""

from pathlib import Path

from gaps.cli import CLICommandFromClass, CLICommandFromFunction

from revrt.utilities.handlers import LayeredFile


def layers_from_file(fp, _out_layer_dir, layers=None, profile_kwargs=None):
    """Extract layers from a layered file on disk

    Parameters
    ----------
    fp : path-like
        Path to layered file on disk.
    layers : list, optional
        List of layer names to extract. Layer names must match layers in
        the `fp`, otherwise an error will be raised. If ``None``,
        extracts all layers from the :class:`LayeredFile`.
        By default, ``None``.
    profile_kwargs : dict, optional
        Additional keyword arguments to pass into writing each raster.
        The following attributes ar ignored (they are set using
        properties of the source :class:`LayeredFile`):

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


layers_to_file_command = CLICommandFromClass(
    LayeredFile, method="layers_to_file", add_collect=False
)
layers_from_file_command = CLICommandFromFunction(
    function=layers_from_file,
    add_collect=False,
    config_preprocessor=_preprocess_layers_from_file_config,
)
