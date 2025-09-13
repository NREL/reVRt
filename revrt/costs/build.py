"""Code to build cost layer file"""

import logging
from warnings import warn

from dask.distributed import Client

from revrt.models.cost_layers import ALL, TransmissionLayerCreationConfig
from revrt.costs.layer_creator import LayerCreator
from revrt.costs.dry_costs_creator import DryCostsCreator
from revrt.costs.masks import Masks
from revrt.utilities import (
    LayeredFile,
    load_data_using_layer_file_profile,
    save_data_using_layer_file_profile,
)
from revrt.exceptions import revrtAttributeError, revrtConfigurationError
from revrt.warn import revrtWarning


logger = logging.getLogger(__name__)
CONFIG_ACTIONS = ["layers", "dry_costs", "merge_friction_and_barriers"]


def build_costs_file(
    fp,
    template_file=None,
    input_layer_dir=".",
    output_tiff_dir=".",
    masks_dir=".",
    layers=None,
    dry_costs=None,
    merge_friction_and_barriers=None,
    max_workers=1,
    memory_limit_per_worker="auto",
):
    """Create costs, barriers, and frictions from a config file

    You can re-run this function on an existing file to add new layers
    without overwriting existing layers or needing to change your
    original config.

    Parameters
    ----------
    fp : path-like
        Path to GeoTIFF/Zarr file to store cost layers in. If the file
        does not exist, it will be created based on the `template_file`
        input.
    template_file : path-like, optional
        Path to template GeoTIFF (``*.tif`` or ``*.tiff``) or Zarr
        (``*.zarr``) file containing the profile and transform to be
        used for the layered costs file. If ``None``, then the `fp`
        is assumed to exist on disk already. By default, ``None``.
    input_layer_dir : path-like, optional
        Directory to search for input layers in, if not found in
        current directory. By default, ``'.'``.
    output_tiff_dir : path-like, optional
        Directory where cost layers should be saved as GeoTIFF.
        By default, ``"."``.
    masks_dir : path-like, optional
        Directory for storing/finding mask GeoTIFFs (wet, dry, landfall,
        wet+, dry+). By default, ``"."``.
    layers : list of LayerConfig dicts, optional
        Configuration for layers to be built and added to the file.
        At least one of `layers`, `dry_costs`, or
        `merge_friction_and_barriers` must be defined.
        By default, ``None``.
    dry_costs : DryCosts dict, optional
        Configuration for dry cost layers to be built and added to the
        file. At least one of `layers`, `dry_costs`, or
        `merge_friction_and_barriers` must be defined.
        By default, ``None``.
    merge_friction_and_barriers : MergeFrictionBarriers dict, optional
        Configuration for merging friction and barriers and adding to
        the layered costs file. At least one of `layers`, `dry_costs`,
        or `merge_friction_and_barriers` must be defined.
        By default, ``None``
    max_workers : int, optional
        Number of parallel workers to use for file creation. If ``None``
        or >1, processing is performed in parallel using Dask.
        By default, ``1``.
    memory_limit_per_worker : str, float, int, or None, default="auto"
        Sets the memory limit *per worker*. This only applies if
        ``max_workers != 1``. If ``None`` or ``0``, no limit is applied.
        If ``"auto"``, the total system memory is split evenly between
        the workers. If a float, that fraction of the system memory is
        used *per worker*. If a string giving a number  of bytes (like
        "1GiB"), that amount is used *per worker*. If an int, that
        number of bytes is used *per worker*. By default, ``"auto"``
    """
    config = _validated_config(
        fp=fp,
        template_file=template_file or fp,
        input_layer_dir=input_layer_dir,
        output_tiff_dir=output_tiff_dir,
        masks_dir=masks_dir,
        layers=layers,
        dry_costs=dry_costs,
        merge_friction_and_barriers=merge_friction_and_barriers,
    )

    if max_workers != 1:
        __ = Client(
            n_workers=max_workers, memory_limit=memory_limit_per_worker
        )

    lf_handler = LayeredFile(fp=config.fp)
    if not lf_handler.fp.exists():
        logger.info(
            "%s not found. Creating new layered file...", lf_handler.fp
        )
        lf_handler.create_new(template_file=config.template_file)

    masks = _load_masks(config, lf_handler)

    builder = LayerCreator(
        lf_handler,
        masks,
        input_layer_dir=config.input_layer_dir,
        output_tiff_dir=config.output_tiff_dir,
    )
    _build_layers(config, builder, lf_handler)

    if config.dry_costs is not None:
        _build_dry_costs(config, masks, lf_handler)

    if config.merge_friction_and_barriers is not None:
        _combine_friction_and_barriers(config, lf_handler)


def _validated_config(**config_dict):
    """Validate use config inputs"""
    config = TransmissionLayerCreationConfig.model_validate(config_dict)
    if not any(config.model_dump()[key] is not None for key in CONFIG_ACTIONS):
        msg = f"At least one of {CONFIG_ACTIONS!r} must be in the config file"
        raise revrtConfigurationError(msg)

    return config


def _load_masks(config, lf_handler):
    """Load masks based on config file"""
    masks = Masks(
        shape=lf_handler.shape,
        crs=lf_handler.profile["crs"],
        transform=lf_handler.profile["transform"],
        masks_dir=config.masks_dir,
    )
    if not config.layers:
        return masks

    build_configs = [lc.build for lc in config.layers]
    need_masks = any(
        lc.extent != ALL for bc in build_configs for lc in bc.values()
    )
    if need_masks:
        masks.load(lf_handler.fp)

    return masks


def _build_layers(config, builder, lf_handler):
    """Build layers from config file"""
    existing_layers = set(lf_handler.data_layers)

    for lc in config.layers or []:
        if lc.layer_name in existing_layers:
            logger.info(
                "Layer %r already exists in %s! Skipping...",
                lc.layer_name,
                lf_handler.fp,
            )
            continue

        builder.build(
            lc.layer_name,
            lc.build,
            values_are_costs_per_mile=lc.values_are_costs_per_mile,
            write_to_file=lc.include_in_file,
            description=lc.description,
        )


def _build_dry_costs(config, masks, lf_handler):
    """Build dry costs from config file"""
    dc = config.dry_costs

    dry_mask = None
    try:
        dry_mask = masks.dry_mask
    except revrtAttributeError:
        msg = "Dry mask not found! Computing dry costs for full extent!"
        warn(msg, revrtWarning)

    dcc = DryCostsCreator(
        lf_handler,
        input_layer_dir=config.input_layer_dir,
        output_tiff_dir=config.output_tiff_dir,
    )
    cost_configs = None if not dc.cost_configs else str(dc.cost_configs)
    dcc.build(
        iso_region_tiff=dc.iso_region_tiff,
        nlcd_tiff=dc.nlcd_tiff,
        slope_tiff=dc.slope_tiff,
        transmission_config=cost_configs,
        mask=dry_mask,
        default_mults=dc.default_mults,
        extra_tiffs=dc.extra_tiffs,
    )


def _combine_friction_and_barriers(config, io_handler):
    """Combine friction and barriers and save to layered file"""

    logger.info("Loading friction and raw barriers")

    merge_config = config.merge_friction_and_barriers
    friction = load_data_using_layer_file_profile(
        io_handler.fp,
        f"{merge_config.friction_layer}.tif",
        layer_dirs=[config.output_tiff_dir, config.input_layer_dir],
    )
    barriers = load_data_using_layer_file_profile(
        io_handler.fp,
        f"{merge_config.barrier_layer}.tif",
        layer_dirs=[config.output_tiff_dir, config.input_layer_dir],
    )
    combined = friction + barriers * merge_config.barrier_multiplier

    out_fp = config.output_tiff_dir / f"{merge_config.output_layer_name}.tif"
    logger.debug("Saving combined barriers to %s", out_fp)
    save_data_using_layer_file_profile(
        layer_fp=io_handler.fp, data=combined, geotiff=out_fp
    )

    logger.info("Writing combined barriers to H5")
    io_handler.write_layer(combined, merge_config.output_layer_name)
