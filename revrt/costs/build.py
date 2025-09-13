"""Code to build cost layer file"""

import logging
from warnings import warn

from pydantic import ValidationError
from gaps.config import load_config

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


def build_from_config(config_fpath):
    """Create costs, barriers, and frictions from a config file"""
    config = _validated_config(config_fpath)

    output_tiff_dir = config.output_tiff_dir.expanduser().resolve()
    output_tiff_dir.mkdir(exist_ok=True, parents=True)

    lf_handler = LayeredFile(fp=config.fp)
    if not lf_handler.fp.exists():
        logger.info(
            "%s not found. Creating new layered file...", lf_handler.fp
        )
        lf_handler.create_new(template_file=config.template_raster_fpath)

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


def _validated_config(config_fpath):
    """Validate use config inputs"""
    config_dict = load_config(config_fpath)
    try:
        config = TransmissionLayerCreationConfig.model_validate(config_dict)
    except ValidationError as exc:
        msg = f"Error loading config file {config_fpath}"
        raise revrtConfigurationError(msg) from exc

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
