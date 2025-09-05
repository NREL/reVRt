"""Test reVRt cost layer building"""

from pathlib import Path
from itertools import chain, combinations

import pytest
import numpy as np
import xarray as xr
from rasterio.transform import Affine, from_origin

from revrt.models.cost_layers import LayerBuildConfig, RangeConfig, Rasterize
from revrt.costs.layer_creator import LayerCreator
from revrt.costs.masks import Masks
from revrt.utilities import LayeredFile
from revrt.exceptions import revrtAttributeError, revrtValueError
from revrt.warn import revrtWarning


_MUTEX_CONFIG_PARAMS = [
    {"global_value": 1},
    {"map": {1: 2}},
    {"bins": [RangeConfig(min=0, max=10, value=5)]},
    {"pass_through": True},
]


@pytest.fixture(scope="module")
def tiff_layers_for_testing(tmp_path_factory):
    """Test TIFF layers for testing LayerCreator"""
    layer_dir = tmp_path_factory.mktemp("data")

    width = height = 3
    cell_size = 1
    x0 = y0 = 0
    transform = from_origin(x0, y0, cell_size, cell_size)

    layers = {
        "friction_1.tif": np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]),
        "fi_1.tif": np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]),
        "fi_2.tif": np.array([[0, 0, 0], [0, 0, 0], [2, 2, 2]]),
    }

    for layer_fn, data in layers.items():
        da = xr.DataArray(
            data,
            dims=("y", "x"),
            coords={
                "x": x0 + np.arange(width) * cell_size + cell_size / 2,
                "y": y0 - np.arange(height) * cell_size - cell_size / 2,
            },
            name="test_band",
        )

        da = da.rio.write_crs("EPSG:4326")
        da.rio.write_transform(transform)
        da.rio.to_raster(layer_dir / layer_fn, driver="GTiff")

    return layer_dir, layers


@pytest.fixture(scope="module")
def lf_instance(tmp_path_factory, tiff_layers_for_testing):
    """LayeredFile instance for testing"""
    test_fp = tmp_path_factory.mktemp("zarr_data") / "test.zarr"
    lf = LayeredFile(test_fp)
    layer_dir, layers = tiff_layers_for_testing
    layer_fn = next(iter(layers))
    lf.create_new(layer_dir / layer_fn)
    return lf


@pytest.fixture(scope="module")
def mask_instance():
    """Mask instance for testing"""
    masks = Masks(
        shape=(3, 3),
        crs="ESRI:102008",
        transform=Affine(5.0, 0.0, -12.5, 0.0, -5.0, 12.5),
    )
    masks._dry_mask = np.array(
        [
            [False, False, True],
            [False, False, True],
            [False, False, True],
        ]
    )

    masks._wet_mask = np.array(
        [
            [True, False, False],
            [True, False, False],
            [True, False, False],
        ]
    )

    masks._landfall_mask = np.array(
        [
            [False, True, False],
            [False, True, False],
            [False, True, False],
        ]
    )
    return masks


@pytest.fixture(scope="module")
def builder_instance(
    tmp_path_factory, lf_instance, mask_instance, tiff_layers_for_testing
):
    """LayerCreator instance for testing"""
    layer_dir, _ = tiff_layers_for_testing
    out_dir = tmp_path_factory.mktemp("out_data")
    return LayerCreator(
        lf_instance,
        mask_instance,
        input_layer_dir=layer_dir,
        output_tiff_dir=out_dir,
    )


def test_forced_inclusion(lf_instance, builder_instance):
    """Test forced inclusions"""
    config = {
        "fi_1.tif": LayerBuildConfig(
            extent="wet+",
            forced_inclusion=True,
        ),
        "friction_1.tif": LayerBuildConfig(
            extent="all", map={1: 1, 2: 2, 3: 3}
        ),
        "fi_2.tif": LayerBuildConfig(
            extent="dry+",
            forced_inclusion=True,
        ),
    }
    builder_instance.build("friction", config, write_to_file=True)

    with xr.open_dataset(
        lf_instance.fp, consolidated=False, engine="zarr"
    ) as ds:
        assert np.allclose(
            np.array([[[1, 1, 1], [0, 0, 2], [3, 0, 0]]]), ds["friction"]
        )


def test_bins(builder_instance):
    """Test bins key in LayerBuildConfig"""
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    config = LayerBuildConfig(
        extent="wet+", bins=[RangeConfig(min=1, max=5, value=4)]
    )
    result = builder_instance._process_raster_data(data, config)
    assert (result == np.array([[4, 4, 0], [4, 0, 0], [0, 0, 0]])).all()

    config = LayerBuildConfig(
        extent="dry+", bins=[RangeConfig(min=5, max=9, value=5)]
    )
    result = builder_instance._process_raster_data(data, config)
    assert (result == np.array([[0, 0, 0], [0, 5, 5], [0, 5, 0]])).all()

    config = LayerBuildConfig(
        extent="all", bins=[RangeConfig(min=2, max=9, value=5)]
    )
    result = builder_instance._process_raster_data(data, config)
    assert (result == np.array([[0, 5, 5], [5, 5, 5], [5, 5, 0]])).all()


def test_complex_bins(builder_instance):
    """Test bins key with multiple bins in LayerBuildConfig"""
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    config = LayerBuildConfig(
        extent="wet+",
        bins=[
            RangeConfig(min=1, max=5, value=4),
            RangeConfig(min=4, max=10, value=10),
        ],
    )
    result = builder_instance._process_raster_data(data, config)
    assert (result == np.array([[4, 4, 0], [14, 10, 0], [10, 10, 0]])).all()

    config = LayerBuildConfig(
        extent="all",
        bins=[
            RangeConfig(min=1, max=6, value=5),
            RangeConfig(min=4, max=9, value=1),
        ],
    )
    result = builder_instance._process_raster_data(data, config)
    assert (result == np.array([[5, 5, 5], [6, 6, 1], [1, 1, 0]])).all()


def test_map(builder_instance):
    """Test map key in LayerBuildConfig"""
    data = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    config = LayerBuildConfig(
        extent="wet",
        map={1: 5, 3: 9},
    )
    result = builder_instance._process_raster_data(data, config)
    assert (result == np.array([[5, 0, 0], [0, 0, 0], [9, 0, 0]])).all()

    data = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    config = LayerBuildConfig(
        extent="landfall",
        map={1: 5, 3: 9},
    )
    result = builder_instance._process_raster_data(data, config)
    assert (result == np.array([[0, 5, 0], [0, 0, 0], [0, 9, 0]])).all()


def test_pass_through(builder_instance):
    """Test pass through key in LayerBuildConfig"""
    data = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    config = LayerBuildConfig(extent="wet", pass_through=True)
    result = builder_instance._process_raster_data(data, config)
    assert (result == np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0]])).all()

    config = LayerBuildConfig(extent="landfall", pass_through=True)
    result = builder_instance._process_raster_data(data, config)
    assert (result == np.array([[0, 1, 0], [0, 2, 0], [0, 3, 0]])).all()

    config = LayerBuildConfig(extent="all", pass_through=True)
    result = builder_instance._process_raster_data(data, config)
    assert (result == data).all()


def test_bin_config_sanity_checking(builder_instance, tiff_layers_for_testing):
    """Test cost binning config sanity checking"""
    __, layers = tiff_layers_for_testing
    layer_fn = next(iter(layers))

    reverse_bins = [RangeConfig(min=10, max=0, value=1)]
    config = LayerBuildConfig(extent="all", bins=reverse_bins)
    with pytest.raises(
        revrtAttributeError, match="Min is greater than max for bin config"
    ):
        builder_instance._process_raster_layer(layer_fn, config)

    bin_config = [
        RangeConfig(min=1, max=2, value=3),
        RangeConfig(min=2, max=5, value=4),
    ]
    good_config = LayerBuildConfig(extent="all", bins=bin_config)
    builder_instance._process_raster_layer(layer_fn, good_config)


def test_bin_config_warns_full_range(
    builder_instance, tiff_layers_for_testing
):
    """Test cost binning config warns for full range"""
    __, layers = tiff_layers_for_testing
    layer_fn = next(iter(layers))

    reverse_bins = [RangeConfig(value=1)]
    config = LayerBuildConfig(extent="all", bins=reverse_bins)
    with pytest.warns(
        revrtWarning,
        match=(
            "Bin covers all possible values, did you forget to set min or max?"
        ),
    ):
        builder_instance._process_raster_layer(layer_fn, config)


def test_tiff_config_rasterize(builder_instance, tiff_layers_for_testing):
    """Test error if including `rasterize` for TIFF"""
    __, layers = tiff_layers_for_testing
    layer_fn = next(iter(layers))

    config = LayerBuildConfig(rasterize=Rasterize(value=1))
    with pytest.raises(
        revrtValueError,
        match=f"'rasterize' is only for vectors. Found in {layer_fn!r} config",
    ):
        builder_instance._process_raster_layer(layer_fn, config)


@pytest.mark.parametrize(
    "mutex_params",
    chain.from_iterable(
        combinations(_MUTEX_CONFIG_PARAMS, r)
        for r in range(2, len(_MUTEX_CONFIG_PARAMS) + 1)
    ),
)
def test_tiff_config_mutually_exclusive_entries(
    builder_instance, mutex_params, tiff_layers_for_testing
):
    """Test error for mutually exclusive entries in LayerBuildConfig"""
    __, layers = tiff_layers_for_testing
    layer_fn = next(iter(layers))

    config_params = {k: v for param in mutex_params for k, v in param.items()}
    config = LayerBuildConfig(**config_params)
    with pytest.raises(
        revrtValueError,
        match=f"Keys .* are mutually exclusive .* {layer_fn!r} raster config",
    ):
        builder_instance._process_raster_layer(layer_fn, config)


def test_tiff_config_missing_entries(
    builder_instance, tiff_layers_for_testing
):
    """Test error for missing required entries in LayerBuildConfig"""
    __, layers = tiff_layers_for_testing
    layer_fn = next(iter(layers))

    config = LayerBuildConfig()
    with pytest.raises(
        revrtValueError,
        match=(
            f"Either .* must be specified for a raster, .* {layer_fn!r} config"
        ),
    ):
        builder_instance._process_raster_layer(layer_fn, config)


def test_cost_binning_results(builder_instance):
    """Test results of creating cost raster using bins"""
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    bins = [
        RangeConfig(max=2, value=1),
        RangeConfig(min=2, max=4, value=2),
        RangeConfig(min=4, max=8, value=3),
        RangeConfig(min=8, value=4),
    ]
    config = LayerBuildConfig(extent="all", bins=bins)
    output = builder_instance._process_raster_data(data, config)
    assert (output == np.array([[1, 2, 2], [3, 3, 3], [3, 4, 4]])).all()

    bins = [
        RangeConfig(min=2, max=4, value=2),
        RangeConfig(min=4, max=8, value=3),
    ]
    config = LayerBuildConfig(extent="all", bins=bins)
    output = builder_instance._process_raster_data(data, config)
    assert (output == np.array([[0, 2, 2], [3, 3, 3], [3, 0, 0]])).all()

    data = np.array([[-600, -400, -50], [-700, -250, 70], [-500, -150, -70]])
    bins = [
        RangeConfig(max=-500, value=999),
        RangeConfig(min=-500, max=-300, value=666),
        RangeConfig(min=-300, max=-100, value=333),
        RangeConfig(min=-100, value=111),
    ]
    config = LayerBuildConfig(extent="all", bins=bins)
    output = builder_instance._process_raster_data(data, config)
    assert (
        output == np.array([[999, 666, 111], [999, 333, 111], [666, 333, 111]])
    ).all()


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
