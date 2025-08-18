//! Dataset samples for tests and demonstrations

use ndarray::Array2;
use rand::Rng;
use zarrs::storage::ReadableWritableListableStorage;

/// Create a zarr store with a few sample layers
///
/// Just a proof of concept with lots of hardcoded values
/// that must be improved.
pub(crate) fn multi_variable_zarr() -> std::path::PathBuf {
    let ni = 8;
    let nj = 8;
    let ci = 4;
    let cj = 4;

    let tmp_path = tempfile::TempDir::new().unwrap();

    let store: ReadableWritableListableStorage = std::sync::Arc::new(
        zarrs::filesystem::FilesystemStore::new(tmp_path.path())
            .expect("could not open filesystem store"),
    );

    zarrs::group::GroupBuilder::new()
        .build(store.clone(), "/")
        .unwrap()
        .store_metadata()
        .unwrap();

    // Create an array
    // Remember to remove /cost
    for array_path in ["/A", "/B", "/C", "/cost"] {
        let array = zarrs::array::ArrayBuilder::new(
            vec![ni, nj], // array shape
            zarrs::array::DataType::Float32,
            vec![ci, cj].try_into().unwrap(), // regular chunk shape
            zarrs::array::FillValue::from(zarrs::array::ZARR_NAN_F32),
        )
        // .bytes_to_bytes_codecs(vec![]) // uncompressed
        .dimension_names(["y", "x"].into())
        // .storage_transformers(vec![].into())
        .build(store.clone(), array_path)
        .unwrap();

        // Write array metadata to store
        array.store_metadata().unwrap();

        let mut rng = rand::rng();
        let mut a = vec![];
        for _x in 0..(ni * nj) {
            a.push(rng.random_range(0.0..=1.0));
        }
        let data: Array2<f32> =
            ndarray::Array::from_shape_vec((ni.try_into().unwrap(), nj.try_into().unwrap()), a)
                .unwrap();

        array
            .store_chunks_ndarray(
                &zarrs::array_subset::ArraySubset::new_with_ranges(&[0..(ni / ci), 0..(nj / cj)]),
                data,
            )
            .unwrap();
    }

    tmp_path.keep()
}

/// Create a zarr store with a cost layer comprised of a single value
pub(crate) fn constant_value_cost_zarr(cost_value: f32) -> std::path::PathBuf {
    let (ni, nj) = (8, 8);
    let (ci, cj) = (4, 4);

    let tmp_path = tempfile::TempDir::new().unwrap();

    let store: zarrs::storage::ReadableWritableListableStorage = std::sync::Arc::new(
        zarrs::filesystem::FilesystemStore::new(tmp_path.path())
            .expect("could not open filesystem store"),
    );

    zarrs::group::GroupBuilder::new()
        .build(store.clone(), "/")
        .unwrap()
        .store_metadata()
        .unwrap();

    let array = zarrs::array::ArrayBuilder::new(
        vec![ni, nj], // array shape
        zarrs::array::DataType::Float32,
        vec![ci, cj].try_into().unwrap(), // regular chunk shape
        zarrs::array::FillValue::from(zarrs::array::ZARR_NAN_F32),
    )
    .dimension_names(["y", "x"].into())
    .build(store.clone(), "/cost")
    .unwrap();

    // Write array metadata to store
    array.store_metadata().unwrap();

    let (uni, unj): (usize, usize) = (ni.try_into().unwrap(), nj.try_into().unwrap());
    let data: Array2<f32> =
        ndarray::Array::from_shape_vec((uni, unj), vec![cost_value; uni * unj]).unwrap();

    array
        .store_chunks_ndarray(
            &zarrs::array_subset::ArraySubset::new_with_ranges(&[0..(ni / ci), 0..(nj / cj)]),
            data,
        )
        .unwrap();

    tmp_path.keep()
}

/// Create a zarr store with a cost layer comprised of cell indices
pub(crate) fn cost_as_index_zarr((ni, nj): (u64, u64), (ci, cj): (u64, u64)) -> std::path::PathBuf {
    let tmp_path = tempfile::TempDir::new().unwrap();

    let store: zarrs::storage::ReadableWritableListableStorage = std::sync::Arc::new(
        zarrs::filesystem::FilesystemStore::new(tmp_path.path())
            .expect("could not open filesystem store"),
    );

    zarrs::group::GroupBuilder::new()
        .build(store.clone(), "/")
        .unwrap()
        .store_metadata()
        .unwrap();

    let array = zarrs::array::ArrayBuilder::new(
        vec![ni, nj], // array shape
        zarrs::array::DataType::Float32,
        vec![ci, cj].try_into().unwrap(), // regular chunk shape
        zarrs::array::FillValue::from(zarrs::array::ZARR_NAN_F32),
    )
    .dimension_names(["y", "x"].into())
    .build(store.clone(), "/cost")
    .unwrap();

    // Write array metadata to store
    array.store_metadata().unwrap();

    let a: Vec<f32> = (0..ni * nj).map(|x| x as f32).collect();
    let data: Array2<f32> =
        ndarray::Array::from_shape_vec((ni.try_into().unwrap(), nj.try_into().unwrap()), a)
            .unwrap();

    array
        .store_chunks_ndarray(
            &zarrs::array_subset::ArraySubset::new_with_ranges(&[0..(ni / ci), 0..(nj / cj)]),
            data,
        )
        .unwrap();

    tmp_path.keep()
}
