use std::sync::RwLock;

use tracing::{debug, trace, warn};
use zarrs::array::ArrayChunkCacheExt;
use zarrs::storage::{
    ListableStorageTraits, ReadableListableStorage, ReadableWritableListableStorage,
};

use crate::ArrayIndex;
use crate::cost::CostFunction;
use crate::error::Result;

/// Manages the features datasets and calculated total cost
pub(super) struct Dataset {
    /// A Zarr storages with the features
    source: ReadableListableStorage,
    // Silly way to keep the tmp path alive
    #[allow(dead_code)]
    cost_path: tempfile::TempDir,
    /// Variables used to define cost
    /// Minimalist solution for the cost calculation. In the future
    /// it will be modified to include weights and other types of
    /// relations such as operations between features.
    /// At this point it just allows custom variables names and the
    /// cost is calculated from multiple variables.
    // cost_variables: Vec<String>,
    /// Storage for the calculated cost
    cost: ReadableWritableListableStorage,
    /// Index of cost chunks already calculated
    cost_chunk_idx: RwLock<ndarray::Array2<bool>>,
    /// Custom cost function definition
    cost_function: CostFunction,
    /// Cache for the cost
    cache: zarrs::array::ChunkCacheLruSizeLimit<zarrs::array::ChunkCacheTypeDecoded>,
}

impl Dataset {
    pub(super) fn open<P: AsRef<std::path::Path>>(
        path: P,
        cost_function: CostFunction,
        cache_size: u64,
    ) -> Result<Self> {
        debug!("Opening dataset: {:?}", path.as_ref());
        let filesystem =
            zarrs::filesystem::FilesystemStore::new(path).expect("could not open filesystem store");
        let source = std::sync::Arc::new(filesystem);

        // ==== Create the cost dataset ====
        let tmp_path = tempfile::TempDir::new().unwrap();
        debug!(
            "Initializing a temporary cost dataset at {:?}",
            tmp_path.path()
        );
        let cost: ReadableWritableListableStorage = std::sync::Arc::new(
            zarrs::filesystem::FilesystemStore::new(tmp_path.path())
                .expect("could not open filesystem store"),
        );

        trace!("Creating a new group for the cost dataset");
        zarrs::group::GroupBuilder::new()
            .build(cost.clone(), "/")
            .unwrap()
            .store_metadata()
            .unwrap();

        // -- Temporary solution to specify cost storage --
        // Assume all variables have the same shape and chunk shape.
        // Find the name of the first variable and use it.
        let varname = source.list().unwrap()[0].to_string();
        let varname = varname.split("/").collect::<Vec<_>>()[0];
        let tmp = zarrs::array::Array::open(source.clone(), &format!("/{varname}")).unwrap();
        let cost_shape = tmp.shape();
        let chunk_shape = tmp.chunk_grid().clone();
        // ----

        trace!("Creating an empty cost array");
        let array = zarrs::array::ArrayBuilder::new(
            cost_shape.into(),
            zarrs::array::DataType::Float32,
            chunk_shape,
            zarrs::array::FillValue::from(zarrs::array::ZARR_NAN_F32),
        )
        .build(cost.clone(), "/cost")
        .unwrap();
        trace!("Cost shape: {:?}", array.shape().to_vec());
        trace!("Cost chunk shape: {:?}", array.chunk_grid());
        array.store_metadata().unwrap();

        trace!("Cost dataset contents: {:?}", cost.list().unwrap());

        let cost_chunk_idx = ndarray::Array2::from_elem(
            (
                array.chunk_grid_shape().unwrap()[0] as usize,
                array.chunk_grid_shape().unwrap()[1] as usize,
            ),
            false,
        )
        .into();

        if cache_size < 1_000_000 {
            warn!("Cache size smaller than 1MB");
        }
        trace!("Creating cache with size {}MB", cache_size / 1_000_000);
        let cache = zarrs::array::ChunkCacheLruSizeLimit::new(cache_size);

        trace!("Dataset opened successfully");
        Ok(Self {
            source,
            cost_path: tmp_path,
            cost,
            cost_chunk_idx,
            cost_function,
            cache,
        })
    }

    fn calculate_chunk_cost(&self, i: u64, j: u64) {
        let output = self.cost_function.calculate_chunk(&self.source, i, j);
        trace!("Cost function: {:?}", self.cost_function);

        /*
        trace!("Getting '/A' variable");
        let array = zarrs::array::Array::open(self.source.clone(), "/A").unwrap();
        let value = array.retrieve_chunk_ndarray::<f32>(&[i, j]).unwrap();
        trace!("Value: {:?}", value);
        trace!("Calculating cost for chunk ({}, {})", i, j);
        let output = value * 10.0;
        */

        let cost = zarrs::array::Array::open(self.cost.clone(), "/cost").unwrap();
        cost.store_metadata().unwrap();
        let chunk_indices: Vec<u64> = vec![i, j];
        trace!("Storing chunk at {:?}", chunk_indices);
        let chunk_subset =
            &zarrs::array_subset::ArraySubset::new_with_ranges(&[i..(i + 1), j..(j + 1)]);
        trace!("Target chunk subset: {:?}", chunk_subset);
        cost.store_chunks_ndarray(chunk_subset, output).unwrap();
    }

    pub(super) fn get_3x3(&self, index: &ArrayIndex) -> Vec<(ArrayIndex, f32)> {
        let &ArrayIndex { i, j } = index;

        trace!("Getting 3x3 neighborhood for (i={}, j={})", i, j);

        trace!("Cost dataset contents: {:?}", self.cost.list().unwrap());
        // What is this size?
        trace!("Cost dataset size: {:?}", self.cost.size().unwrap());

        trace!("Opening cost dataset");
        let cost = zarrs::array::Array::open(self.cost.clone(), "/cost").unwrap();
        trace!("Cost dataset with shape: {:?}", cost.shape());

        // Cutting off the edges for now.
        let shape = cost.shape();
        if i == 0 || i >= (shape[0] - 1) || j == 0 || j >= (shape[1] - 1) {
            warn!("I'm not ready to deal with the edges yet");
            return vec![];
        }

        // Capture the 3x3 neighborhood
        let subset = zarrs::array_subset::ArraySubset::new_with_ranges(&[
            (i - 1)..(i + 2),
            (j - 1)..(j + 2),
        ]);
        trace!("Cost subset: {:?}", subset);

        // Find the chunks that intersect the subset
        let chunks = &cost.chunks_in_array_subset(&subset).unwrap().unwrap();
        trace!("Cost chunks: {:?}", chunks);
        trace!(
            "Cost subset extends to {:?} chunks",
            chunks.num_elements_usize()
        );

        for ci in chunks.start()[0]..(chunks.start()[0] + chunks.shape()[0]) {
            for cj in chunks.start()[1]..(chunks.start()[1] + chunks.shape()[1]) {
                trace!(
                    "Checking if cost for chunk ({}, {}) has been calculated",
                    ci, cj
                );
                if self.cost_chunk_idx.read().unwrap()[[ci as usize, cj as usize]] {
                    trace!("Cost for chunk ({}, {}) already calculated", ci, cj);
                } else {
                    self.calculate_chunk_cost(ci, cj);
                    let mut chunk_idx = self.cost_chunk_idx.write().unwrap();
                    chunk_idx[[ci as usize, cj as usize]] = true;
                }
            }
        }

        // Retrieve the 3x3 neighborhood values
        let value: Vec<f32> = cost
            .retrieve_array_subset_elements_opt_cached::<f32, zarrs::array::ChunkCacheTypeDecoded>(
                &self.cache,
                &subset,
                &zarrs::array::codec::CodecOptions::default(),
            )
            .unwrap();

        trace!("Read values {:?}", value);

        let neighbors = vec![
            (ArrayIndex { i: i - 1, j: j - 1 }, value[0]),
            (ArrayIndex { i: i - 1, j }, value[1]),
            (ArrayIndex { i: i - 1, j: j + 1 }, value[2]),
            (ArrayIndex { i, j: j - 1 }, value[3]),
            (ArrayIndex { i, j: j + 1 }, value[5]),
            (ArrayIndex { i: i + 1, j: j - 1 }, value[6]),
            (ArrayIndex { i: i + 1, j }, value[7]),
            (ArrayIndex { i: i + 1, j: j + 1 }, value[8]),
        ];
        trace!("Neighbors {:?}", neighbors);

        neighbors

        /*
        let mut data = array
            .load_chunks_ndarray(&zarrs::array_subset::ArraySubset::new_with_ranges(&[0..2, 0..2]))
            .unwrap();
        data[[x as usize, y as usize]] = 0.0;
        array
            .store_chunks_ndarray(
                &zarrs::array_subset::ArraySubset::new_with_ranges(&[0..2, 0..2]),
                data,
            )
            .unwrap();
        */
    }
}

#[cfg(test)]
pub(crate) mod samples {
    use ndarray::Array2;
    use rand::Rng;

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

        let store: zarrs::storage::ReadableWritableListableStorage = std::sync::Arc::new(
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
                    &zarrs::array_subset::ArraySubset::new_with_ranges(&[
                        0..(ni / ci),
                        0..(nj / cj),
                    ]),
                    data,
                )
                .unwrap();
        }

        tmp_path.keep()
    }

    /// Create a zarr store with a cost layer comprised of all ones
    pub(crate) fn all_ones_cost_zarr() -> std::path::PathBuf {
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
            ndarray::Array::from_shape_vec((uni, unj), vec![1.0; uni * unj]).unwrap();

        array
            .store_chunks_ndarray(
                &zarrs::array_subset::ArraySubset::new_with_ranges(&[0..(ni / ci), 0..(nj / cj)]),
                data,
            )
            .unwrap();

        tmp_path.keep()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_cost_function_get_3x3() {
        let path = samples::multi_variable_zarr();
        let cost_function =
            CostFunction::from_json(r#"{"cost_layers": [{"layer_name": "A"}]}"#).unwrap();
        let dataset =
            Dataset::open(path, cost_function, 250_000_000).expect("Error opening dataset");

        let test_points = [ArrayIndex { i: 3, j: 1 }, ArrayIndex { i: 2, j: 2 }];
        let array = zarrs::array::Array::open(dataset.source.clone(), "/A").unwrap();
        for point in test_points {
            let results = dataset.get_3x3(&point);

            for (ArrayIndex { i, j }, val) in results {
                let subset =
                    zarrs::array_subset::ArraySubset::new_with_ranges(&[i..(i + 1), j..(j + 1)]);
                let subset_elements: Vec<f32> = array
                    .retrieve_array_subset_elements(&subset)
                    .expect("Error reading zarr data");
                assert_eq!(subset_elements.len(), 1);
                assert_eq!(subset_elements[0], val)
            }
        }
    }

    #[test]
    fn test_sample_cost_function_get_3x3() {
        let path = samples::multi_variable_zarr();
        let cost_function = crate::cost::sample::cost_function();
        let dataset =
            Dataset::open(path, cost_function, 250_000_000).expect("Error opening dataset");

        let test_points = [ArrayIndex { i: 3, j: 1 }, ArrayIndex { i: 2, j: 2 }];
        let array_a = zarrs::array::Array::open(dataset.source.clone(), "/A").unwrap();
        let array_b = zarrs::array::Array::open(dataset.source.clone(), "/B").unwrap();
        let array_c = zarrs::array::Array::open(dataset.source.clone(), "/C").unwrap();
        for point in test_points {
            let results = dataset.get_3x3(&point);

            for (ArrayIndex { i, j }, val) in results {
                let subset =
                    zarrs::array_subset::ArraySubset::new_with_ranges(&[i..(i + 1), j..(j + 1)]);
                let subset_elements_a: Vec<f32> = array_a
                    .retrieve_array_subset_elements(&subset)
                    .expect("Error reading zarr data");
                assert_eq!(subset_elements_a.len(), 1);

                let subset_elements_b: Vec<f32> = array_b
                    .retrieve_array_subset_elements(&subset)
                    .expect("Error reading zarr data");
                assert_eq!(subset_elements_b.len(), 1);

                let subset_elements_c: Vec<f32> = array_c
                    .retrieve_array_subset_elements(&subset)
                    .expect("Error reading zarr data");
                assert_eq!(subset_elements_c.len(), 1);

                assert_eq!(
                    subset_elements_a[0]
                        + subset_elements_b[0] * 100.
                        + subset_elements_a[0] * subset_elements_b[0]
                        + subset_elements_c[0] * subset_elements_a[0] * 2.,
                    val
                )
            }
        }
    }
}
