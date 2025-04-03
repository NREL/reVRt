use std::sync::RwLock;

use tracing::{debug, trace, warn};
use zarrs::array::ArrayChunkCacheExt;
use zarrs::storage::{ReadableListableStorage, ReadableWritableListableStorage};

use crate::Point;
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
    /// Cache for the cost
    cache: zarrs::array::ChunkCacheLruSizeLimit<zarrs::array::ChunkCacheTypeDecoded>,
}

impl Dataset {
    pub(super) fn open<P: AsRef<std::path::Path>>(path: P, cache_size: u64) -> Result<Self> {
        trace!("Opening dataset");
        trace!("Building FilesystemStore with path: {:?}", path.as_ref());
        let filesystem =
            zarrs::filesystem::FilesystemStore::new(path).expect("could not open filesystem store");
        let source = std::sync::Arc::new(filesystem);

        // ==== Create the cost dataset ====
        let tmp_path = tempfile::TempDir::new().unwrap();
        trace!(
            "Initializing a temporary cost dataset at {:?}",
            tmp_path.path()
        );
        let cost: ReadableWritableListableStorage = std::sync::Arc::new(
            zarrs::filesystem::FilesystemStore::new(&tmp_path.path())
                .expect("could not open filesystem store"),
        );

        trace!("Creating a new group for the cost dataset");
        zarrs::group::GroupBuilder::new()
            .build(cost.clone(), "/")
            .unwrap()
            .store_metadata()
            .unwrap();

        // -- Temporary solution to specify cost storage --
        let tmp = zarrs::array::Array::open(source.clone(), "/A").unwrap();
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
        warn!("Cost shape: {:?}", array.shape().to_vec());
        warn!("Cost chunk shape: {:?}", array.chunk_grid());
        array.store_metadata().unwrap();

        trace!("Cost dataset contents: {:?}", cost.list().unwrap());

        let cost_chunk_idx = ndarray::Array2::from_elem((2, 2), false).into();

        if cache_size < 1_000_000 {
            warn!("Cache size smalled than 1MB");
        }
        trace!("Creating cache with size {}MB", cache_size / 1_000_000);
        let cache = zarrs::array::ChunkCacheLruSizeLimit::new(cache_size);

        trace!("Dataset opened successfully");
        Ok(Self {
            source,
            cost_path: tmp_path,
            cost,
            cost_chunk_idx,
            cache,
        })
    }

    fn calculate_chunk_cost(&self, i: u64, j: u64) {
        debug!("Calculating cost for chunk ({}, {})", i, j);

        trace!("Getting '/A' variable");
        let array = zarrs::array::Array::open(self.source.clone(), "/A").unwrap();
        let value = array.retrieve_chunk_ndarray::<f32>(&[i, j]).unwrap();
        trace!("Value: {:?}", value);
        trace!("Calculating cost for chunk ({}, {})", i, j);
        let output = value * 10.0;

        let cost = zarrs::array::Array::open(self.cost.clone(), "/cost").unwrap();
        cost.store_metadata().unwrap();
        let chunk_indices: Vec<u64> = vec![i, j];
        trace!("Storing chunk at {:?}", chunk_indices);
        let chunk_subset =
            &zarrs::array_subset::ArraySubset::new_with_ranges(&[i..(i + 1), j..(j + 1)]);
        trace!("Target chunk subset: {:?}", chunk_subset);
        cost.store_chunks_ndarray(&chunk_subset, output).unwrap();
    }

    pub(super) fn get_3x3(&self, x: u64, y: u64) -> Vec<(Point, f32)> {
        trace!("Getting 3x3 neighborhood for ({}, {})", x, y);

        trace!("Cost dataset contents: {:?}", self.cost.list().unwrap());
        // What is this size?
        trace!("Cost dataset size: {:?}", self.cost.size().unwrap());

        trace!("Opening cost dataset");
        let cost = zarrs::array::Array::open(self.cost.clone(), "/cost").unwrap();
        trace!("Cost dataset with shape: {:?}", cost.shape());

        // Cutting off the edges for now.
        let shape = cost.shape();
        if x == 0 || x >= (shape[0] - 1) || y == 0 || y >= (shape[1] - 1) {
            warn!("I'm not ready to deal with the edges yet");
            return vec![];
        }

        // Capture the 3x3 neighborhood
        let subset = zarrs::array_subset::ArraySubset::new_with_ranges(&[
            (x - 1)..(x + 2),
            (y - 1)..(y + 2),
        ]);
        trace!("Cost subset: {:?}", subset);

        // Find the chunks that intersect the subset
        let chunks = &cost.chunks_in_array_subset(&subset).unwrap().unwrap();
        trace!("Cost chunks: {:?}", chunks);
        trace!(
            "Cost subset extends to {:?} chunks",
            chunks.num_elements_usize()
        );

        for i in chunks.start()[0]..(chunks.start()[0] + chunks.shape()[0]) {
            for j in chunks.start()[1]..(chunks.start()[1] + chunks.shape()[1]) {
                trace!(
                    "Checking if cost for chunk ({}, {}) has been calculated",
                    i, j
                );
                if self.cost_chunk_idx.read().unwrap()[[i as usize, j as usize]] {
                    trace!("Cost for chunk ({}, {}) already calculated", i, j);
                } else {
                    self.calculate_chunk_cost(i, j);
                    let mut chunk_idx = self.cost_chunk_idx.write().unwrap();
                    chunk_idx[[i as usize, j as usize]] = true;
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
            (Point(x - 1, y - 1), value[0]),
            (Point(x, y - 1), value[1]),
            (Point(x + 1, y - 1), value[2]),
            (Point(x - 1, y), value[3]),
            (Point(x + 1, y), value[5]),
            (Point(x - 1, y + 1), value[6]),
            (Point(x, y + 1), value[7]),
            (Point(x + 1, y + 1), value[8]),
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

    /// Create a single variable (/A) zarr store
    ///
    /// Just a proof of concept with lots of hardcoded values
    /// that must be improved.
    pub(crate) fn single_variable_zarr() -> std::path::PathBuf {
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
        for array_path in ["/A", "/B", "/C", "/cost"].iter() {
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

        tmp_path.into_path()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_variable_zarr() {
        let path = samples::single_variable_zarr();
        let mut dataset = Dataset::open(path, 250_000_000).unwrap();

        let results = dataset.get_3x3(3, 2);
        dbg!(&results);
        let results = dataset.get_3x3(2, 2);
        dbg!(&results);
    }
}
