mod lazy_subset;
#[cfg(test)]
pub(crate) mod samples;

use std::iter;
use std::sync::RwLock;

use tracing::{debug, trace, warn};
use zarrs::array::ArrayChunkCacheExt;
use zarrs::storage::{
    ListableStorageTraits, ReadableListableStorage, ReadableWritableListableStorage,
};

use crate::ArrayIndex;
use crate::cost::CostFunction;
use crate::error::Result;
pub(crate) use lazy_subset::LazySubset;

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
    swap: ReadableWritableListableStorage,
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

        // ==== Create the swap dataset ====
        let tmp_path = tempfile::TempDir::new().unwrap();
        debug!(
            "Initializing a temporary swap dataset at {:?}",
            tmp_path.path()
        );
        let swap: ReadableWritableListableStorage = std::sync::Arc::new(
            zarrs::filesystem::FilesystemStore::new(tmp_path.path())
                .expect("could not open filesystem store"),
        );

        trace!("Creating a new group for the cost dataset");
        zarrs::group::GroupBuilder::new()
            .build(swap.clone(), "/")
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
        .build(swap.clone(), "/cost")
        .unwrap();
        trace!("Cost shape: {:?}", array.shape().to_vec());
        trace!("Cost chunk shape: {:?}", array.chunk_grid());
        array.store_metadata().unwrap();

        trace!("Cost dataset contents: {:?}", swap.list().unwrap());

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
            swap,
            cost_chunk_idx,
            cost_function,
            cache,
        })
    }

    fn calculate_chunk_cost(&self, ci: u64, cj: u64) {
        trace!("Creating a LazyChunk for ({}, {})", ci, cj);

        // cost variable is stored in the swap dataset
        let variable = zarrs::array::Array::open(self.swap.clone(), "/cost").unwrap();
        // Get the subset according to cost's chunk
        let subset = variable.chunk_subset(&[ci, cj]).unwrap();
        let data = LazySubset::<f32>::new(self.source.clone(), subset);
        let output = self.cost_function.compute(data);

        trace!("Cost function: {:?}", self.cost_function);

        /*
        trace!("Getting '/A' variable");
        let array = zarrs::array::Array::open(self.source.clone(), "/A").unwrap();
        let value = array.retrieve_chunk_ndarray::<f32>(&[i, j]).unwrap();
        trace!("Value: {:?}", value);
        trace!("Calculating cost for chunk ({}, {})", i, j);
        let output = value * 10.0;
        */

        let cost = zarrs::array::Array::open(self.swap.clone(), "/cost").unwrap();
        cost.store_metadata().unwrap();
        let chunk_indices: Vec<u64> = vec![ci, cj];
        trace!("Storing chunk at {:?}", chunk_indices);
        let chunk_subset =
            &zarrs::array_subset::ArraySubset::new_with_ranges(&[ci..(ci + 1), cj..(cj + 1)]);
        trace!("Target chunk subset: {:?}", chunk_subset);
        cost.store_chunks_ndarray(chunk_subset, output).unwrap();
    }

    pub(super) fn get_3x3(&self, index: &ArrayIndex) -> Vec<(ArrayIndex, f32)> {
        let &ArrayIndex { i, j } = index;

        trace!("Getting 3x3 neighborhood for (i={}, j={})", i, j);

        trace!("Cost dataset contents: {:?}", self.swap.list().unwrap());
        trace!("Cost dataset size: {:?}", self.swap.size().unwrap());

        trace!("Opening cost dataset");
        let cost = zarrs::array::Array::open(self.swap.clone(), "/cost").unwrap();
        trace!("Cost dataset with shape: {:?}", cost.shape());

        // Cutting off the edges for now.
        let shape = cost.shape();
        debug_assert!(!shape.contains(&0));

        let max_i = shape[0] - 1;
        let max_j = shape[1] - 1;

        let i_range = match i {
            0 if max_i == 0 => 0..1,
            0 => 0..2,
            _ if i == max_i => i - 1..i + 1,
            _ => i - 1..i + 2,
        };
        let j_range = match j {
            0 if max_j == 0 => 0..1,
            0 => 0..2,
            _ if j == max_j => j - 1..j + 1,
            _ => j - 1..j + 2,
        };

        // Capture the 3x3 neighborhood
        let subset =
            zarrs::array_subset::ArraySubset::new_with_ranges(&[i_range.clone(), j_range.clone()]);
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
                    debug!("Requesting write lock for cost_chunk_idx ({}, {})", ci, cj);
                    let mut chunk_idx = self
                        .cost_chunk_idx
                        .write()
                        .expect("Failed to acquire write lock");
                    debug!("Acquired write lock for cost_chunk_idx ({}, {})", ci, cj);
                    if chunk_idx[[ci as usize, cj as usize]] {
                        trace!(
                            "Cost for chunk ({}, {}) already calculated while waiting for the lock",
                            ci, cj
                        );
                    } else {
                        self.calculate_chunk_cost(ci, cj);
                        debug!("Recording chunk ({}, {}) as calculated", ci, cj);
                        chunk_idx[[ci as usize, cj as usize]] = true;
                    }
                    debug!("Released write lock for cost_chunk_idx ({}, {})", ci, cj);
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

        trace!("Input index: (i={}, j={})", i, j);

        /*
         * The transition between two gridpoint centers is along half the distance
         * on the original gridpoint, plus half the distance to the target gridpoint
         * (center). Therefore, the transition cost is the average between the origin
         * gridpoint cost and the target gridpoint cost.
         * Note that the same principle is valid for diagonals, it is still the average
         * of both values, but we have to scale for the longer distance along the
         * diagonal, thus a sqrt(2) factor along the diagonals.
         */

        // Match the indices
        let neighbors: Vec<((u64, u64), f32)> = i_range
            .flat_map(|e| iter::repeat(e).zip(j_range.clone()))
            .zip(value)
            .collect();
        trace!("Neighbors {:?}", neighbors);

        // Extract the origin point.
        let center = neighbors
            .iter()
            .find(|((ir, jr), _)| *ir == i && *jr == j)
            .unwrap();
        trace!("Center point: {:?}", center);

        // Calculate the average with center point (half grid + other half grid).
        // Also, apply the diagonal factor for the extra distance.
        let neighbors = neighbors
            .iter()
            .filter(|((ir, jr), _)| !(*ir == i && *jr == j)) // no center point
            .map(|((ir, jr), v)| ((ir, jr), 0.5 * (v + center.1)))
            .map(|((ir, jr), v)| {
                if *ir != i && *jr != j {
                    // Diagonal factor for longer distance (hypotenuse)
                    ((ir, jr), v * f32::sqrt(2.0))
                } else {
                    ((ir, jr), v)
                }
            })
            .map(|((ir, jr), v)| (ArrayIndex { i: *ir, j: *jr }, v))
            .collect::<Vec<_>>();
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
mod tests {
    use super::*;
    use std::f32::consts::SQRT_2;
    use test_case::test_case;

    #[allow(dead_code)]
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

    #[allow(dead_code)]
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

    #[test]
    fn test_get_3x3_single_item_array() {
        let path = samples::cost_as_index_zarr((1, 1), (1, 1));
        let cost_function =
            CostFunction::from_json(r#"{"cost_layers": [{"layer_name": "cost"}]}"#).unwrap();
        let dataset =
            Dataset::open(path, cost_function, 250_000_000).expect("Error opening dataset");

        let results = dataset.get_3x3(&ArrayIndex { i: 0, j: 0 });

        assert_eq!(results, vec![]);
    }

    #[test_case((0, 0), vec![(0, 1, 0.5), (1, 0, 1.0), (1, 1, 1.5 * SQRT_2)] ; "top left corner")]
    #[test_case((0, 1), vec![(0, 0, 0.5), (1, 0, 1.5 * SQRT_2), (1, 1, 2.)] ; "top right corner")]
    #[test_case((1, 0), vec![(0, 0, 1.), (0, 1, 1.5 * SQRT_2), (1, 1, 2.5)] ; "bottom left corner")]
    #[test_case((1, 1), vec![(0, 0, 1.5 * SQRT_2), (0, 1, 2.), (1, 0, 2.5)] ; "bottom right corner")]
    fn test_get_3x3_two_by_two_array((si, sj): (u64, u64), expected_output: Vec<(u64, u64, f32)>) {
        let path = samples::cost_as_index_zarr((2, 2), (2, 2));
        let cost_function =
            CostFunction::from_json(r#"{"cost_layers": [{"layer_name": "cost"}]}"#).unwrap();
        let dataset =
            Dataset::open(path, cost_function, 250_000_000).expect("Error opening dataset");

        let results = dataset.get_3x3(&ArrayIndex { i: si, j: sj });

        assert_eq!(
            results,
            expected_output
                .into_iter()
                .map(|(i, j, v)| (ArrayIndex { i, j }, v))
                .collect::<Vec<_>>()
        );
    }

    #[test_case((0, 0), vec![(0, 1, 0.5), (1, 0, 1.5), (1, 1, 2.0 * SQRT_2)] ; "top left corner")]
    #[test_case((0, 1), vec![(0, 0, 0.5), (0, 2, 1.5), (1, 0, 2.0 * SQRT_2), (1, 1, 2.5), (1, 2, 3. * SQRT_2)] ; "top middle")]
    #[test_case((0, 2), vec![(0, 1, 1.5), (1, 1, 3.0 * SQRT_2), (1, 2, 3.5)] ; "top right corner")]
    #[test_case((1, 0), vec![(0, 0, 1.5), (0, 1, 2.0 * SQRT_2), (1, 1, 3.5), (2, 0, 4.5), (2, 1, 5.0 * SQRT_2)] ; "middle left")]
    #[test_case((1, 1), vec![(0, 0, 2.0 * SQRT_2), (0, 1, 2.5), (0, 2, 3.0 * SQRT_2), (1, 0, 3.5), (1, 2, 4.5), (2, 0, 5.0 * SQRT_2), (2, 1, 5.5), (2, 2, 6.0 * SQRT_2)] ; "middle middle")]
    #[test_case((1, 2), vec![(0, 1, 3.0 * SQRT_2), (0, 2, 3.5), (1, 1, 4.5), (2, 1, 6.0 * SQRT_2), (2, 2, 6.5)] ; "middle right")]
    #[test_case((2, 0), vec![(1, 0, 4.5), (1, 1, 5.0 * SQRT_2), (2, 1, 6.5)] ; "bottom left corner")]
    #[test_case((2, 1), vec![(1, 0, 5.0 * SQRT_2), (1, 1, 5.5), (1, 2, 6.0 * SQRT_2), (2, 0, 6.5), (2, 2, 7.5)] ; "bottom middle")]
    #[test_case((2, 2), vec![(1, 1, 6.0 * SQRT_2), (1, 2, 6.5), (2, 1, 7.5)] ; "bottom right corner")]
    fn test_get_3x3_three_by_three_array(
        (si, sj): (u64, u64),
        expected_output: Vec<(u64, u64, f32)>,
    ) {
        let path = samples::cost_as_index_zarr((3, 3), (3, 3));
        let cost_function =
            CostFunction::from_json(r#"{"cost_layers": [{"layer_name": "cost"}]}"#).unwrap();
        let dataset =
            Dataset::open(path, cost_function, 250_000_000).expect("Error opening dataset");

        let results = dataset.get_3x3(&ArrayIndex { i: si, j: sj });

        assert_eq!(
            results,
            expected_output
                .into_iter()
                .map(|(i, j, v)| (ArrayIndex { i, j }, v))
                .collect::<Vec<_>>()
        );
    }

    #[test_case((0, 0), vec![(0, 1, 0.5), (1, 0, 2.), (1, 1, 2.5 * SQRT_2)] ; "top left corner")]
    #[test_case((0, 1), vec![(0, 0, 0.5), (0, 2, 1.5), (1, 0, 2.5 * SQRT_2), (1, 1, 3.), (1, 2, 3.5 * SQRT_2)] ; "top left edge")]
    #[test_case((0, 2), vec![(0, 1, 1.5), (0, 3, 2.5), (1, 1, 3.5 * SQRT_2), (1, 2, 4.), (1, 3, 4.5 * SQRT_2)] ; "top right edge")]
    #[test_case((0, 3), vec![(0, 2, 2.5), (1, 2, 4.5 * SQRT_2), (1, 3, 5.)] ; "top right corner")]
    #[test_case((1, 0), vec![(0, 0, 2.), (0, 1, 2.5 * SQRT_2), (1, 1, 4.5), (2, 0, 6.), (2, 1, 6.5 * SQRT_2)] ; "left top edge")]
    #[test_case((1, 3), vec![(0, 2, 4.5 * SQRT_2), (0, 3, 5.), (1, 2, 6.5), (2, 2, 8.5 * SQRT_2), (2, 3, 9.)] ; "right top edge")]
    #[test_case((2, 0), vec![(1, 0, 6.), (1, 1, 6.5 * SQRT_2), (2, 1, 8.5), (3, 0, 10.), (3, 1, 10.5 * SQRT_2)] ; "left bottom edge")]
    #[test_case((2, 3), vec![(1, 2, 8.5 * SQRT_2), (1, 3, 9.), (2, 2, 10.5), (3, 2, 12.5 * SQRT_2), (3, 3, 13.)] ; "right bottom edge")]
    #[test_case((3, 0), vec![(2, 0, 10.), (2, 1, 10.5 * SQRT_2), (3, 1, 12.5)] ; "bottom left corner")]
    #[test_case((3, 1), vec![(2, 0, 10.5 * SQRT_2), (2, 1, 11.), (2, 2, 11.5 * SQRT_2), (3, 0, 12.5), (3, 2, 13.5)] ; "bottom left edge")]
    #[test_case((3, 2), vec![(2, 1, 11.5 * SQRT_2), (2, 2, 12.), (2, 3, 12.5 * SQRT_2), (3, 1, 13.5), (3, 3, 14.5)] ; "bottom right edge")]
    #[test_case((3, 3), vec![(2, 2, 12.5 * SQRT_2), (2, 3, 13.), (3, 2, 14.5)] ; "bottom right corner")]
    fn test_get_3x3_four_by_four_array(
        (si, sj): (u64, u64),
        expected_output: Vec<(u64, u64, f32)>,
    ) {
        let path = samples::cost_as_index_zarr((4, 4), (2, 2));
        let cost_function =
            CostFunction::from_json(r#"{"cost_layers": [{"layer_name": "cost"}]}"#).unwrap();
        let dataset =
            Dataset::open(path, cost_function, 250_000_000).expect("Error opening dataset");

        let results = dataset.get_3x3(&ArrayIndex { i: si, j: sj });

        assert_eq!(
            results,
            expected_output
                .into_iter()
                .map(|(i, j, v)| (ArrayIndex { i, j }, v))
                .collect::<Vec<_>>()
        );
    }
}

/// Lazy chunk of a Zarr dataset
pub(crate) struct LazyChunk {
    /// Source Zarr storage
    source: ReadableListableStorage,
    /// Chunk index 1st dimension
    ci: u64,
    /// Chunk index 2nd dimension
    cj: u64,
    /// Data
    // We know it is a 2D array of f32. We might want to simplify and strict this definition.
    // data: std::collections::HashMap<String, ndarray::Array2<f32>>,
    data: std::collections::HashMap<
        String,
        ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<ndarray::IxDynImpl>>,
    >,
}

#[allow(dead_code)]
impl LazyChunk {
    pub(super) fn ci(&self) -> u64 {
        self.ci
    }

    pub(super) fn cj(&self) -> u64 {
        self.cj
    }

    //fn get(&self, variable: &str) -> Result<&ndarray::Array2<f32>> {
    pub(crate) fn get(
        &mut self,
        variable: &str,
    ) -> Result<ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<ndarray::IxDynImpl>>> {
        trace!("Getting chunk data for variable: {}", variable);

        Ok(match self.data.get(variable) {
            Some(v) => {
                trace!("Chunk data for variable {} already loaded", variable);
                v.clone()
            }
            None => {
                trace!("Loading chunk data for variable: {}", variable);
                let array = zarrs::array::Array::open(self.source.clone(), &format!("/{variable}"))
                    .unwrap();
                let chunk_indices = &[self.ci, self.cj];
                let chunk_subset = zarrs::array_subset::ArraySubset::new_with_ranges(&[
                    chunk_indices[0]..(chunk_indices[0] + 1),
                    chunk_indices[1]..(chunk_indices[1] + 1),
                ]);
                trace!("Storing chunk data for variable: {}", variable);
                let values = array.retrieve_chunks_ndarray::<f32>(&chunk_subset).unwrap();
                // array.retrieve_chunk_ndarray::<f32>(&[ci, cj]).unwrap();
                self.data.insert(variable.to_string(), values.clone());
                values
            }
        })
    }
}

#[cfg(test)]
mod chunk_tests {
    use super::*;

    #[test]
    fn dev() {
        let path = samples::multi_variable_zarr();
        let store: zarrs::storage::ReadableListableStorage =
            std::sync::Arc::new(zarrs::filesystem::FilesystemStore::new(&path).unwrap());

        let mut chunk = LazyChunk {
            source: store,
            ci: 0,
            cj: 0,
            data: std::collections::HashMap::new(),
        };

        assert_eq!(chunk.ci, 0);
        assert_eq!(chunk.cj, 0);

        let _tmp = chunk.get("A").unwrap();
    }
}
