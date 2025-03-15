//! # Path optimization with weigthed costs
//!
//!

mod error;

use pathfinding::prelude::dijkstra;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use tracing::trace;
use zarrs::array::ArrayChunkCacheExt;

use error::Result;

/// Manages the features datasets and calculated total cost
struct Dataset {
    /// One or more storages with the features
    // Might benefit of creating a catalog mapping the features
    // to datasets and repective paths.
    source: zarrs::storage::ReadableListableStorage,
    // source: AsyncReadableListableStorage,
    //source: Vec<AsyncReadableListableStorage>,
    /// Variables used to define cost
    /// Minimalist solution for the cost calculation. In the future
    /// it will be modified to include weights and other types of
    /// relations such as operations between features.
    /// At this point it just allows custom variables names and the
    /// cost is calculated from multiple variables.
    cost_variables: Vec<String>,
    /// Storage for the calculated cost
    cost: zarrs::storage::ReadableWritableListableStorage,
    /// Cache for the cost
    cache: zarrs::array::ChunkCacheLruSizeLimit<zarrs::array::ChunkCacheTypeDecoded>,
}

impl Dataset {
    fn new<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let filesystem =
            zarrs::filesystem::FilesystemStore::new(path).expect("could not open filesystem store");

        let source: zarrs::storage::ReadableListableStorage = std::sync::Arc::new(filesystem);
        // ATENTION: Hardcoded in ~250MB
        let cache = zarrs::array::ChunkCacheLruSizeLimit::new(250_000_000);

        let cost_variables = vec!["A".to_string()];
        let tmp_path = tempfile::TempDir::new().unwrap();
        let cost = zarrs::filesystem::FilesystemStore::new(tmp_path.path())
            .expect("could not open filesystem store");
        let cost = std::sync::Arc::new(cost);

        Ok(Self {
            source,
            cost_variables,
            cache,
            cost,
        })
    }

    /*
    fn value(&self, Pos(x, y): Pos) -> f32 {
        let array = zarrs::array::Array::open(self.store.clone(), "/A").unwrap();
        let subset = zarrs::array_subset::ArraySubset::new_with_ranges(&[x..(x + 1), y..(y + 1)]);
        let value = array
            .retrieve_array_subset_elements::<f32>(&subset)
            .unwrap();
        value[0]
    }

    //let value = array.retrieve_array_subset_ndarray::<f32>(&subset).unwrap();
    // let value = array.retrieve_array_subset_elements::<f32>(&subset).unwrap();
    */
}

#[cfg(test)]
mod test_dataset {
    use super::*;

    fn dev() {
        let store_path = samples::single_variable_zarr();
        let dataset = Dataset::new(&store_path).unwrap();

        let array = zarrs::array::Array::open(dataset.source.clone(), "/cost").unwrap();
        let subset = zarrs::array_subset::ArraySubset::new_with_ranges(&[0..5, 0..2]);
        dbg!(&subset);

        // Find the chunks that intersect the subset
        let target_chunks = &array.chunks_in_array_subset(&subset).unwrap();
        dbg!(&target_chunks);

        let value = array
            .retrieve_array_subset_elements::<f32>(&subset)
            .unwrap();
        dbg!(&value);

        assert!(false);
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct Point(u64, u64);

struct Simulation {
    store: zarrs::storage::ReadableListableStorage,
    cache: zarrs::array::ChunkCacheLruSizeLimit<zarrs::array::ChunkCacheTypeDecoded>,
}


impl Simulation {
    fn new<P: AsRef<std::path::Path>>(store_path: P, cache_size: u64) -> Result<Self> {
        let filesystem = zarrs::filesystem::FilesystemStore::new(store_path)
            .expect("could not open filesystem store");

        let store: zarrs::storage::ReadableListableStorage = std::sync::Arc::new(filesystem);

        let cache = zarrs::array::ChunkCacheLruSizeLimit::new(cache_size);

        Ok(Self { store, cache })
    }

    /// Determine the successors of a position.
    ///
    /// ToDo:
    /// - Handle the edges of the array.
    /// - Include diagonal moves.
    /// - Weight the cost. Remember that the cost is for a side,
    ///   thus a diagnonal move has to calculate consider the longer
    ///   distance.
    fn successors(&self, position: &Point) -> Vec<(Point, usize)> {
        trace!("Successors of {:?}", position);

        let &Point(x, y) = position;

        let array = zarrs::array::Array::open(self.store.clone(), "/cost").unwrap();

        // Cutting off the edges for now.
        let shape = array.shape();
        if x == 0 || x >= (shape[0]) || y == 0 || y >= (shape[1]) {
            return vec![];
        }

        // Capture the 3x3 neighborhood
        let subset = zarrs::array_subset::ArraySubset::new_with_ranges(&[
            (x - 1)..(x + 2),
            (y - 1)..(y + 2),
        ]);
        trace!("Subset {:?}", subset);

        // Retrieve the 3x3 neighborhood values
        let value = array
            .retrieve_array_subset_elements_opt_cached::<f32, zarrs::array::ChunkCacheTypeDecoded>(
                &self.cache,
                &subset,
                &zarrs::array::codec::CodecOptions::default(),
            )
            .unwrap();

        trace!("Read values {:?}", value);

        let neighbors = vec![
            (Point(x - 1, y - 1), (value[0] * 1e4) as usize),
            (Point(x, y - 1), (value[1] * 1e4) as usize),
            (Point(x + 1, y - 1), (value[2] * 1e4) as usize),
            (Point(x - 1, y), (value[3] * 1e4) as usize),
            (Point(x + 1, y), (value[5] * 1e4) as usize),
            (Point(x - 1, y + 1), (value[6] * 1e4) as usize),
            (Point(x, y + 1), (value[7] * 1e4) as usize),
            (Point(x + 1, y + 1), (value[8] * 1e4) as usize),
        ];
        trace!("Neighbors {:?}", neighbors);
        neighbors
    }

    fn scout(&self, start: &[Point], end: Vec<Point>) -> Vec<(Vec<Point>, usize)> {
        start
            .into_par_iter()
            .filter_map(|s| dijkstra(s, |p| self.successors(p), |p| end.contains(p)))
            .collect::<Vec<_>>()
    }
}

#[cfg(test)]
pub(crate) mod samples {
    use ndarray::Array2;
    use rand::Rng;

    /// Create a single variable (/cost) zarr store
    ///
    /// Just a proof of concept with lots of hardcoded values
    /// that must be improved.
    pub(crate) fn single_variable_zarr() -> std::path::PathBuf {
        let ni = 8;
        let nj = 8;

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
        let array_path = "/cost";
        let array = zarrs::array::ArrayBuilder::new(
            vec![ni, nj], // array shape
            zarrs::array::DataType::Float32,
            vec![4, 4].try_into().unwrap(), // regular chunk shape
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
                &zarrs::array_subset::ArraySubset::new_with_ranges(&[0..2, 0..2]),
                data,
            )
            .unwrap();

        tmp_path.into_path()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimalist() {
        let store_path = samples::single_variable_zarr();
        let simulation = Simulation::new(&store_path, 250_000_000).unwrap();
        let start = vec![Point(2, 3)];
        let end = vec![Point(6, 6)];
        let solutions = simulation.scout(&start, end);
        assert!(solutions.len() == 1);
        let (track, cost) = &solutions[0];
        assert!(track.len() > 1);
        assert!(cost > &0);
        dbg!(&solutions);
    }
}
