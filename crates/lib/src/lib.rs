//! # Path optimization with weigthed costs
//!
//!

mod error;

use pathfinding::prelude::dijkstra;
use tracing::trace;

use error::Result;

struct Dataset {
    store: zarrs::storage::ReadableListableStorage,
    cache: zarrs::array::ChunkCacheLruChunkLimit<zarrs::array::ChunkCacheTypeDecoded>,
}

impl Dataset {
    fn new<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let filesystem = zarrs::filesystem::FilesystemStore::new(path).expect("could not open filesystem store");

        let store: zarrs::storage::ReadableListableStorage = std::sync::Arc::new(filesystem);
        // ATENTION: Hardcoded in ~250MB
        let cache = zarrs::array::ChunkCacheLruChunkLimit::new(250_000_000);

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
    fn successors(&self, position: &Pos) -> Vec<(Pos, u64)> {
        let &Pos(x, y) = position;
        let array = zarrs::array::Array::open(self.store.clone(), "/A").unwrap();


        // Cutting off the edges for now.
        let shape = array.shape();
        if x == 0 || x >= (shape[0]) || y == 0 || y >= (shape[1]) {
            return vec![];
        }

        let subset = zarrs::array_subset::ArraySubset::new_with_ranges(&[(x-1)..(x+2), (y-1)..(y+2)]);

        //let value = array.retrieve_array_subset_ndarray::<f64>(&subset).unwrap();
        let value = array.retrieve_array_subset_elements::<f64>(&subset).unwrap();
        dbg!(&value);

        let output = vec![
            (Pos(x-1, y-1), (value[0] * 1e4) as u64),
            (Pos(x, y-1), (value[1]* 1e4) as u64),
            (Pos(x+1, y-1), (value[2]* 1e4) as u64),
            (Pos(x-1, y), (value[3]* 1e4) as u64),
            (Pos(x+1, y), (value[5]* 1e4) as u64),
            (Pos(x-1, y+1), (value[6]* 1e4) as u64),
            (Pos(x, y+1), (value[7]* 1e4) as u64),
            (Pos(x+1, y+1), (value[8]* 1e4) as u64),
        ];
        output


    }
}


#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct Point(u32, u32);

struct Simulation {
    store: zarrs::storage::ReadableListableStorage,
    cache: zarrs::array::ChunkCacheLruChunkLimit<zarrs::array::ChunkCacheTypeDecoded>,
}

use ndarray::parallel::prelude::{IntoParallelIterator, ParallelIterator};
impl Simulation {
    fn new<P: AsRef<std::path::Path>>(store_path: P, cache_size: u64) -> Result<Self> {

        let filesystem = zarrs::filesystem::FilesystemStore::new(store_path)
            .expect("could not open filesystem store");
        dbg!(&filesystem);

        let store: zarrs::storage::ReadableListableStorage = std::sync::Arc::new(filesystem);

        let cache = zarrs::array::ChunkCacheLruChunkLimit::new(cache_size);

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

        let array = zarrs::array::Array::open(self.store.clone(), "/A").unwrap();

        // Cutting off the edges for now.
        let shape = array.shape();
        if x == 0 || x as u64 >= (shape[0]) || y == 0 || y as u64 >= (shape[1]) {
            return vec![];
        }

        // Capture the 3x3 neighborhood
        let subset = zarrs::array_subset::ArraySubset::new_with_ranges(&[
            (x as u64 - 1)..(x as u64 + 2),
            (y as u64 - 1)..(y as u64 + 2),
        ]);
        trace!("Subset {:?}", subset);

        // Retrieve the 3x3 neighborhood values
        let value = array
            .retrieve_array_subset_elements_opt_cached::<f64, zarrs::array::ChunkCacheTypeDecoded>(
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
