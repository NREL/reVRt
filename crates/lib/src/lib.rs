//! # Path optimization with weigthed costs
//!
//!

mod dataset;
mod error;

use pathfinding::prelude::dijkstra;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use tracing::trace;

use error::Result;


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
