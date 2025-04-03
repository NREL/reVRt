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
    dataset: dataset::Dataset,
}

impl Simulation {
    pub fn new<P: AsRef<std::path::Path>>(store_path: P, cache_size: u64) -> Result<Self> {
        let dataset = dataset::Dataset::open(store_path, cache_size)?;

        Ok(Self { dataset })
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
        trace!("Position {} {}", x, y);

        let neighbors = self.dataset.get_3x3(x, y);
        //dbg!(&neighbors);
        let neighbors = neighbors
            .into_iter()
            .map(|(p, c)| (p, (1e4 * c) as usize))
            .collect();
        //dbg!(&neighbors);
        return neighbors;
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
