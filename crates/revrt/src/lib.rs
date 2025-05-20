//! # Path optimization with weigthed costs
//!
//!

mod cost;
mod dataset;
mod error;
mod ffi;

use pathfinding::prelude::dijkstra;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use tracing::trace;

use cost::CostFunction;
use error::Result;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ArrayIndex {
    pub i: u64,
    pub j: u64,
}

impl From<ArrayIndex> for (u64, u64) {
    fn from(ArrayIndex { i, j }: ArrayIndex) -> (u64, u64) {
        (i, j)
    }
}

struct Simulation {
    dataset: dataset::Dataset,
}

impl Simulation {
    fn new<P: AsRef<std::path::Path>>(
        store_path: P,
        cost_function: CostFunction,
        cache_size: u64,
    ) -> Result<Self> {
        let dataset = dataset::Dataset::open(store_path, cost_function, cache_size)?;

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
    fn successors(&self, position: &ArrayIndex) -> Vec<(ArrayIndex, usize)> {
        trace!("Successors of {:?}", position);

        let &ArrayIndex { i, j } = position;
        trace!("Array index i={} j={}", i, j);

        let neighbors = self.dataset.get_3x3(i, j);
        let neighbors = neighbors
            .into_iter()
            .map(|(p, c)| (p, (1e4 * c) as usize))
            .collect();
        trace!("Adjusting neighbors' types: {:?}", neighbors);
        neighbors
    }

    fn scout(
        &mut self,
        start: &[ArrayIndex],
        end: Vec<ArrayIndex>,
    ) -> Vec<(Vec<ArrayIndex>, usize)> {
        start
            .into_par_iter()
            .filter_map(|s| dijkstra(s, |p| self.successors(p), |p| end.contains(p)))
            .collect::<Vec<_>>()
    }
}

pub fn resolve<P: AsRef<std::path::Path>>(
    store_path: P,
    cost_function: &str,
    cache_size: u64,
    start: &[ArrayIndex],
    end: Vec<ArrayIndex>,
) -> Result<Vec<(Vec<ArrayIndex>, usize)>> {
    tracing::trace!("Cost function: {}", cost_function);
    let cost_function = CostFunction::from_json(cost_function)?;
    tracing::trace!("Cost function: {:?}", cost_function);
    let mut simulation: Simulation =
        Simulation::new(store_path, cost_function, cache_size).unwrap();
    let result = simulation.scout(start, end);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tuple_from_index() {
        let index_tuple: (u64, u64) = From::from(ArrayIndex { i: 2, j: 3 });
        assert_eq!(index_tuple.0, 2);
        assert_eq!(index_tuple.1, 3);
    }

    #[test]
    fn index_into_tuple() {
        let index_tuple: (u64, u64) = ArrayIndex { i: 2, j: 3 }.into();
        assert_eq!(index_tuple.0, 2);
        assert_eq!(index_tuple.1, 3);
    }

    #[test]
    fn vec_contains_index() {
        let vec_of_indices = vec![ArrayIndex { i: 2, j: 3 }, ArrayIndex { i: 5, j: 6 }];
        assert!(vec_of_indices.contains(&ArrayIndex { i: 5, j: 6 }));
        assert!(!vec_of_indices.contains(&ArrayIndex { i: 8, j: 9 }));
    }

    #[test]
    fn minimalist() {
        let store_path = dataset::samples::single_variable_zarr();
        let cost_function = cost::sample::cost_function();
        //let cost_function = CostFunction::from_json(&cost::sample::as_text_v1()).unwrap();
        let mut simulation = Simulation::new(&store_path, cost_function, 250_000_000).unwrap();
        let start = vec![ArrayIndex { i: 2, j: 3 }];
        let end = vec![ArrayIndex { i: 6, j: 6 }];
        let solutions = simulation.scout(&start, end);
        dbg!(&solutions);
        assert!(solutions.len() == 1);
        let (track, cost) = &solutions[0];
        assert!(track.len() > 1);
        assert!(cost > &0);
    }
}
