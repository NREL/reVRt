//! # Path optimization with weighted costs
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
    i: u64,
    j: u64,
}

impl ArrayIndex {
    pub fn new(i: u64, j: u64) -> Self {
        Self { i, j }
    }
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
    const PRECISION_SCALAR: f32 = 1e4;

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
    /// - Weight the cost. Remember that the cost is for a side,
    ///   thus a diagonal move has to calculate consider the longer
    ///   distance.
    /// - Add starting cell cost by adding a is_start parameter and
    ///   passing it down to the get_3x3 function so that it can add
    ///   the center pixel to all successor cost values
    fn successors(&self, position: &ArrayIndex) -> Vec<(ArrayIndex, u64)> {
        trace!("Position {:?}", position);
        let neighbors = self.dataset.get_3x3(position);
        let neighbors = neighbors
            .into_iter()
            .map(|(p, c)| (p, cost_as_u64(c))) // ToDo: Maybe it's better to have get_3x3 return a u64 - then we can skip this map altogether
            .collect();
        trace!("Adjusting neighbors' types: {:?}", neighbors);
        neighbors
    }

    fn scout(&mut self, start: &[ArrayIndex], end: Vec<ArrayIndex>) -> Vec<(Vec<ArrayIndex>, f32)> {
        start
            .into_par_iter()
            .filter_map(|s| dijkstra(s, |p| self.successors(p), |p| end.contains(p)))
            .map(|(path, final_cost)| (path, unscaled_cost(final_cost)))
            .collect()
    }
}

fn cost_as_u64(cost: f32) -> u64 {
    let cost = cost * Simulation::PRECISION_SCALAR;
    cost as u64
}

fn unscaled_cost(cost: u64) -> f32 {
    (cost as f32) / Simulation::PRECISION_SCALAR
}

pub fn resolve<P: AsRef<std::path::Path>>(
    store_path: P,
    cost_function: &str,
    cache_size: u64,
    start: &[ArrayIndex],
    end: Vec<ArrayIndex>,
) -> Result<Vec<(Vec<ArrayIndex>, f32)>> {
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
    use test_case::test_case;

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
        let vec_of_indices = [ArrayIndex { i: 2, j: 3 }, ArrayIndex { i: 5, j: 6 }];
        assert!(vec_of_indices.contains(&ArrayIndex { i: 5, j: 6 }));
        assert!(!vec_of_indices.contains(&ArrayIndex { i: 8, j: 9 }));
    }

    #[test]
    fn minimalist() {
        let store_path = dataset::samples::multi_variable_zarr();
        let cost_function = cost::sample::cost_function();
        let mut simulation = Simulation::new(&store_path, cost_function, 250_000_000).unwrap();
        let start = vec![ArrayIndex { i: 2, j: 3 }];
        let end = vec![ArrayIndex { i: 6, j: 6 }];
        let solutions = simulation.scout(&start, end);
        dbg!(&solutions);
        assert_eq!(solutions.len(), 1);
        let (track, cost) = &solutions[0];
        assert!(track.len() > 1);
        assert!(cost > &0.);
    }

    #[test_case((1, 1), (1, 1), 1, 0.; "no movement")]
    #[test_case((1, 1), (1, 2), 2, 1.; "step one cell to the side")]
    #[test_case((1, 1), (2, 1), 2, 1.; "step one cell down")]
    #[test_case((1, 1), (2, 2), 2, 1.; "step one cell diagonally")]
    #[test_case((1, 1), (2, 3), 3, 2.; "step diagonally and across")]
    fn basic_routing_point_to_point(
        (si, sj): (u64, u64),
        (ei, ej): (u64, u64),
        expected_num_steps: usize,
        expected_cost: f32,
    ) {
        let store_path = dataset::samples::all_ones_cost_zarr();
        let cost_function =
            CostFunction::from_json(r#"{"cost_layers": [{"layer_name": "cost"}]}"#).unwrap();
        let mut simulation = Simulation::new(&store_path, cost_function, 250_000_000).unwrap();
        let start = vec![ArrayIndex { i: si, j: sj }];
        let end = vec![ArrayIndex { i: ei, j: ej }];
        let solutions = simulation.scout(&start, end);
        dbg!(&solutions);
        assert_eq!(solutions.len(), 1);
        let (track, cost) = &solutions[0];
        assert_eq!(track.len(), expected_num_steps);
        assert_eq!(cost, &expected_cost);
    }

    #[test_case((1, 1), vec![(1, 4), (3, 1), (4, 4)], (3, 1), 3, 2.; "different cost endpoints")]
    fn basic_routing_one_point_to_many(
        (si, sj): (u64, u64),
        endpoints: Vec<(u64, u64)>,
        expected_endpoint: (u64, u64),
        expected_num_steps: usize,
        expected_cost: f32,
    ) {
        let store_path = dataset::samples::all_ones_cost_zarr();
        let cost_function =
            CostFunction::from_json(r#"{"cost_layers": [{"layer_name": "cost"}]}"#).unwrap();
        let mut simulation = Simulation::new(&store_path, cost_function, 250_000_000).unwrap();
        let start = vec![ArrayIndex { i: si, j: sj }];
        let end = endpoints
            .clone()
            .into_iter()
            .map(|(i, j)| ArrayIndex { i, j })
            .collect();
        let solutions = simulation.scout(&start, end);
        dbg!(&solutions);
        assert_eq!(solutions.len(), 1);
        let (track, cost) = &solutions[0];
        assert_eq!(track.len(), expected_num_steps);
        assert_eq!(cost, &expected_cost);
        assert_eq!(track[0], start[0]);

        let &ArrayIndex { i: ei, j: ej } = track.last().unwrap();
        assert_eq!((ei, ej), expected_endpoint);
    }
}
