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
pub struct Point(u64, u64);

impl Point {
    pub fn new(x: u64, y: u64) -> Self {
        Self(x, y)
    }
}

impl From<(u64, u64)> for Point {
    fn from((x, y): (u64, u64)) -> Self {
        Self(x, y)
    }
}

impl From<Point> for (u64, u64) {
    fn from(Point(x, y): Point) -> (u64, u64) {
        (x, y)
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
    ///   thus a diagonal move has to calculate consider the longer
    ///   distance.
    fn successors(&self, position: &Point) -> Vec<(Point, usize)> {
        trace!("Position {:?}", position);
        let neighbors = self.dataset.get_3x3(position);
        let neighbors = neighbors
            .into_iter()
            .map(|(p, c)| (p, (1e4 * c) as usize))
            .collect();
        trace!("Adjusting neighbors' types: {:?}", neighbors);
        neighbors
    }

    fn scout(&mut self, start: &[Point], end: Vec<Point>) -> Vec<(Vec<Point>, usize)> {
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
    start: &[Point],
    end: Vec<Point>,
) -> Result<Vec<(Vec<Point>, usize)>> {
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
    fn point_from_tuple() {
        let point = Point::from((2, 3));
        assert_eq!(point.0, 2);
        assert_eq!(point.1, 3);
    }

    #[test]
    fn tuple_into_point() {
        let point: Point = (2, 3).into();
        assert_eq!(point.0, 2);
        assert_eq!(point.1, 3);
    }

    #[test]
    fn tuple_from_point() {
        let point_tuple: (u64, u64) = From::from(Point(2, 3));
        assert_eq!(point_tuple.0, 2);
        assert_eq!(point_tuple.1, 3);
    }

    #[test]
    fn point_into_tuple() {
        let point_tuple: (u64, u64) = Point(2, 3).into();
        assert_eq!(point_tuple.0, 2);
        assert_eq!(point_tuple.1, 3);
    }

    #[test]
    fn minimalist() {
        let store_path = dataset::samples::single_variable_zarr();
        let cost_function = cost::sample::cost_function();
        //let cost_function = CostFunction::from_json(&cost::sample::as_text_v1()).unwrap();
        let mut simulation = Simulation::new(&store_path, cost_function, 250_000_000).unwrap();
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
