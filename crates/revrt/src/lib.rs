//! # Path optimization with weighted costs
//!
//!

mod cost;
mod dataset;
mod error;
mod ffi;

use pathfinding::prelude::dijkstra;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use tracing::{debug, trace};

use cost::CostFunction;
use error::Result;

#[allow(missing_docs)]
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ArrayIndex {
    i: u64,
    j: u64,
}

impl ArrayIndex {
    #[allow(missing_docs)]
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
        debug!("Starting scout with {} start points", start.len());

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

#[allow(missing_docs)]
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

#[inline]
/// A public interface to run benchmarks
///
/// This function is intended for use during development only. It will
/// eventually be replaced by a builder, thus more flexible and usable
/// for other purposes.
pub fn bench_minimalist(
    features_path: std::path::PathBuf,
    start: Vec<ArrayIndex>,
    end: Vec<ArrayIndex>,
) {
    // temporary solution for a cost function until we have a builder
    let cost_json = r#"
      {
        "cost_layers": [
          {"layer_name": "A"},
          {"layer_name": "B", "multiplier_scalar": 100},
          {"layer_name": "A",
            "multiplier_layer": "B"},
          {"layer_name": "C",
            "multiplier_layer": "A",
            "multiplier_scalar": 2}
]
        }
        "#
    .to_string();
    let cost_function = CostFunction::from_json(&cost_json).unwrap();

    let mut simulation: Simulation =
        Simulation::new(&features_path, cost_function, 250_000_000).unwrap();
    let solutions: Vec<(Vec<ArrayIndex>, f32)> = simulation.scout(&start, end);
    assert!(!solutions.is_empty(), "No solutions found");
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
    #[allow(clippy::approx_constant)]
    // Due to truncation solution to handle f32 costs.
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

    // Due to truncation solution to handle f32 costs.
    #[allow(clippy::approx_constant)]
    #[test_case((1, 1), (1, 1), 1, 0.; "no movement")]
    #[test_case((1, 1), (1, 2), 2, 1.; "step one cell to the side")]
    #[test_case((1, 1), (2, 1), 2, 1.; "step one cell down")]
    #[test_case((1, 1), (2, 2), 2, 1.4142; "step one cell diagonally")]
    #[test_case((1, 1), (2, 3), 3, 2.4142; "step diagonally and across")]
    fn basic_routing_point_to_point(
        (si, sj): (u64, u64),
        (ei, ej): (u64, u64),
        expected_num_steps: usize,
        expected_cost: f32,
    ) {
        let store_path = dataset::samples::constant_value_cost_zarr(1.0);
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
        let store_path = dataset::samples::constant_value_cost_zarr(1.0);
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

    #[test_case((1, 1), vec![(1, 3), (3, 1)], 1.; "horizontal and vertical")]
    #[test_case((3, 3), vec![(3, 5), (1, 1), (3, 1)], 1.; "horizontal")]
    #[test_case((3, 3), vec![(5, 3), (5, 5), (1, 3)], 1.; "vertical")]
    #[test_case((3, 3), vec![(3, 1), (3, 4)], 0.; "zero costs")]
    fn routing_one_point_to_many_same_cost_and_length(
        (si, sj): (u64, u64),
        endpoints: Vec<(u64, u64)>,
        cost_array_fill: f32,
    ) {
        let store_path = dataset::samples::constant_value_cost_zarr(cost_array_fill);
        let cost_function =
            CostFunction::from_json(r#"{"cost_layers": [{"layer_name": "cost"}]}"#).unwrap();
        let mut simulation = Simulation::new(&store_path, cost_function, 250_000_000).unwrap();
        let start = vec![ArrayIndex { i: si, j: sj }];
        let end = endpoints
            .clone()
            .into_iter()
            .map(|(i, j)| ArrayIndex { i, j })
            .collect();
        let mut solutions = simulation.scout(&start, end);
        dbg!(&solutions);
        assert_eq!(solutions.len(), 1);

        let (track, cost) = solutions.swap_remove(0);
        assert_eq!(track.len(), 3);
        assert_eq!(cost, 2. * cost_array_fill);
        assert_eq!(track[0], start[0]);

        let &ArrayIndex { i: ei, j: ej } = track.last().unwrap();
        assert!(endpoints.contains(&(ei, ej)));
    }

    #[test]
    #[allow(clippy::approx_constant)]
    // Due to truncation solution to handle f32 costs.
    fn routing_many_to_many() {
        let store_path = dataset::samples::constant_value_cost_zarr(1.);
        let cost_function =
            CostFunction::from_json(r#"{"cost_layers": [{"layer_name": "cost"}]}"#).unwrap();
        let mut simulation = Simulation::new(&store_path, cost_function, 250_000_000).unwrap();
        let start = vec![
            ArrayIndex { i: 1, j: 1 },
            ArrayIndex { i: 3, j: 3 },
            ArrayIndex { i: 5, j: 5 },
        ];
        let end = vec![
            ArrayIndex { i: 1, j: 2 },
            ArrayIndex { i: 4, j: 4 },
            ArrayIndex { i: 7, j: 7 },
        ];
        let solutions = simulation.scout(&start, end);
        dbg!(&solutions);
        assert_eq!(solutions.len(), 3);

        let expected_solution = vec![
            (ArrayIndex { i: 1, j: 2 }, 1.0),
            (ArrayIndex { i: 4, j: 4 }, 1.4142),
            (ArrayIndex { i: 4, j: 4 }, 1.4142),
        ];
        for ((track, cost), eep) in solutions.into_iter().zip(expected_solution) {
            assert_eq!(track.len(), 2);
            assert_eq!(*track.last().unwrap(), eep.0);
            assert_eq!(cost, eep.1);
        }
    }

    #[test]
    fn routing_many_to_one() {
        let store_path = dataset::samples::constant_value_cost_zarr(1.);
        let cost_function =
            CostFunction::from_json(r#"{"cost_layers": [{"layer_name": "cost"}]}"#).unwrap();
        let mut simulation = Simulation::new(&store_path, cost_function, 250_000_000).unwrap();
        let start = vec![ArrayIndex { i: 1, j: 1 }, ArrayIndex { i: 5, j: 5 }];
        let end = vec![ArrayIndex { i: 3, j: 3 }];
        let solutions = simulation.scout(&start, end);
        dbg!(&solutions);
        assert_eq!(solutions.len(), 2);

        for (track, cost) in solutions {
            assert_eq!(track.len(), 3);
            assert_eq!(cost, 2.8284);
            assert_eq!(*track.last().unwrap(), ArrayIndex { i: 3, j: 3 });
        }
    }

    #[test]
    fn test_routing_along_boundary() {
        use ndarray::Array2;

        let (ni, nj) = (4, 4);
        let (ci, cj) = (2, 2);

        let store_path = tempfile::TempDir::new().unwrap();

        let store: zarrs::storage::ReadableWritableListableStorage = std::sync::Arc::new(
            zarrs::filesystem::FilesystemStore::new(store_path.path())
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

        #[rustfmt::skip]
        let a = vec![1., 50.,  1., 1.,
                     1., 50., 50., 1.,
                     1., 50., 50., 1.,
                     1.,  1.,  1., 1.];

        let data: Array2<f32> =
            ndarray::Array::from_shape_vec((ni.try_into().unwrap(), nj.try_into().unwrap()), a)
                .unwrap();

        array
            .store_chunks_ndarray(
                &zarrs::array_subset::ArraySubset::new_with_ranges(&[0..(ni / ci), 0..(nj / cj)]),
                data,
            )
            .unwrap();

        let cost_function =
            CostFunction::from_json(r#"{"cost_layers": [{"layer_name": "cost"}]}"#).unwrap();
        let mut simulation = Simulation::new(&store_path, cost_function, 250_000_000).unwrap();

        let start = vec![ArrayIndex { i: 0, j: 0 }];
        let end = vec![ArrayIndex { i: 0, j: 2 }];
        let mut solutions = simulation.scout(&start, end);
        assert_eq!(solutions.len(), 1);

        let (track, cost) = solutions.swap_remove(0);
        // 4 straight moves + 3 diagonal moves
        assert_eq!(cost, 8.2426);

        let expected_track = vec![
            ArrayIndex { i: 0, j: 0 },
            ArrayIndex { i: 1, j: 0 },
            ArrayIndex { i: 2, j: 0 },
            ArrayIndex { i: 3, j: 1 },
            ArrayIndex { i: 3, j: 2 },
            ArrayIndex { i: 2, j: 3 },
            ArrayIndex { i: 1, j: 3 },
            ArrayIndex { i: 0, j: 2 },
        ];
        assert_eq!(track, expected_track);
    }
}
