use tracing::{debug, trace};

use crate::ArrayIndex;
use crate::error::Result;

pub(super) struct Simulation {
    scenario: Scenario,
    // algorithm: Algorithm,
}

impl Simulation {
    pub(super) fn compute(
        &mut self,
        start: &[ArrayIndex],
        end: Vec<ArrayIndex>,
    ) -> impl Iterator<Item = (Vec<ArrayIndex>, f32)> {
        self.scout(start, end).into_iter()
    }

    const PRECISION_SCALAR: f32 = 1e4;

    pub(super) fn new<P: AsRef<std::path::Path>>(
        store_path: P,
        cost_function: crate::cost::CostFunction,
        cache_size: u64,
    ) -> Result<Self> {
        let scenario = Scenario::new(store_path, cost_function, cache_size)?;

        Ok(Self {
            scenario,
            // algorithm,
        })
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
        let neighbors = self.scenario.get_3x3(position);
        let neighbors = neighbors
            .into_iter()
            .map(|(p, c)| (p, cost_as_u64(c))) // ToDo: Maybe it's better to have get_3x3 return a u64 - then we can skip this map altogether
            .collect();
        trace!("Adjusting neighbors' types: {:?}", neighbors);
        neighbors
    }

    pub(super) fn scout(
        &mut self,
        start: &[ArrayIndex],
        end: Vec<ArrayIndex>,
    ) -> Vec<(Vec<ArrayIndex>, f32)> {
        debug!("Starting scout with {} start points", start.len());

        start
            .into_iter()
            .filter_map(|s| {
                pathfinding::prelude::dijkstra(s, |p| self.successors(p), |p| end.contains(p))
            })
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

struct Solution {}

struct Scenario {
    dataset: crate::dataset::Dataset,
    // features: Features,
    cost_function: crate::CostFunction,
}

impl Scenario {
    fn new<P: AsRef<std::path::Path>>(
        store_path: P,
        cost_function: crate::cost::CostFunction,
        cache_size: u64,
        // features: Features,
        // cost_function: CostFunction,
    ) -> Result<Self> {
        let dataset = crate::dataset::Dataset::open(store_path, cost_function.clone(), cache_size)?;

        Ok(Self {
            dataset,
            // features,
            cost_function,
        })
    }

    fn get_3x3(&self, position: &ArrayIndex) -> Vec<(ArrayIndex, f32)> {
        self.dataset.get_3x3(position)
    }
}

struct Features {}

struct Algorithm {}
