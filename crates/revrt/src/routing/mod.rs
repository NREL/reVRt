mod features;
mod scenario;

use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use tracing::{debug, trace};

use crate::ArrayIndex;
use crate::Solution;
use crate::error::Result;
use features::Features;
use scenario::Scenario;

pub(super) struct Routing {
    scenario: Scenario,
    // algorithm: Algorithm,
}

impl Routing {
    pub(super) fn compute(
        &mut self,
        start: &[ArrayIndex],
        end: Vec<ArrayIndex>,
    ) -> impl Iterator<Item = Solution<ArrayIndex, f32>> {
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
    ) -> Vec<Solution<ArrayIndex, f32>> {
        debug!("Starting scout with {} start points", start.len());

        start
            .into_par_iter()
            .filter_map(|s| {
                pathfinding::prelude::dijkstra(s, |p| self.successors(p), |p| end.contains(p))
            })
            .map(|(route, total_cost)| Solution::new(route, unscaled_cost(total_cost)))
            .collect()
    }
}

fn cost_as_u64(cost: f32) -> u64 {
    let cost = cost * Routing::PRECISION_SCALAR;
    cost as u64
}

fn unscaled_cost(cost: u64) -> f32 {
    (cost as f32) / Routing::PRECISION_SCALAR
}

// struct Algorithm {}
