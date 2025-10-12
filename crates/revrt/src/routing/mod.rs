mod features;
mod scenario;

use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use tracing::debug;

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

    pub(super) fn scout(
        &mut self,
        start: &[ArrayIndex],
        end: Vec<ArrayIndex>,
    ) -> Vec<Solution<ArrayIndex, f32>> {
        debug!("Starting scout with {} start points", start.len());

        start
            .into_par_iter()
            .filter_map(|s| {
                pathfinding::prelude::dijkstra(
                    s,
                    |p| self.scenario.successors(p),
                    |p| end.contains(p),
                )
            })
            .map(|(route, total_cost)| Solution::new(route, unscaled_cost(total_cost)))
            .collect()
    }
}

const PRECISION_SCALAR: f32 = 1e4;
fn cost_as_u64(cost: f32) -> u64 {
    let cost = cost * PRECISION_SCALAR;
    cost as u64
}

fn unscaled_cost(cost: u64) -> f32 {
    (cost as f32) / PRECISION_SCALAR
}

// struct Algorithm {}
