//! Algorithms to find optimal path
//!
//! A collection of different strategies to find optimal paths.
//! Common algorithms are based on the external crate `pathfinding`.

/*
 * pathfinding::dijkstra(start, successor, success)
 * pathfinging::astar(start, successor, heuristic, success)
 * pathfinding::dfs(start, successor, success)
 */

use num_traits::Zero;
use std::hash::Hash;

use crate::{ArrayIndex, Solution};

#[derive(Clone, Debug)]
/// Types of algorithms to determine optimal paths
pub(super) enum AlgorithmType {
    // Astar,
    Dijkstra,
    // LongRangeDijkstra,
}

#[derive(Debug)]
pub(super) struct Algorithm {
    algorithm_type: AlgorithmType,
}

#[allow(dead_code)]
fn manhattan_distance(from: &ArrayIndex, to: &ArrayIndex) -> u64 {
    let ArrayIndex { i: i1, j: j1 } = from;
    let ArrayIndex { i: i2, j: j2 } = to;

    let di = if i1 > i2 { i1 - i2 } else { i2 - i1 };
    let dj = if j1 > j2 { j1 - j2 } else { j2 - j1 };
    di + dj
}

impl Algorithm {
    pub(super) fn new() -> Self {
        Self {
            algorithm_type: AlgorithmType::Dijkstra,
        }
    }

    #[allow(unused_variables)]
    pub(super) fn compute<I, C, FN, IN, FH, FS>(
        &self,
        start: &I,
        successors: FN,
        heuristic: Option<FH>,
        success: FS,
    ) -> Option<Solution<I, f32>>
    //) -> Option<Solution<I, C>>
    where
        I: Eq + Hash + Clone,
        C: Zero + Ord + Copy,
        FN: FnMut(&I) -> IN,
        IN: IntoIterator<Item = (I, C)>,
        FH: FnMut(&I) -> C,
        FS: FnMut(&I) -> bool,
        // Temporary solution while we can't compare f32
        u64: From<C>,
    {
        let ans = match self.algorithm_type {
            AlgorithmType::Dijkstra => pathfinding::prelude::dijkstra(start, successors, success),
        };

        match ans {
            Some((route, total_cost)) => Some(Solution::new(
                route,
                super::unscaled_cost(u64::from(total_cost)),
            )),
            None => None,
        }
    }
}
