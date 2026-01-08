//! Algorithms to find optimal path
//!
//! A collection of different strategies to find optimal paths.
//! Common algorithms are based on the external crate `pathfinding`.

/*
 * pathfinding::dijkstra(start, successor, success)
 * pathfinding::astar(start, successor, heuristic, success)
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
/// Manhattan distance
///
/// For a given start point, calculates the shortest manhattan distance to a
/// collection of possible end points, i.e. assume that there are multiple
/// possible ends.
fn manhattan_distance(start: &ArrayIndex, end: &[ArrayIndex]) -> u64 {
    end.iter()
        .map(|end| {
            let di = start.i.abs_diff(end.i);
            let dj = start.j.abs_diff(end.j);
            di + dj
        })
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
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

        ans.map(|(route, total_cost)| {
            Solution::new(route, super::unscaled_cost(u64::from(total_cost)))
        })
    }
}
