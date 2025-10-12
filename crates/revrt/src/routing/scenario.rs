//! A scenario for routing
//!
//! A `Scenario` encapsulates the cost surface and eveything required to
//! define that, such as the input features and the cost function.
//!
//! A typical scenario uses a cost function that weights several
//! features to determine the cost per grid point.
//!
//! Given the relatively high resolution and the desire on routing long
//! distances, the data involved can be relatively large. A major
//! accomplishment of this crate is in this module, by working in chunks
//! with asynchronous I/O we keep the memory footprint low while
//! sustaining high performance as possible.

use tracing::trace;

use super::Features;
use crate::{ArrayIndex, Result};

const PRECISION_SCALAR: f32 = 1e4;

fn cost_as_u64(cost: f32) -> u64 {
    let cost = cost * PRECISION_SCALAR;
    cost as u64
}

#[allow(dead_code)]
fn unscaled_cost(cost: u64) -> f32 {
    (cost as f32) / PRECISION_SCALAR
}

pub(super) struct Scenario {
    pub dataset: crate::dataset::Dataset,
    #[allow(dead_code)]
    features: Features,
    #[allow(dead_code)]
    cost_function: crate::CostFunction,
}

impl Scenario {
    pub(super) fn new<P: AsRef<std::path::Path>>(
        store_path: P,
        cost_function: crate::cost::CostFunction,
        cache_size: u64,
    ) -> Result<Self> {
        let features = Features::new(&store_path)?;
        let dataset = crate::dataset::Dataset::open(store_path, cost_function.clone(), cache_size)?;

        Ok(Self {
            dataset,
            features,
            cost_function,
        })
    }

    pub(super) fn get_3x3(&self, position: &ArrayIndex) -> Vec<(ArrayIndex, f32)> {
        self.dataset.get_3x3(position)
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
    pub(super) fn successors(&self, position: &ArrayIndex) -> Vec<(ArrayIndex, u64)> {
        trace!("Position {:?}", position);
        let neighbors = self.get_3x3(position);
        let neighbors = neighbors
            .into_iter()
            .map(|(p, c)| (p, cost_as_u64(c))) // ToDo: Maybe it's better to have get_3x3 return a u64 - then we can skip this map altogether
            .collect();
        trace!("Adjusting neighbors' types: {:?}", neighbors);
        neighbors
    }
}
