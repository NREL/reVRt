//! A scenario for routing
//!
//! A `Scenario` encapsulates the input features and the cost function
//! which defines how the input features affect the cost.
//!
//! Given the relatively high resolution and the desire on routing long
//! distances, the data involved can be relatively large. A major
//! accomplishment of this crate is in this module, by working in chunks
//! with asynchronous I/O we keep the memory footprint low while
//! sustaining high performance as possible.

use tracing::trace;

use super::Features;
use crate::{ArrayIndex, Result};

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
}
