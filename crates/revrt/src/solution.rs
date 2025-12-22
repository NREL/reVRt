//! Routing solution representation

#[allow(missing_docs, dead_code)]
#[derive(Debug)]
/// Solution for one single routing case
///
pub(super) struct Solution<I, C> {
    pub(super) route: Vec<I>,
    pub(super) total_cost: C,
}

impl<I, C> Solution<I, C> {
    #[allow(dead_code, missing_docs)]
    pub(crate) fn new(route: Vec<I>, total_cost: C) -> Self {
        Self { route, total_cost }
    }

    #[allow(dead_code, missing_docs)]
    pub(super) fn route(&self) -> &Vec<I> {
        &self.route
    }

    #[allow(dead_code, missing_docs)]
    pub(super) fn total_cost(&self) -> &C {
        &self.total_cost
    }
}

pub type RevrtRoutingSolutions = Vec<Solution<crate::ArrayIndex, f32>>;
