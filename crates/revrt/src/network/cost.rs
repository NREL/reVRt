//! Support cost for a node in a network graph.

use std::cmp::Ordering;

use crate::ArrayIndex;

#[derive(Debug)]
/// Cost for a node in a network graph
///
/// Provides support to compare nodes based on their cost and estimated cost.
pub(super) struct NodeCost<T> {
    /// Estimated cost to reach the goal from this node.
    // This is not used now, but preparing for A* algorithm.
    pub(super) index: ArrayIndex,
    /// Cost (cheapest) to reach this node from the start node.
    pub(super) cost: T,
    /// Estimated cost to reach the goal from this node.
    pub(super) estimated_cost: T,
}

impl<T: PartialEq> PartialEq for NodeCost<T> {
    fn eq(&self, other: &Self) -> bool {
        self.estimated_cost.eq(&other.estimated_cost) && self.cost.eq(&other.cost)
    }
}

impl<T: PartialEq> Eq for NodeCost<T> {}

impl PartialOrd for NodeCost<i32> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialOrd for NodeCost<f32> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/*
impl<T: Ord> PartialOrd for NodeCost<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
*/

/// Priority order for the node cost.
///
/// Note that the order is reversed compared to the default ordering.
impl Ord for NodeCost<i32> {
    fn cmp(&self, other: &Self) -> Ordering {
        match other.estimated_cost.cmp(&self.estimated_cost) {
            Ordering::Equal => other.cost.cmp(&self.cost),
            s => s,
        }
    }
}
impl Ord for NodeCost<f32> {
    fn cmp(&self, other: &Self) -> Ordering {
        match other.estimated_cost.total_cmp(&self.estimated_cost) {
            Ordering::Equal => other.cost.total_cmp(&self.cost),
            s => s,
        }
    }
}
/*
impl<T: Ord> Ord for NodeCost<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        match other.estimated_cost.cmp(&self.estimated_cost) {
            Ordering::Equal => self.cost.partial_cmp(&other.cost).unwrap(),
            s => s,
        }
    }
}
*/

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn equal_integers() {
        let a = NodeCost {
            index: ArrayIndex::new(0, 0),
            cost: 5,
            estimated_cost: 10,
        };
        let b = NodeCost {
            index: ArrayIndex::new(1, 1),
            cost: 5,
            estimated_cost: 10,
        };
        assert!(a == b);
    }

    #[test]
    /// Order of estimated_cost leads to the comparison.
    fn gt_integer() {
        let a = NodeCost {
            index: ArrayIndex::new(0, 0),
            cost: 3,
            estimated_cost: 10,
        };
        let b = NodeCost {
            index: ArrayIndex::new(1, 1),
            cost: 5,
            estimated_cost: 7,
        };
        assert!(a < b);
    }

    #[test]
    fn cost_gt_integer() {
        let a = NodeCost {
            index: ArrayIndex::new(0, 0),
            cost: 5,
            estimated_cost: 10,
        };
        let b = NodeCost {
            index: ArrayIndex::new(1, 1),
            cost: 3,
            estimated_cost: 10,
        };
        assert!(a < b);
    }

    #[test]
    fn estimated_gt_integer() {
        let a = NodeCost {
            index: ArrayIndex::new(0, 0),
            cost: 3,
            estimated_cost: 10,
        };
        let b = NodeCost {
            index: ArrayIndex::new(1, 1),
            cost: 3,
            estimated_cost: 7,
        };
        assert!(a < b);
    }

    #[test]
    fn equal_floats() {
        let a = NodeCost {
            index: ArrayIndex::new(0, 0),
            cost: 5.0,
            estimated_cost: 10.0,
        };
        let b = NodeCost {
            index: ArrayIndex::new(1, 1),
            cost: 5.0,
            estimated_cost: 10.0,
        };
        assert!(a == b);
    }

    #[test]
    fn gt_float() {
        let a = NodeCost {
            index: ArrayIndex::new(0, 0),
            cost: 3.0,
            estimated_cost: 10.0,
        };
        let b = NodeCost {
            index: ArrayIndex::new(1, 1),
            cost: 5.0,
            estimated_cost: 7.0,
        };
        assert!(a < b);
    }

    #[test]
    fn cost_gt_float() {
        let a = NodeCost {
            index: ArrayIndex::new(0, 0),
            cost: 5.0,
            estimated_cost: 10.0,
        };
        let b = NodeCost {
            index: ArrayIndex::new(1, 1),
            cost: 3.0,
            estimated_cost: 10.0,
        };
        assert!(a < b);
    }

    #[test]
    fn binary_heap_integer() {
        let mut heap = std::collections::BinaryHeap::new();
        heap.push(NodeCost {
            index: ArrayIndex::new(0, 0),
            cost: 5,
            estimated_cost: 10,
        });
        heap.push(NodeCost {
            index: ArrayIndex::new(1, 0),
            cost: 3,
            estimated_cost: 10,
        });
        heap.push(NodeCost {
            index: ArrayIndex::new(0, 1),
            cost: 3,
            estimated_cost: 12,
        });
        assert_eq!(heap.pop().unwrap().index, ArrayIndex::new(1, 0));
        assert_eq!(heap.pop().unwrap().index, ArrayIndex::new(0, 0));
        assert_eq!(heap.pop().unwrap().index, ArrayIndex::new(0, 1));
    }

    #[test]
    fn binary_heap_float() {
        let mut heap = std::collections::BinaryHeap::new();
        heap.push(NodeCost {
            index: ArrayIndex::new(0, 0),
            cost: 5.0,
            estimated_cost: 10.0,
        });
        heap.push(NodeCost {
            index: ArrayIndex::new(1, 0),
            cost: 3.0,
            estimated_cost: 10.0,
        });
        heap.push(NodeCost {
            index: ArrayIndex::new(0, 1),
            cost: 3.0,
            estimated_cost: 12.0,
        });
        assert_eq!(heap.pop().unwrap().index, ArrayIndex::new(1, 0));
        assert_eq!(heap.pop().unwrap().index, ArrayIndex::new(0, 0));
        assert_eq!(heap.pop().unwrap().index, ArrayIndex::new(0, 1));
    }
}
