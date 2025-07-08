/// Support cost for a node in a network graph.
use std::cmp::Ordering;

use crate::ArrayIndex;

#[derive(Debug)]
/// Cost for a node of a network graph
///
/// Provides support to compare nodes based on their cost and estimated cost.
pub(super) struct NodeCost<T> {
    /// Estimated cost to reach the goal from this node.
    // This is not used now, but preparing for A* algorithm.
    pub(super) index: ArrayIndex,
    pub(super) cost: T,
    pub(super) estimated_cost: T,
}

impl<T: PartialEq> PartialEq for NodeCost<T> {
    fn eq(&self, other: &Self) -> bool {
        self.estimated_cost.eq(&other.estimated_cost) && self.cost.eq(&other.cost)
    }
}

impl<T: PartialEq> Eq for NodeCost<T> {}

impl<T: Ord> PartialOrd for NodeCost<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Ord> Ord for NodeCost<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        match other.estimated_cost.cmp(&self.estimated_cost) {
            Ordering::Equal => self.cost.cmp(&other.cost),
            s => s,
        }
    }
}

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
    fn integer() {
        use super::*;
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
        assert!(a > b);
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

    /*
    fn float() {
        let a = NodeCost {
            estimated_cost: 10.0,
            cost: 5.0,
            index: 0,
        };
        let b = NodeCost {
            estimated_cost: 10.0,
            cost: 3.0,
            index: 1,
        };
        assert!(a > b);
    }
    */

    #[test]
    fn binary_heap() {
        let mut heap = std::collections::BinaryHeap::new();
        heap.push(NodeCost {
            index: ArrayIndex::new(0, 0),
            cost: 5,
            estimated_cost: 5,
        });
        heap.push(NodeCost {
            index: ArrayIndex::new(1, 1),
            cost: 3,
            estimated_cost: 3,
        });
        assert_eq!(heap.pop().unwrap().index, ArrayIndex::new(1, 1));
        assert_eq!(heap.pop().unwrap().index, ArrayIndex::new(0, 0));
    }
}
