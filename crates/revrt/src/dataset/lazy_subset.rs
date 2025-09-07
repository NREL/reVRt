//! Lazy load a subset of the source Dataset
//!
//! This was originally developed to support the cost calculation, where the
//! variables that will be used are not known until the cost is actually
//! computed, and the same variable may be used multiple times. Thus the goal
//! is to load each variable only once, don't load unecessary variables.
//!
//! The subset is fixed at the time of creation, so all variables are
//! consistent for the same domain.
//!
//! Before we used the LazyChunk, which assumed that the intended outcome
//! would match the source chunks. Here we replace that concept by a
//! `LazySubset`, which is tied to an `ArraySubset`, thus it has no assumtions
//! on the source's chunk. Therefore, the source can have variable chunk shapes,
//! one for each variable, and don't need to match the desired cost chunk shape.
//!
//! Note that we could have used Zarrs' intrinsic cache here, but a common
//! use for LazySubset is to load the features to compute cost for a chunk.
//! Therefore, those chunks of features are loaded only once and we don't
//! expect to use that anymore since we save the resulted cost. Using Zarrs'
//! cache would lead to unnecessary memory usage. Another problem is how
//! large should be that cache? It gets more difficult to estimate once we
//! consider the possibility of multiple threads working on different chunks.

use std::collections::HashMap;
use std::fmt;

use tracing::trace;
use zarrs::array::{Array, ElementOwned};
use zarrs::array_subset::ArraySubset;
use zarrs::storage::ReadableListableStorage;

use crate::error::Result;

/// Lazy loaded subset of a Zarr Dataset.
///
/// This struct is intended to work as a cache for a subset of a Zarr
/// Dataset.
pub(crate) struct LazySubset<T> {
    /// Source Zarr storage
    source: ReadableListableStorage,
    /// Subset of the source to be lazily loaded
    subset: ArraySubset,
    /// Data
    data: HashMap<
        String,
        ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>>,
    >,
}

impl<T> fmt::Display for LazySubset<T> {
    /// Display a LazySubset.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Add information on the source and the data HashMap.
        write!(f, "LazySubset {{ subset: {:?}, ... }}", self.subset,)
    }
}
impl<T: ElementOwned> LazySubset<T> {
    /// Create a new LazySubset for a given source and subset.
    pub(super) fn new(source: ReadableListableStorage, subset: ArraySubset) -> Self {
        trace!("Creating LazySubset for subset: {:?}", subset);

        LazySubset {
            source,
            subset,
            data: HashMap::new(),
        }
    }

    /// Show the subset used by this LazySubset.
    pub(crate) fn subset(&self) -> &ArraySubset {
        &self.subset
    }

    /// Get a data for a specific variable.
    pub(crate) fn get(
        &mut self,
        varname: &str,
    ) -> Result<ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>>> {
        trace!("Getting data subset for variable: {}", varname);

        let data = match self.data.get(varname) {
            Some(v) => {
                trace!("Data for variable {} already loaded", varname);
                v.clone()
            }
            None => {
                trace!(
                    "Loading data subset ({:?}) for variable: {}",
                    self.subset, varname
                );

                let variable = Array::open(self.source.clone(), &format!("/{varname}"))
                    .expect("Failed to open variable");

                let values = variable
                    .retrieve_array_subset_ndarray(&self.subset)
                    .expect("Failed to retrieve array subset");

                self.data.insert(varname.to_string(), values.clone());

                values
            }
        };

        Ok(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::samples;
    use std::sync::Arc;
    // use zarrs::storage::store::MemoryStore;
    use zarrs::storage::ReadableListableStorage;

    #[test]
    fn sample() {
        let path = samples::multi_variable_zarr();
        let store: ReadableListableStorage =
            Arc::new(zarrs::filesystem::FilesystemStore::new(&path).unwrap());

        let subset = ArraySubset::new_with_start_shape(vec![0, 0], vec![2, 2]).unwrap();
        let mut dataset = LazySubset::<f32>::new(store, subset);
        let tmp = dataset.get("A").unwrap();
        assert_eq!(tmp.shape(), &[2, 2]);
    }

    /*
    #[test]
    fn test_lazy_dataset() {
        let storage = MemoryStore::new();
        let subset = ArraySubset::default();
        let mut lazy_dataset = LazySubset::<f32>::new(Arc::new(storage), subset);

        if let Some(data) = lazy_dataset.get("test_var") {
            assert!(!data.is_empty());
        } else {
            panic!("Failed to retrieve data for 'test_var'");
        }
    }
    */
}
