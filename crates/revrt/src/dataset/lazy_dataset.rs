//! Lazy loading of a subset of the source Dataset
//!
//! This was originally developed to support the cost calculation,
//! where the variables that will be used are not known until the
//! cost is actually computed, and the same variable may be used
//! multiple times.
//!
//! The subset is fixed at the time of creation, so all variables
//! are consistent.
//!
//! Before we used the LazyChunk, which assumed that the intended
//! outcome would match the source chunks. Here, the LazyDataset,
//! is defined with a Subset instead, giving us more flexibility.

use std::collections::HashMap;
use std::fmt;

use tracing::trace;
use zarrs::array::{Array, ElementOwned};
use zarrs::array_subset::ArraySubset;
use zarrs::storage::ReadableListableStorage;

pub(crate) struct LazyDataset<T> {
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

impl<T> fmt::Display for LazyDataset<T> {
    // Add information on the source and the data HashMap.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "LazyDataset {{ subset: {:?}, ... }}", self.subset,)
    }
}
impl<T: ElementOwned> LazyDataset<T> {
    /// Create a new LazyDataset with the given source and subset.
    pub(super) fn new(source: ReadableListableStorage, subset: ArraySubset) -> Self {
        trace!("Creating LazyDataset for subset: {:?}", subset);

        LazyDataset {
            source,
            subset,
            data: HashMap::new(),
        }
    }

    pub(crate) fn subset(&self) -> &ArraySubset {
        &self.subset
    }

    /// Get a data subset for the given variable name.
    pub(crate) fn get(
        &mut self,
        varname: &str,
    ) -> Option<ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Dim<ndarray::IxDynImpl>>> {
        trace!("Getting data subset for variable: {}", varname);

        let data = match self.data.get(varname) {
            Some(v) => {
                trace!("Data for variable {} already loaded", varname);
                v.clone()
            }
            None => {
                trace!(
                    "Loading data subset ({:?}) for variable: {}",
                    self.subset,
                    varname
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

        Some(data)
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
        let mut dataset = LazyDataset::<f32>::new(store, subset);
        let _tmp = dataset.get("A").unwrap();
    }

    /*
    #[test]
    fn test_lazy_dataset() {
        let storage = MemoryStore::new();
        let subset = ArraySubset::default();
        let mut lazy_dataset = LazyDataset::<f32>::new(Arc::new(storage), subset);

        if let Some(data) = lazy_dataset.get("test_var") {
            assert!(!data.is_empty());
        } else {
            panic!("Failed to retrieve data for 'test_var'");
        }
    }
    */
}
