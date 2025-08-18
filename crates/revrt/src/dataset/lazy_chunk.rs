use tracing::trace;
use zarrs::storage::ReadableListableStorage;

use crate::error::Result;

/// Lazy chunk of a Zarr dataset
pub(crate) struct LazyChunk {
    /// Source Zarr storage
    pub(crate) source: ReadableListableStorage,
    /// Chunk index 1st dimension
    pub(crate) ci: u64,
    /// Chunk index 2nd dimension
    pub(crate) cj: u64,
    /// Data
    // We know it is a 2D array of f32. We might want to simplify and strict this definition.
    // data: std::collections::HashMap<String, ndarray::Array2<f32>>,
    pub(crate) data: std::collections::HashMap<
        String,
        ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<ndarray::IxDynImpl>>,
    >,
}

impl LazyChunk {
    pub(crate) fn ci(&self) -> u64 {
        self.ci
    }

    pub(crate) fn cj(&self) -> u64 {
        self.cj
    }

    //fn get(&self, variable: &str) -> Result<&ndarray::Array2<f32>> {
    pub(crate) fn get(
        &mut self,
        variable: &str,
    ) -> Result<ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<ndarray::IxDynImpl>>> {
        trace!("Getting chunk data for variable: {}", variable);

        Ok(match self.data.get(variable) {
            Some(v) => {
                trace!("Chunk data for variable {} already loaded", variable);
                v.clone()
            }
            None => {
                trace!("Loading chunk data for variable: {}", variable);
                let array = zarrs::array::Array::open(self.source.clone(), &format!("/{variable}"))
                    .unwrap();
                let chunk_indices = &[self.ci, self.cj];
                let chunk_subset = zarrs::array_subset::ArraySubset::new_with_ranges(&[
                    chunk_indices[0]..(chunk_indices[0] + 1),
                    chunk_indices[1]..(chunk_indices[1] + 1),
                ]);
                trace!("Storing chunk data for variable: {}", variable);
                let values = array.retrieve_chunks_ndarray::<f32>(&chunk_subset).unwrap();
                // array.retrieve_chunk_ndarray::<f32>(&[ci, cj]).unwrap();
                self.data.insert(variable.to_string(), values.clone());
                values
            }
        })
    }
}

#[cfg(test)]
mod chunk_tests {
    use super::*;
    use crate::dataset::samples;
    use std::path::PathBuf;

    #[test]
    fn dev() {
        let path: PathBuf = samples::multi_variable_zarr();
        let store: zarrs::storage::ReadableListableStorage =
            std::sync::Arc::new(zarrs::filesystem::FilesystemStore::new(&path).unwrap());
        let mut chunk = LazyChunk {
            source: store,
            ci: 0,
            cj: 0,
            data: std::collections::HashMap::new(),
        };

        assert_eq!(chunk.ci, 0);
        assert_eq!(chunk.cj, 0);

        let _tmp = chunk.get("A").unwrap();
    }
}
