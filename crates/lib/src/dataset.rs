
use tracing::trace;
 use zarrs::array::ArrayChunkCacheExt;
use zarrs::storage::{ReadableListableStorage, ReadableWritableListableStorage};

use crate::Point;
use crate::error::Result;

struct Dataset {
    source: ReadableListableStorage,
    cost: ReadableWritableListableStorage,
    cost_chunk_idx: ndarray::Array2<bool>,
    /// Cache for the cost
    cache: zarrs::array::ChunkCacheLruSizeLimit<zarrs::array::ChunkCacheTypeDecoded>,
}

impl Dataset {
    fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let filesystem = zarrs::filesystem::FilesystemStore::new(path).unwrap();
        let source= std::sync::Arc::new(filesystem);

        // ==== Create the cost dataset ====
        let tmp_path = tempfile::TempDir::new().unwrap();
        let tmp_path = std::path::PathBuf::from("./cost_demo.zarr");
        let cost: ReadableWritableListableStorage = std::sync::Arc::new(
            zarrs::filesystem::FilesystemStore::new(tmp_path.as_path())
                .expect("could not open filesystem store"),
        );

        zarrs::group::GroupBuilder::new()
            .build(cost.clone(), "/")
            .unwrap()
            .store_metadata()
            .unwrap();

        let array = zarrs::array::ArrayBuilder::new(
            vec![8, 8], // array shape
            zarrs::array::DataType::Float32,
            vec![4, 4].try_into().unwrap(), // regular chunk shape
            zarrs::array::FillValue::from(zarrs::array::ZARR_NAN_F32),
        ).build(cost.clone(), "/cost").unwrap();
        array.store_metadata().unwrap();

        let cost_chunk_idx = ndarray::Array2::from_elem((2, 2), false);

        let cache = zarrs::array::ChunkCacheLruSizeLimit::new(250_000_000);

        Ok(Self { source, cost, cost_chunk_idx, cache })
    }

    fn calculate_chunk_cost(&self, i: u64, j: u64) {
                dbg!(i);
                dbg!(j);
        let array = zarrs::array::Array::open(self.source.clone(), "/A").unwrap();
        let value = array.retrieve_chunk_ndarray::<f32>(&[i, j]).unwrap();
                dbg!(&value);
                //let value = array.retrieve_chunk(&[i, j]).unwrap();
        let output = value * 10.0;
                dbg!(&output);
        let cost = zarrs::array::Array::open(self.cost.clone(), "/cost").unwrap();
        cost.store_metadata().unwrap();
        let chunk_indices: Vec<u64> = vec![i, j];
                dbg!(&chunk_indices);
                // let chunk_subset = array.chunk_grid().subset(&chunk_indices, array.shape()).unwrap().unwrap();
                //dbg!(&chunk_subset);
        let chunk_subset = &zarrs::array_subset::ArraySubset::new_with_ranges(&[i..(i+1), j..(j+1)]);
                dbg!(&chunk_subset);
                // array.store_chunk_elements(&chunk_indices
        cost.store_chunks_ndarray(&chunk_subset, output).unwrap();
    }

    fn get_3x3(&mut self, x: u64, y: u64) -> Vec<(Point, f32)>{
        let cost = zarrs::array::Array::open(self.cost.clone(), "/cost").unwrap();

        let subset = zarrs::array_subset::ArraySubset::new_with_ranges(&[(x-1)..(x + 2), (y-1)..(y + 2)]);
        dbg!(&subset);

        // Find the chunks that intersect the subset
        let chunks = &cost.chunks_in_array_subset(&subset).unwrap().unwrap();
        dbg!(&chunks);
        dbg!(chunks.num_elements_usize());
        dbg!(chunks.start());
        dbg!(&self.cost_chunk_idx);
        for i in chunks.start()[0]..(chunks.start()[0] + chunks.shape()[0]) {
            for j in chunks.start()[0]..(chunks.start()[0] + chunks.shape()[1]) {
                if self.cost_chunk_idx[[i as usize, j as usize]] {
                    continue;
                }
                self.calculate_chunk_cost(i, j);
                dbg!(&self.cost_chunk_idx[[i as usize, j as usize]]);
                self.cost_chunk_idx[[i as usize, j as usize]] = true;
                dbg!(&self.cost_chunk_idx[[i as usize, j as usize]]);
            }
        }

        // Retrieve the 3x3 neighborhood values
        let value: Vec<f32> = cost.retrieve_array_subset_elements_opt_cached::<f32, zarrs::array::ChunkCacheTypeDecoded>(&self.cache, &subset, &zarrs::array::codec::CodecOptions::default()).unwrap();
        dbg!(&value);

        trace!("Read values {:?}", value);

          let neighbors = vec![
              (Point(x - 1, y - 1), value[0]),
              (Point(x, y - 1), value[1] ),
              (Point(x + 1, y - 1), value[2] ),
              (Point(x - 1, y), value[3] ),
              (Point(x + 1, y), value[5] ),
              (Point(x - 1, y + 1), value[6] ),
              (Point(x, y + 1), value[7] ),
              (Point(x + 1, y + 1), value[8] ),
          ];
          trace!("Neighbors {:?}", neighbors);

          neighbors


        /*
        let mut data = array
            .load_chunks_ndarray(&zarrs::array_subset::ArraySubset::new_with_ranges(&[0..2, 0..2]))
            .unwrap();
        data[[x as usize, y as usize]] = 0.0;
        array
            .store_chunks_ndarray(
                &zarrs::array_subset::ArraySubset::new_with_ranges(&[0..2, 0..2]),
                data,
            )
            .unwrap();
        */

    }
}

#[cfg(test)]
pub(crate) mod samples {
    use ndarray::Array2;
    use rand::Rng;

    /// Create a single variable (/A) zarr store
    ///
    /// Just a proof of concept with lots of hardcoded values
    /// that must be improved.
    pub(crate) fn single_variable_zarr() -> std::path::PathBuf {
        let ni = 8;
        let nj = 8;

        let tmp_path = tempfile::TempDir::new().unwrap();

        let store: zarrs::storage::ReadableWritableListableStorage = std::sync::Arc::new(
            zarrs::filesystem::FilesystemStore::new(tmp_path.path())
                .expect("could not open filesystem store"),
        );

        zarrs::group::GroupBuilder::new()
            .build(store.clone(), "/")
            .unwrap()
            .store_metadata()
            .unwrap();

        // Create an array
        let array_path = "/A";
        let array = zarrs::array::ArrayBuilder::new(
            vec![ni, nj], // array shape
            zarrs::array::DataType::Float32,
            vec![4, 4].try_into().unwrap(), // regular chunk shape
            zarrs::array::FillValue::from(zarrs::array::ZARR_NAN_F32),
        )
        // .bytes_to_bytes_codecs(vec![]) // uncompressed
        .dimension_names(["y", "x"].into())
        // .storage_transformers(vec![].into())
        .build(store.clone(), array_path)
        .unwrap();

        // Write array metadata to store
        array.store_metadata().unwrap();

        let mut rng = rand::rng();
        let mut a = vec![];
        for _x in 0..(ni * nj) {
            a.push(rng.random_range(0.0..=1.0));
        }
        let data: Array2<f32> =
            ndarray::Array::from_shape_vec((ni.try_into().unwrap(), nj.try_into().unwrap()), a)
                .unwrap();

        array
            .store_chunks_ndarray(
                &zarrs::array_subset::ArraySubset::new_with_ranges(&[0..2, 0..2]),
                data,
            )
            .unwrap();

        tmp_path.into_path()
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_variable_zarr() {
        let path = samples::single_variable_zarr();
        let mut dataset = Dataset::open(path).unwrap();

        let results = dataset.get_3x3(3, 2);
        dbg!(&results);
        let results = dataset.get_3x3(2, 2);
        dbg!(&results);
    }
}
