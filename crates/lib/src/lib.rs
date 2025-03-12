//! # Path optimization with weigthed costs
//!
//!

mod error;

use pathfinding::prelude::dijkstra;

use error::Result;

struct Dataset {
    store: zarrs::storage::ReadableListableStorage,
    cache: zarrs::array::ChunkCacheLruChunkLimit<zarrs::array::ChunkCacheTypeDecoded>,
}

impl Dataset {
    fn new<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let filesystem = zarrs::filesystem::FilesystemStore::new(path).expect("could not open filesystem store");

        let store: zarrs::storage::ReadableListableStorage = std::sync::Arc::new(filesystem);
        // ATENTION: Hardcoded in ~250MB
        let cache = zarrs::array::ChunkCacheLruChunkLimit::new(250_000_000);

        Ok(Self { store, cache })
    }

    /// Determine the successors of a position.
    ///
    /// ToDo:
    /// - Handle the edges of the array.
    /// - Include diagonal moves.
    /// - Weight the cost. Remember that the cost is for a side,
    ///   thus a diagnonal move has to calculate consider the longer
    ///   distance.
    fn successors(&self, position: &Pos) -> Vec<(Pos, u64)> {
        let &Pos(x, y) = position;
        let array = zarrs::array::Array::open(self.store.clone(), "/A").unwrap();


        // Cutting off the edges for now.
        let shape = array.shape();
        if x == 0 || x >= (shape[0]) || y == 0 || y >= (shape[1]) {
            return vec![];
        }

        let subset = zarrs::array_subset::ArraySubset::new_with_ranges(&[(x-1)..(x+2), (y-1)..(y+2)]);

        //let value = array.retrieve_array_subset_ndarray::<f64>(&subset).unwrap();
        let value = array.retrieve_array_subset_elements::<f64>(&subset).unwrap();
        dbg!(&value);

        let output = vec![
            (Pos(x-1, y-1), (value[0] * 1e4) as u64),
            (Pos(x, y-1), (value[1]* 1e4) as u64),
            (Pos(x+1, y-1), (value[2]* 1e4) as u64),
            (Pos(x-1, y), (value[3]* 1e4) as u64),
            (Pos(x+1, y), (value[5]* 1e4) as u64),
            (Pos(x-1, y+1), (value[6]* 1e4) as u64),
            (Pos(x, y+1), (value[7]* 1e4) as u64),
            (Pos(x+1, y+1), (value[8]* 1e4) as u64),
        ];
        output


    }
}


#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct Pos(u64, u64);

}
