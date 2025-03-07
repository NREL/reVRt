
mod error;

use pathfinding::prelude::dijkstra;

use error::Result;

struct Dataset {
    store: zarrs::storage::ReadableListableStorage,
}

impl Dataset {
    fn new<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let filesystem = zarrs::filesystem::FilesystemStore::new(path).expect("could not open filesystem store");

        let store: zarrs::storage::ReadableListableStorage = std::sync::Arc::new(filesystem);

        Ok(Self { store })
    }
}
