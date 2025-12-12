/// Possible errors

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    IO(#[from] std::io::Error),

    #[error(transparent)]
    ZarrsArrayCreate(#[from] zarrs::array::ArrayCreateError),

    #[error(transparent)]
    ZarrsGroupCreate(#[from] zarrs::group::GroupCreateError),

    #[error(transparent)]
    ZarrsStorage(#[from] zarrs::storage::StorageError),

    #[allow(dead_code)]
    #[error("Undefined error")]
    // Used during development while it is not clear a category of error
    // or when it is not worth to create a new error type.
    /// Undefined error
    Undefined(String),
}

pub(crate) type Result<T> = core::result::Result<T, Error>;
