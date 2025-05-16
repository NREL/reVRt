use std::path::PathBuf;

use pyo3::exceptions::{PyException, PyIOError};
use pyo3::prelude::*;

use crate::error::{Error, Result};
use crate::{Point, resolve};

pyo3::create_exception!(reVRt, reVRtRustError, PyException);

impl From<Error> for PyErr {
    fn from(err: Error) -> PyErr {
        match err {
            Error::IO(msg) => PyIOError::new_err(msg),
            Error::Undefined(msg) => reVRtRustError::new_err(msg),
        }
    }
}

/// A Python module implemented in Rust
#[pymodule]
fn reVRt(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_path, m)?)?;
    m.add("reVRtRustError", py.get_type::<reVRtRustError>())?;
    Ok(())
}

/// A Python module implemented in Rust
#[pyfunction]
fn find_path(
    zarr_fp: PathBuf,
    cost_layers: String,
    start: (u64, u64),
    end: Vec<(u64, u64)>,
    cache_size: u64,
) -> Result<()> {
    let end: Vec<Point> = end.into_iter().map(Into::into).collect();
    println!("Hello from Rust!");
    println!("Got zarr path: {:?}", &zarr_fp);
    println!("Got cost layers: {:?}", &cost_layers);
    println!("Got stating point: {:?}", &start);
    println!("Got ending points: {:?}", &end);
    println!("Got cache size: {:?}", &cache_size);
    Err(Error::Undefined("This is an error".into()))
    // Ok(())
}
