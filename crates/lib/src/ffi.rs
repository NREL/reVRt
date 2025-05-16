use std::path::PathBuf;

use pyo3::exceptions::{PyException, PyIOError};
use pyo3::prelude::*;

use crate::error::{Error, Result};
use crate::{Point, resolve};

pyo3::create_exception!(_rust, reVRtRustError, PyException);

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
fn _rust(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_paths, m)?)?;
    m.add("reVRtRustError", py.get_type::<reVRtRustError>())?;
    Ok(())
}

/// A Python module implemented in Rust
///
/// Testing this
///
/// Parameters
/// ----------
/// zarr_fp : path-like
///     Path to zarr file containing cost layers.
/// cost_layers : str
///     JSON string representation of a list of dictionaries
///     that define the cost layer computation. Each dictionary
///     must have at least one key: "layer_name". This key points
///     to the layer in the Zarr file that should be read in. The
///     other optional keys are "multiplier_layer", which is the
///     name of a layer in the Zarr file that should be multiplied
///     onto the original layer and "multiplier_scalar", which
///     should be a float that should be applied to scale the layer.
///     All of the layers in the list are processed this way and then
///     summed to obtain the final cost layer for routing.
/// start : list of (int, int)
///     List of two-tuples containing non-negative integers representing
///     the indices in the array for the pixel from which routing should
///     begin. A unique path will be returned for each of the starting
///     points.
/// end : list of (int, int)
///     List of two-tuples containing non-negative integers representing
///     the indices in the array for the any allowed final pixel.
///     When the algorithm reaches any of these points, the routing
///     is terminated and the final path + cost is returned.
/// cache_size : int, default=250_000_000
///     Cache size to use for computation, in bytes.
///     By default, `250,000,000` (250MB).
///
/// Returns
/// -------
/// list of tuples
///     List of path routing results. Each result is a tuple
///     where the first element is a list of points that the
///     route goes through and the second element is the final
///     route cost.
#[pyfunction]
#[pyo3(signature = (zarr_fp, cost_layers, start, end, cache_size=250_000_000))]
fn find_paths(
    zarr_fp: PathBuf,
    cost_layers: String,
    start: Vec<(u64, u64)>,
    end: Vec<(u64, u64)>,
    cache_size: u64,
) -> Result<Vec<(Vec<(u64, u64)>, usize)>> {
    let start: Vec<Point> = start.into_iter().map(Into::into).collect();
    let end: Vec<Point> = end.into_iter().map(Into::into).collect();
    let paths = resolve(zarr_fp, &cost_layers, cache_size, &start, end)?;
    Ok(paths
        .into_iter()
        .map(|(path, cost)| (path.into_iter().map(Into::into).collect(), cost))
        .collect())
}
