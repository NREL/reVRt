#![allow(dead_code)]
use std::path::PathBuf;

use pyo3::exceptions::{PyException, PyIOError};
use pyo3::prelude::*;

use crate::error::{Error, Result};
use crate::{ArrayIndex, resolve};

pyo3::create_exception!(_rust, revrtRustError, PyException);

impl From<Error> for PyErr {
    fn from(err: Error) -> PyErr {
        match err {
            Error::IO(msg) => PyIOError::new_err(msg),
            Error::ZarrsArrayCreate(e) => PyIOError::new_err(e.to_string()),
            Error::ZarrsStorage(e) => PyIOError::new_err(e.to_string()),
            Error::ZarrsGroupCreate(e) => PyIOError::new_err(e.to_string()),
            Error::Undefined(msg) => revrtRustError::new_err(msg),
        }
    }
}

/// A Python module implemented in Rust
#[pymodule]
fn _rust(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_paths, m)?)?;
    m.add("revrtRustError", py.get_type::<revrtRustError>())?;
    m.add_class::<Number>()?;
    Ok(())
}

/// Find least-cost paths for one or more starting points.
///
/// This function determined the least cost path for one or more starting
/// points to one or more ending points. A unique path is returned for
/// every starting point, but each route terminates when any of the ending
/// points are found. To ensure that a path is found to every end point,
/// call this function N times if you have N end points and pass a single
/// end point each time.
///
/// Parameters
/// ----------
/// zarr_fp : path-like
///     Path to zarr file containing cost layers.
/// cost_layers : str
///     JSON string representation of the cost function. The following
///     keys are allowed in the cost function: "cost_layers",
///     "friction_layers", and "ignore_invalid_costs". See the
///     documentation of the cost function for details on each of these
///     inputs.
/// start : list of tuple
///     List of two-tuples containing non-negative integers representing
///     the indices in the array for the pixel from which routing should
///     begin. A unique path will be returned for each of the starting
///     points.
/// end : list of tuple
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
/// list of tuple
///     List of path routing results. Each result is a tuple
///     where the first element is a list of points that the
///     route goes through and the second element is the final
///     route cost.
#[pyfunction]
#[pyo3(signature = (zarr_fp, cost_function, start, end, cache_size=250_000_000))]
#[allow(clippy::type_complexity)]
fn find_paths(
    zarr_fp: PathBuf,
    cost_function: String,
    start: Vec<(u64, u64)>,
    end: Vec<(u64, u64)>,
    cache_size: u64,
) -> Result<Vec<(Vec<(u64, u64)>, f32)>> {
    let start: Vec<ArrayIndex> = start
        .into_iter()
        .map(|(i, j)| ArrayIndex { i, j })
        .collect();
    let end: Vec<ArrayIndex> = end.into_iter().map(|(i, j)| ArrayIndex { i, j }).collect();
    let paths = resolve(zarr_fp, &cost_function, cache_size, &start, end)?;
    Ok(paths
        .into_iter()
        .map(|(path, cost)| (path.into_iter().map(Into::into).collect(), cost))
        .collect())
}

/// A test docstring for a class.
#[pyclass]
struct Number {
    value: i32,
}

#[pyclass]
struct NumberIter {
    data: std::vec::IntoIter<i32>,
}

// use pyo3::types::PyString;

#[pymethods]
impl NumberIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<i32> {
        println!("Calling __next__ from Rust!");
        slf.data.next()
    }
}

#[pymethods]
impl Number {
    #[new]
    #[pyo3(signature = (value=100))]
    fn new(value: i32) -> Self {
        Number { value }
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("Number({})", self.value))
    }

    // fn __iter__<'py>(slf: PyRef<'py, Self>) -> PyResult<pyo3::Bound<'py, pyo3::types::PyIterator>> {
    //     PyString::new(slf.py(), "hello, world").try_iter()
    // }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<NumberIter>> {
        Py::new(
            slf.py(),
            NumberIter {
                data: vec![1, 2, 3, 4, 5].into_iter(),
            },
        )
    }
}
