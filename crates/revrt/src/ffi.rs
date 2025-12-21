#![allow(dead_code)]
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use pyo3::exceptions::{PyException, PyIOError};
use pyo3::prelude::*;

use rand::Rng;
use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::{ArrayIndex, resolve};

/// A multi-dimensional array representing cost data
type PythonRouteDefinition = (Vec<(u64, u64)>, Vec<(u64, u64)>);

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

// struct RouteDefinition {
//     start_inds: Vec<ArrayIndex>,
//     end_inds: Vec<ArrayIndex>,
// }

// impl From<RouteDefinition> for PythonRouteDefinition {
//     fn from(
//         RouteDefinition {
//             start_inds,
//             end_inds,
//         }: RouteDefinition,
//     ) -> PythonRouteDefinition {
//         (
//             start_inds
//                 .into_iter()
//                 .map(|ArrayIndex { i, j }| (i, j))
//                 .collect(),
//             end_inds
//                 .into_iter()
//                 .map(|ArrayIndex { i, j }| (i, j))
//                 .collect(),
//         )
//     }
// }

// impl From<PythonRouteDefinition> for RouteDefinition {
//     fn from((start_points, end_points): PythonRouteDefinition) -> RouteDefinition {
//         let start_inds = start_points
//             .into_iter()
//             .map(|(i, j)| ArrayIndex { i, j })
//             .collect();
//         let end_inds = end_points
//             .into_iter()
//             .map(|(i, j)| ArrayIndex { i, j })
//             .collect();

//         RouteDefinition {
//             start_inds,
//             end_inds,
//         }
//     }
// }

#[pyclass]
struct Number {
    // data: Arc<[i32]>,
    zarr_fp: PathBuf,
    cost_function: String,
    route_definitions: Vec<PythonRouteDefinition>,
    cache_size: u64,
}

#[pymethods]
impl Number {
    // #[new]
    // #[pyo3(signature = (values=None))]
    // fn new(values: Option<Vec<i32>>) -> PyResult<Self> {
    //     let data = values.unwrap_or_else(|| vec![1, 2, 3, 4, 5]);
    //     Ok(Self {
    //         data: Arc::from(data),
    //     })
    // }

    #[new]
    fn new(
        zarr_fp: PathBuf,
        cost_function: String,
        route_definitions: Vec<PythonRouteDefinition>,
        cache_size: u64,
    ) -> PyResult<Self> {
        Ok(Self {
            zarr_fp,
            cost_function,
            route_definitions,
            cache_size,
        })
        // Ok(Self {
        //     data: Arc::from(values),
        // })
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<NumberIter>> {
        // Py::new(slf.py(), NumberIter::new(Arc::clone(&slf.data)))
        Py::new(slf.py(), NumberIter::new(&slf))
    }

    // fn __str__(&self) -> PyResult<String> {
    //     Ok(format!("Number(len={})", self.data.len()))
    // }
}

#[pyclass(unsendable)]
struct NumberIter {
    // receiver: mpsc::Receiver<i32>,
    receiver: mpsc::Receiver<Result<Vec<(Vec<(u64, u64)>, f32)>>>,
    finished: bool,
}

impl NumberIter {
    // fn new(data: Arc<[i32]>) -> Self {
    fn new(scenario: &Number) -> Self {
        let (tx, rx) = mpsc::channel();
        let zarr_fp = scenario.zarr_fp.clone();
        let cost_function = scenario.cost_function.clone();
        let route_definitions = scenario.route_definitions.clone();
        let cache_size = scenario.cache_size;
        rayon::spawn(move || {
            //     fn find_paths(
            //     zarr_fp: PathBuf,
            //     cost_function: String,
            //     start: Vec<(u64, u64)>,
            //     end: Vec<(u64, u64)>,
            //     cache_size: u64,
            // ) -> Result<Vec<(Vec<(u64, u64)>, f32)>>
            // let len = data.len();
            // let _ = (0..len)
            //     .into_par_iter()
            //     .try_for_each_with(tx, |sender, idx| {
            //         let value = data[idx];
            //         let mut rng = rand::rng();
            //         let delay_secs = rng.random_range(1..=5);
            //         println!("Sleeping {delay_secs}s before yielding {value}");
            //         thread::sleep(Duration::from_secs(delay_secs));
            //         sender.send(value).map_err(|_| ())
            //     });
            let _ = route_definitions.into_par_iter().try_for_each_with(
                tx,
                |sender, (start_points, end_points)| {
                    // println!("Sleeping {delay_secs}s before yielding {value}");
                    println!("Computing routes between {start_points:?} and {end_points:?}");
                    let routes = find_paths(
                        zarr_fp.clone(),
                        cost_function.clone(),
                        start_points,
                        end_points,
                        cache_size,
                    );
                    sender.send(routes)
                },
            );
        });

        Self {
            receiver: rx,
            finished: false,
        }
    }
}

#[pymethods]
impl NumberIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    // fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<i32>>  {
    fn __next__(
        mut slf: PyRefMut<'_, Self>,
    ) -> PyResult<Option<Result<Vec<(Vec<(u64, u64)>, f32)>>>> {
        if slf.finished {
            return Ok(None);
        }

        match slf.receiver.recv() {
            Ok(value) => Ok(Some(value)),
            Err(_) => {
                slf.finished = true;
                Ok(None)
            }
        }
    }
}
