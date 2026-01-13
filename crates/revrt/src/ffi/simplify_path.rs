//! Simplify a path by removing points that fall on straight segments

/// Compute slope between two points
///
/// `p1`, `p2`: 2-tuples representing points (x, y).
///
/// This function returns `None` for vertical slopes.
fn slope(p1: &(f64, f64), p2: &(f64, f64)) -> Option<f64> {
    let dx = p2.0 - p1.0;
    let dy = p2.1 - p1.1;
    if dx.abs() < 1e-12 {
        None
    } else {
        Some(dy / dx)
    }
}

/// Check two slopes for equality within a tolerance
///
/// `a`, `b`: Slope values. `None` indicates a vertical slope.
/// `tol`: A float representing a slope tolerance to allow.
///  The tolerance is non-inclusive.
fn equal_slopes(a: Option<f64>, b: Option<f64>, tol: f64) -> bool {
    match (a, b) {
        (None, None) => true, // both vertical
        (Some(x), Some(y)) => (x - y).abs() < tol,
        _ => false, // one vertical, the other not
    }
}

/// Function to simplify a path by removing points that fall on straight segments
///
/// `points`: A collection points (x, y) representing a path.
/// `slope_tolerance`: A float representing a slope tolerance to allow.
///  For typical 8-direction routing, a value of 1 should suffice
pub(super) fn simplify_path(points: Vec<(f64, f64)>, slope_tolerance: f64) -> Vec<(f64, f64)> {
    let len = points.len();
    if len <= 2 {
        return points;
    }

    points.into_iter().fold(
        Vec::with_capacity(len),
        |mut simplified_path, next_point| {
            match simplified_path.len() {
                0 | 1 => simplified_path.push(next_point),
                size => {
                    let prev_prev_point = &simplified_path[size - 2];
                    let prev_point = &simplified_path[size - 1];
                    if equal_slopes(
                        slope(prev_prev_point, prev_point),
                        slope(prev_point, &next_point),
                        slope_tolerance,
                    ) {
                        simplified_path.pop();
                    }
                    simplified_path.push(next_point);
                }
            }
            simplified_path
        },
    )
}

#[cfg(test)]
mod test_simplify_path {
    use super::*;

    #[test]
    fn basic_simplify() {
        let path = vec![(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)];
        let simplified = simplify_path(path, 0.01);

        assert_eq!(simplified, vec![(1.0, 1.0), (3.0, 3.0)]);
    }

    #[test]
    fn basic_simplify_all_points() {
        let path = vec![(1.0, 1.0), (2.0, 2.0), (3.0, 2.0)];
        let simplified = simplify_path(path.clone(), 0.01);

        assert_eq!(simplified, path);
    }

    #[test]
    fn empty_simplify() {
        let path = vec![];
        let simplified = simplify_path(path.clone(), 0.01);

        assert_eq!(simplified, path);
    }

    #[test]
    fn single_item_simplify() {
        let path = vec![(1.0, 1.0)];
        let simplified = simplify_path(path.clone(), 0.01);

        assert_eq!(simplified, path);
    }

    #[test]
    fn two_item_simplify() {
        let path = vec![(1.0, 1.0), (2.0, 2.0)];
        let simplified = simplify_path(path.clone(), 0.01);

        assert_eq!(simplified, path);
    }

    #[test]
    fn v_shape() {
        let path = vec![(0.0, 2.0), (1.0, 1.0), (2.0, 2.0)];
        let simplified = simplify_path(path.clone(), 0.01);

        assert_eq!(simplified, path);
    }

    #[test]
    fn long_path() {
        let path = vec![
            (0.0, 0.0),
            (1.0, 1.0),
            (2.0, 2.0),
            (3.0, 3.0),
            (4.0, 4.0),
            (5.0, 5.0),
            (6.0, 6.0),
            (7.0, 7.0),
            (8.0, 8.0),
            (9.0, 9.0),
            (10.0, 10.0),
            (11.0, 10.0),
        ];
        let simplified = simplify_path(path, 0.01);

        assert_eq!(simplified, vec![(0.0, 0.0), (10.0, 10.0), (11.0, 10.0)]);
    }
}
