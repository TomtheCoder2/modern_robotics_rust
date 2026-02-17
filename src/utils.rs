use nalgebra::{SMatrix, Vector3, Vector4, RowVector3, RowVector4};
use nalgebra::{DMatrix, SVD};
// ========== Basics: small vector helpers ==========

#[inline]
pub fn row3(a: f64, b: f64, c: f64) -> RowVector3<f64> {
    RowVector3::new(a, b, c)
}

#[inline]
pub fn col3(a: f64, b: f64, c: f64) -> Vector3<f64> {
    Vector3::new(a, b, c)
}

#[inline]
pub fn row4(a: f64, b: f64, c: f64, d: f64) -> RowVector4<f64> {
    RowVector4::new(a, b, c, d)
}

#[inline]
pub fn col4(a: f64, b: f64, c: f64, d: f64) -> Vector4<f64> {
    Vector4::new(a, b, c, d)
}

// ========== Fixed-size horizontal/vertical stack ==========
// "c_" like NumPy: concat columns, rows must match.
/// Stack two matrices horizontally (columns), rows must match. Result has same number of rows, and sum of columns.
/// Example: c_stack([[1, 2], [3, 4]], [[5], [6]]) = [[1, 2, 5], [3, 4, 6]]
/// ```
///   ┌     ┐     ┌   ┐       ┌       ┐ 
///   │ 1 2 │ and │ 5 │ gives │ 1 2 5 │
///   │ 3 4 │     │ 6 │       │ 3 4 6 │
///   └     ┘     └   ┘       └       ┘
/// ```
pub fn c_stack<const R: usize, const C1: usize, const C2: usize>(
    a: &SMatrix<f64, R, C1>,
    b: &SMatrix<f64, R, C2>,
) -> SMatrix<f64, R, { C1 + C2 }> {
    let mut out = SMatrix::<f64, R, { C1 + C2 }>::zeros();
    out.fixed_view_mut::<R, C1>(0, 0).copy_from(a);
    out.fixed_view_mut::<R, C2>(0, C1).copy_from(b);
    out
}

// "r_" like NumPy: stack rows, columns must match.
/// Stack two matrices vertically (rows), columns must match. Result has sum of rows, and same number of columns.
/// Example: r_stack([[1, 2], [3, 4]], [[5, 6]]) = [[1, 2], [3, 4], [5, 6]]
/// ```
///   ┌     ┐     ┌       ┐       ┌     ┐ 
///   │ 1 2 │ and │ 5 6   │ gives │ 1 2 │
///   │ 3 4 │     │       │       │ 3 4 │
///   └     ┘     └       ┘       │ 5 6 │
///                               └     ┘
/// ```
pub fn r_stack<const R1: usize, const R2: usize, const C: usize>(
    a: &SMatrix<f64, R1, C>,
    b: &SMatrix<f64, R2, C>,
) -> SMatrix<f64, { R1 + R2 }, C> {
    let mut out = SMatrix::<f64, { R1 + R2 }, C>::zeros();
    out.fixed_view_mut::<R1, C>(0, 0).copy_from(a);
    out.fixed_view_mut::<R2, C>(R1, 0).copy_from(b);
    out
}

// Variadic convenience macros for small counts (2–4 terms)

// c_[A, B, ...]
#[macro_export]
macro_rules! c_ {
    ($a:expr, $b:expr) => {{
        $crate::utils::c_stack(&$a, &$b)
    }};
    ($a:expr, $b:expr, $c:expr) => {{
        let t = $crate::utils::c_stack(&$a, &$b);
        $crate::utils::c_stack(&t, &$c)
    }};
    ($a:expr, $b:expr, $c:expr, $d:expr) => {{
        let t1 = $crate::utils::c_stack(&$a, &$b);
        let t2 = $crate::utils::c_stack(&t1, &$c);
        $crate::utils::c_stack(&t2, &$d)
    }};
}

// r_[A, B, ...]
#[macro_export]
macro_rules! r_ {
    ($a:expr, $b:expr) => {{
        $crate::utils::r_stack(&$a, &$b)
    }};
    ($a:expr, $b:expr, $c:expr) => {{
        let t = $crate::utils::r_stack(&$a, &$b);
        $crate::utils::r_stack(&t, &$c)
    }};
    ($a:expr, $b:expr, $c:expr, $d:expr) => {{
        let t1 = $crate::utils::r_stack(&$a, &$b);
        let t2 = $crate::utils::r_stack(&t1, &$c);
        $crate::utils::r_stack(&t2, &$d)
    }};
}

// ========== Block assembly like np.r_[[A, B], [C, D]] ==========

pub fn block2x2<
    const R1: usize,
    const R2: usize,
    const C1: usize,
    const C2: usize,
>(
    a: &SMatrix<f64, R1, C1>,
    b: &SMatrix<f64, R1, C2>,
    c: &SMatrix<f64, R2, C1>,
    d: &SMatrix<f64, R2, C2>,
) -> SMatrix<f64, { R1 + R2 }, { C1 + C2 }> {
    let mut out = SMatrix::<f64, { R1 + R2 }, { C1 + C2 }>::zeros();
    out.fixed_view_mut::<R1, C1>(0, 0).copy_from(a);
    out.fixed_view_mut::<R1, C2>(0, C1).copy_from(b);
    out.fixed_view_mut::<R2, C1>(R1, 0).copy_from(c);
    out.fixed_view_mut::<R2, C2>(R1, C1).copy_from(d);
    out
}

// ========== Handy literals for 3x3, 4x4 in row-major spirit ==========

#[macro_export]
macro_rules! m3 {
    ([$a11:expr, $a12:expr, $a13:expr],
     [$a21:expr, $a22:expr, $a23:expr],
     [$a31:expr, $a32:expr, $a33:expr] $(,)?) => {{
        // Row-major feel with nalgebra construction that yields expected rows.
        nalgebra::SMatrix::<f64, 3, 3>::from_row_slice(&[
            $a11 as f64, $a12 as f64, $a13 as f64,
            $a21 as f64, $a22 as f64, $a23 as f64,
            $a31 as f64, $a32 as f64, $a33 as f64,
        ])
    }};
}

#[macro_export]
macro_rules! m4 {
    ([$a11:expr, $a12:expr, $a13:expr, $a14:expr],
     [$a21:expr, $a22:expr, $a23:expr, $a24:expr],
     [$a31:expr, $a32:expr, $a33:expr, $a34:expr],
     [$a41:expr, $a42:expr, $a43:expr, $a44:expr] $(,)?) => {{
        nalgebra::SMatrix::<f64, 4, 4>::from_row_slice(&[
            $a11 as f64, $a12 as f64, $a13 as f64, $a14 as f64,
            $a21 as f64, $a22 as f64, $a23 as f64, $a24 as f64,
            $a31 as f64, $a32 as f64, $a33 as f64, $a34 as f64,
            $a41 as f64, $a42 as f64, $a43 as f64, $a44 as f64,
        ])
    }};
}

// todo maybe make faster versions for small fixed sizes, but this is general and works fine for now

pub fn pinv_lapack(a: &DMatrix<f64>) -> DMatrix<f64> {
    let svd = SVD::new(a.clone_owned(), true, true);

    let u = svd.u.expect("U not computed");
    let s = svd.singular_values;
    let vt = svd.v_t.expect("V^T not computed");

    let (m, n) = a.shape();
    let r = s.len();

    let eps = f64::EPSILON;
    let smax = s.iter().cloned().fold(0.0, f64::max);
    let tol = (m.max(n) as f64) * eps * smax;

    // Build Σ⁺ as r × r
    let mut sigma_p = DMatrix::<f64>::zeros(r, r);
    for i in 0..r {
        if s[i] > tol {
            sigma_p[(i, i)] = 1.0 / s[i];
        }
    }

    // A⁺ = V Σ⁺ Uᵀ
    // vt is r × n → V = vt.transpose() is n × r
    // u is m × r → Uᵀ is r × m
    vt.transpose() * sigma_p * u.transpose()
}

pub fn smat_to_dmat<const R: usize, const C: usize>(
    m: &SMatrix<f64, R, C>,
) -> DMatrix<f64> {
    DMatrix::<f64>::from_column_slice(R, C, m.as_slice())
}


pub fn dmat_to_smat<const R: usize, const C: usize>(
    m: &DMatrix<f64>,
) -> SMatrix<f64, R, C> {
    assert_eq!(m.nrows(), R);
    assert_eq!(m.ncols(), C);
    SMatrix::<f64, R, C>::from_column_slice(m.as_slice())
}