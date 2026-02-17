/// macro definitions to construct matrices row major like in NumPy, but using nalgebra's column-major storage

/// 3x3 matrix, NumPy-style rows
#[macro_export]
macro_rules! mat3 {
    (
        [$a11:expr, $a12:expr, $a13:expr],
        [$a21:expr, $a22:expr, $a23:expr],
        [$a31:expr, $a32:expr, $a33:expr] $(,)?
    ) => {
        nalgebra::SMatrix::<f64, 3, 3>::from_row_slice(&[
            $a11 as f64,
            $a12 as f64,
            $a13 as f64,
            $a21 as f64,
            $a22 as f64,
            $a23 as f64,
            $a31 as f64,
            $a32 as f64,
            $a33 as f64,
        ])
    };
    (
        $m11:expr, $m12:expr, $m13:expr,
        $m21:expr, $m22:expr, $m23:expr,
        $m31:expr, $m32:expr, $m33:expr $(,)?
    ) => {
        nalgebra::SMatrix::<f64, 3, 3>::from_row_slice(&[
            $m11 as f64,
            $m12 as f64,
            $m13 as f64,
            $m21 as f64,
            $m22 as f64,
            $m23 as f64,
            $m31 as f64,
            $m32 as f64,
            $m33 as f64,
        ])
    };
}

/// 4x4 matrix, NumPy-style rows
#[macro_export]
macro_rules! mat4 {
    (
        [$a11:expr, $a12:expr, $a13:expr, $a14:expr],
        [$a21:expr, $a22:expr, $a23:expr, $a24:expr],
        [$a31:expr, $a32:expr, $a33:expr, $a34:expr],
        [$a41:expr, $a42:expr, $a43:expr, $a44:expr] $(,)?
    ) => {
        nalgebra::SMatrix::<f64, 4, 4>::from_row_slice(&[
            $a11 as f64,
            $a12 as f64,
            $a13 as f64,
            $a14 as f64,
            $a21 as f64,
            $a22 as f64,
            $a23 as f64,
            $a24 as f64,
            $a31 as f64,
            $a32 as f64,
            $a33 as f64,
            $a34 as f64,
            $a41 as f64,
            $a42 as f64,
            $a43 as f64,
            $a44 as f64,
        ])
    };
    (
        $m11:expr, $m12:expr, $m13:expr, $m14:expr,
        $m21:expr, $m22:expr, $m23:expr, $m24:expr,
        $m31:expr, $m32:expr, $m33:expr, $m34:expr,
        $m41:expr, $m42:expr, $m43:expr, $m44:expr $(,)?
    ) => {
        nalgebra::SMatrix::<f64, 4, 4>::from_row_slice(&[
            $m11 as f64,
            $m12 as f64,
            $m13 as f64,
            $m14 as f64,
            $m21 as f64,
            $m22 as f64,
            $m23 as f64,
            $m24 as f64,
            $m31 as f64,
            $m32 as f64,
            $m33 as f64,
            $m34 as f64,
            $m41 as f64,
            $m42 as f64,
            $m43 as f64,
            $m44 as f64,
        ])
    };
}
