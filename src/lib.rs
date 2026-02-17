#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![allow(deprecated)]

pub mod macros;
pub mod utils;
pub mod dynamics;
pub mod trajectory_gen;
pub mod robot_control;

use crate::utils::{c_stack, dmat_to_smat, pinv_lapack, r_stack, smat_to_dmat};
use nalgebra::{SMatrix, SVector, Vector3, Vector6};
use std::ops::Mul;
pub use dynamics::*;
pub use trajectory_gen::*;
pub use robot_control::*;

/// Determines whether a scalar is small enough to be treated as zero
///
/// # Arguments
/// * `z` - A scalar input to check
/// # Returns
/// * `true` if z is close to zero, false otherwise
/// # Example
/// ```
/// use modern_robotics_rust::near_zero;
/// let z = -1e-7;
/// assert!(near_zero(z));
/// ```
pub fn near_zero(z: f64) -> bool {
    z.abs() < 1e-6
}

/// Normalizes a vector to have unit length
/// # Arguments
/// * `v` - A vector to normalize
/// # Returns
/// * A unit vector pointing in the same direction as v
/// # Example
/// ```
/// use nalgebra::Vector3;
/// use modern_robotics_rust::normalize;
/// let v = Vector3::new(1.0, 2.0, 3.0);
/// let normalized_v = normalize(v);
/// assert_eq!(normalized_v, Vector3::new(0.2672612419124244, 0.5345224838248488, 0.8017837257372732));
/// ```
pub fn normalize(v: Vector3<f64>) -> Vector3<f64> {
    v / v.norm()
}

/// Inverts a rotation matrix
/// # Arguments
/// * `R` - A 3x3 rotation matrix
/// # Returns
/// * The inverse of R
///
/// # Example
/// ```
/// use nalgebra::SMatrix;
/// use modern_robotics_rust::rot_inv;
/// use modern_robotics_rust::mat3;
/// let R = mat3!(0.0, 0.0, 1.0,
///                                     1.0, 0.0, 0.0,
///                                     0.0, 1.0, 0.0);
/// let R_inv = rot_inv(R);
/// assert_eq!(R_inv, mat3!(0.0, 1.0, 0.0,
///                                     0.0, 0.0, 1.0,
///                                     1.0, 0.0, 0.0));
/// ```
pub fn rot_inv(r: SMatrix<f64, 3, 3>) -> SMatrix<f64, 3, 3> {
    r.transpose()
}

/// Converts a 3-vector to an so(3) representation
/// # Arguments
/// * `omg` - A 3-vector representing angular velocity
/// # Returnsr
/// * The skew symmetric representation of omg
/// # Example
/// ```
/// use nalgebra::Vector3;
/// use modern_robotics_rust::vec_to_so3;
/// use nalgebra::SMatrix;
/// use modern_robotics_rust::mat3;
/// let omg = Vector3::new(1.0, 2.0, 3.0);
/// let omg_hat = vec_to_so3(omg);
/// assert_eq!(omg_hat, mat3!(0.0, -3.0, 2.0,
///                                     3.0, 0.0, -1.0,
///                                     -2.0, 1.0, 0.0));
/// ```
pub fn vec_to_so3(omg: Vector3<f64>) -> SMatrix<f64, 3, 3> {
    mat3!(
        0.0, -omg[2], omg[1], omg[2], 0.0, -omg[0], -omg[1], omg[0], 0.0,
    )
}

/// Converts an so(3) representation to a 3-vector
/// # Arguments
/// * `so3mat` - A 3x3 skew-symmetric matrix
/// # Returns
/// * The 3-vector corresponding to so3mat
/// # Example
/// ```
/// use nalgebra::SMatrix;
/// use nalgebra::Vector3;
/// use modern_robotics_rust::so3_to_vec;
/// use modern_robotics_rust::mat3;
/// let so3mat = mat3!(0.0, -3.0, 2.0,
///                                     3.0, 0.0, -1.0,
///                                     -2.0, 1.0, 0.0);
/// let omg = so3_to_vec(so3mat);
/// assert_eq!(omg, Vector3::new(1.0, 2.0, 3.0));
/// ```
pub fn so3_to_vec(so3mat: SMatrix<f64, 3, 3>) -> Vector3<f64> {
    Vector3::new(so3mat[(2, 1)], so3mat[(0, 2)], so3mat[(1, 0)])
}

/// Converts a 3-vector of exponential coordinates for rotation into axis-angle form
/// # Arguments
/// * `expc3` - A 3-vector of exponential coordinates for rotation
/// # Returns
/// * A tuple containing the unit rotation axis and the corresponding rotation angle    
/// # Example
/// ```
/// use nalgebra::Vector3;
/// use modern_robotics_rust::axis_ang_3;
/// let expc3 = Vector3::new(1.0, 2.0, 3.0);
/// let (omghat, theta) = axis_ang_3(expc3);
/// assert_eq!(omghat, Vector3::new(0.2672612419124244, 0.5345224838248488, 0.8017837257372732));
/// assert_eq!(theta, 3.7416573867739413);
/// ```
pub fn axis_ang_3(expc3: Vector3<f64>) -> (Vector3<f64>, f64) {
    let theta = expc3.norm();
    if near_zero(theta) {
        (Vector3::zeros(), 0.0)
    } else {
        (expc3 / theta, theta)
    }
}

/// Converts a 3x3 skew-symmetric matrix in so(3) to a rotation matrix using the matrix exponential
/// # Arguments
/// * `so3mat` - A 3x3 skew-symmetric matrix
/// # Returns
/// * The rotation matrix corresponding to so3mat
/// # Example
/// ```
/// use nalgebra::SMatrix;
/// use modern_robotics_rust::{matrix_exp3, near_zero};
/// use modern_robotics_rust::mat3;
/// let so3mat = mat3!(0.0, -3.0, 2.0,
///                                     3.0, 0.0, -1.0,
///                                     -2.0, 1.0, 0.0);
/// let R = matrix_exp3(so3mat);
/// assert!(near_zero((R - mat3!(-0.69492056, 0.71352099, 0.08929286,
///                                     -0.19200697, -0.30378504, 0.93319235,
///                                     0.69297817, 0.6313497, 0.34810748)).abs().max()));
/// ```
pub fn matrix_exp3(so3mat: SMatrix<f64, 3, 3>) -> SMatrix<f64, 3, 3> {
    let omgtheta = so3_to_vec(so3mat);
    if near_zero(omgtheta.norm()) {
        SMatrix::<f64, 3, 3>::identity()
    } else {
        let (omghat, theta) = axis_ang_3(omgtheta);
        let omgmat = vec_to_so3(omghat);
        SMatrix::<f64, 3, 3>::identity()
            + theta.sin() * omgmat
            + (1.0 - theta.cos()) * (omgmat * omgmat)
    }
}

/// Computes the matrix logarithm of a rotation matrix
/// # Arguments
/// * `R` - A 3x3 rotation matrix
/// # Returns
/// * The matrix logarithm of R
/// # Example
/// ```
/// use nalgebra::SMatrix;
/// use modern_robotics_rust::{matrix_log3, near_zero};
/// use modern_robotics_rust::mat3;
/// let R = mat3!(0.0, 0.0, 1.0,
///                                     1.0, 0.0, 0.0,
///                                     0.0, 1.0, 0.0);
/// let log_R = matrix_log3(R);
/// assert!(near_zero((log_R - mat3!(0.0, -1.20919958, 1.20919958,
///                                     1.20919958, 0.0, -1.20919958,
///                                     -1.20919958, 1.20919958, 0.0)).abs().max()));
/// ```
pub fn matrix_log3(r: SMatrix<f64, 3, 3>) -> SMatrix<f64, 3, 3> {
    let acosinput = (r.trace() - 1.0) / 2.0;
    if acosinput >= 1.0 {
        SMatrix::<f64, 3, 3>::zeros()
    } else if acosinput <= -1.0 {
        let omg;
        if !near_zero(1.0 + r[(2, 2)]) {
            omg = (1.0 / (2.0 * (1.0 + r[(2, 2)]).sqrt()))
                * Vector3::new(r[(0, 2)], r[(1, 2)], 1.0 + r[(2, 2)]);
        } else if !near_zero(1.0 + r[(1, 1)]) {
            omg = (1.0 / (2.0 * (1.0 + r[(1, 1)]).sqrt()))
                * Vector3::new(r[(0, 1)], 1.0 + r[(1, 1)], r[(2, 1)]);
        } else {
            omg = (1.0 / (2.0 * (1.0 + r[(0, 0)]).sqrt()))
                * Vector3::new(1.0 + r[(0, 0)], r[(1, 0)], r[(2, 0)]);
        }
        vec_to_so3(std::f64::consts::PI * omg)
    } else {
        let theta = acosinput.acos();
        (theta / (2.0 * theta.sin())) * (r - r.transpose())
    }
}

/// Converts a rotation matrix and a position vector into homogeneous transformation matrix
/// # Arguments
/// * `R` - A 3x3 rotation matrix
/// * `p` - A 3-vector representing position
/// # Returns
/// * A 4x4 homogeneous transformation matrix corresponding to the inputs
/// # Example
/// ```
/// use nalgebra::SMatrix;
/// use nalgebra::Vector3;
/// use modern_robotics_rust::rp_to_trans;
/// use modern_robotics_rust::mat3;
/// use modern_robotics_rust::mat4;
/// let R = mat3!(1.0, 0.0, 0.0,
///                                     0.0, 0.0, -1.0,
///                                     0.0, 1.0, 0.0);
/// let p = Vector3::new(1.0, 2.0, 5.0);
/// let T = rp_to_trans(R, p);
/// assert_eq!(T, mat4!(1.0, 0.0, 0.0, 1.0,
///                                     0.0, 0.0, -1.0, 2.0,
///                                     0.0, 1.0, 0.0, 5.0,
///                                     0.0, 0.0, 0.0, 1.0));
/// ```
pub fn rp_to_trans(r: SMatrix<f64, 3, 3>, p: Vector3<f64>) -> SMatrix<f64, 4, 4> {
    mat4!(
        r[(0, 0)],
        r[(0, 1)],
        r[(0, 2)],
        p[0],
        r[(1, 0)],
        r[(1, 1)],
        r[(1, 2)],
        p[1],
        r[(2, 0)],
        r[(2, 1)],
        r[(2, 2)],
        p[2],
        0.0,
        0.0,
        0.0,
        1.0,
    )
}

/// Converts a homogeneous transformation matrix into a rotation matrix and position vector
/// # Arguments
/// * `T` - A 4x4 homogeneous transformation matrix
/// # Returns
/// * A tuple containing the corresponding rotation matrix and position vector
/// # Example
/// ```
/// use nalgebra::SMatrix;
/// use nalgebra::Vector3;
/// use modern_robotics_rust::trans_to_rp;
/// use modern_robotics_rust::mat3;
/// use modern_robotics_rust::mat4;
/// let T = mat4!(1.0, 0.0, 0.0, 0.0,
///                                     0.0, 0.0, -1.0, 0.0,
///                                     0.0, 1.0, 0.0, 3.0,
///                                     0.0, 0.0, 0.0, 1.0);
/// let (R, p) = trans_to_rp(T);
/// assert_eq!(R, mat3!(1.0, 0.0, 0.0,
///                                     0.0, 0.0, -1.0,
///                                     0.0, 1.0, 0.0));
/// assert_eq!(p, Vector3::new(0.0, 0.0, 3.0));
/// ```
pub fn trans_to_rp(t: SMatrix<f64, 4, 4>) -> (SMatrix<f64, 3, 3>, Vector3<f64>) {
    (
        t.fixed_slice::<3, 3>(0, 0).into(),
        Vector3::new(t[(0, 3)], t[(1, 3)], t[(2, 3)]),
    )
}

/// Inverts a homogeneous transformation matrix
/// # Arguments
/// * `T` - A 4x4 homogeneous transformation matrix
/// # Returns
/// * The inverse of T
/// # Example
/// ```
/// use nalgebra::SMatrix;
/// use nalgebra::Vector3;
/// use modern_robotics_rust::trans_inv;
/// use modern_robotics_rust::mat4;
/// let T = mat4!(1.0, 0.0, 0.0, 0.0,
///                                     0.0, 0.0, -1.0, 0.0,
///                                     0.0, 1.0, 0.0, 3.0,
///                                     0.0, 0.0, 0.0, 1.0);
/// let T_inv = trans_inv(T);
/// println!("T_inv: {T_inv}");
/// assert_eq!(T_inv, mat4!(1.0, 0.0, 0.0, 0.0,
///                                     0.0, 0.0, 1.0, -3.0,
///                                     0.0, -1.0, 0.0, 0.0,
///                                     0.0, 0.0, 0.0, 1.0));
/// ```
pub fn trans_inv(t: SMatrix<f64, 4, 4>) -> SMatrix<f64, 4, 4> {
    let (r, p) = trans_to_rp(t);
    let rt = r.transpose();
    mat4!(
        rt[(0, 0)],
        rt[(0, 1)],
        rt[(0, 2)],
        -rt.row(0).mul(&p).sum(),
        rt[(1, 0)],
        rt[(1, 1)],
        rt[(1, 2)],
        -rt.row(1).mul(&p).sum(),
        rt[(2, 0)],
        rt[(2, 1)],
        rt[(2, 2)],
        -rt.row(2).mul(&p).sum(),
        0.0,
        0.0,
        0.0,
        1.0,
    )
}

/// Converts a spatial velocity vector into a 4x4 matrix in se3
/// # Arguments
/// * `V` - A 6-vector representing a spatial velocity
/// # Returns
/// * The 4x4 se3 representation of V
/// # Example
/// ```
/// use nalgebra::Vector6;
///  use nalgebra::SMatrix;
/// use modern_robotics_rust::vec_to_se3;
/// use modern_robotics_rust::mat4;
/// let V = Vector6::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
/// let se3mat = vec_to_se3(V);
/// assert_eq!(se3mat, mat4!(0.0, -3.0, 2.0, 4.0,
///                                     3.0, 0.0, -1.0, 5.0,
///                                     -2.0, 1.0, 0.0, 6.0,
///                                     0.0, 0.0, 0.0, 0.0));
/// ```
pub fn vec_to_se3(v: Vector6<f64>) -> SMatrix<f64, 4, 4> {
    mat4!(
        0.0, -v[2], v[1], v[3], v[2], 0.0, -v[0], v[4], -v[1], v[0], 0.0, v[5], 0.0, 0.0, 0.0, 0.0,
    )
}

/// Converts a 4x4 matrix in se3 into a spatial velocity vector
/// # Arguments
/// * `se3mat` - A 4x4 matrix in se3
/// # Returns
/// * The 6-vector corresponding to se3mat
/// # Example
/// ```
/// use nalgebra::SMatrix;
/// use nalgebra::Vector6;
/// use modern_robotics_rust::se3_to_vec;
/// use modern_robotics_rust::mat4;
/// let se3mat = mat4!(0.0, -3.0, 2.0, 4.0,
///                                     3.0, 0.0, -1.0, 5.0,
///                                     -2.0, 1.0, 0.0, 6.0,
///                                     0.0, 0.0, 0.0, 0.0);
/// let V = se3_to_vec(se3mat);
/// assert_eq!(V, Vector6::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0));
/// ```
pub fn se3_to_vec(se3mat: SMatrix<f64, 4, 4>) -> Vector6<f64> {
    Vector6::new(
        se3mat[(2, 1)],
        se3mat[(0, 2)],
        se3mat[(1, 0)],
        se3mat[(0, 3)],
        se3mat[(1, 3)],
        se3mat[(2, 3)],
    )
}

/// Computes the adjoint representation of a homogeneous transformation matrix
/// # Arguments
/// * `T` - A 4x4 homogeneous transformation matrix
/// # Returns
/// * The 6x6 adjoint representation of T
/// # Example
/// ```
/// use nalgebra::SMatrix;
/// use modern_robotics_rust::adjoint;
/// use modern_robotics_rust::mat4;
/// let T = mat4!(1.0, 0.0, 0.0, 0.0,
///                                     0.0, 0.0, -1.0, 0.0,
///                                     0.0, 1.0, 0.0, 3.0,
///                                     0.0, 0.0, 0.0, 1.0);
/// let AdT = adjoint(T);
/// assert_eq!(AdT, SMatrix::<f64, 6, 6>::from_row_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
///                                     0.0, 0.0, -1.0, 0.0, 0.0, 0.0,
///                                     0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
///                                     0.0, 0.0, 3.0, 1.0, 0.0, 0.0,
///                                     3.0, 0.0, 0.0, 0.0, 0.0, -1.0,
///                                     0.0, 0.0, 0.0, 0.0, 1.0, 0.0]));
/// ```
pub fn adjoint(t: SMatrix<f64, 4, 4>) -> SMatrix<f64, 6, 6> {
    // np.r_[np.c_[R, np.zeros((3, 3))],
    //                  np.c_[np.dot(VecToso3(p), R), R]]
    let (r, p) = trans_to_rp(t);
    let mut adt = SMatrix::<f64, 6, 6>::zeros();
    adt.fixed_slice_mut::<3, 3>(0, 0).copy_from(&r);
    adt.fixed_slice_mut::<3, 3>(3, 0)
        .copy_from(&(vec_to_so3(p) * r));
    adt.fixed_slice_mut::<3, 3>(3, 3).copy_from(&r);
    adt
}

/// Converts a parametric description of a screw axis into a normalized screw axis
/// # Arguments
/// * `q` - A point lying on the screw axis
/// * `s` - A unit vector in the direction of the screw axis
/// * `h` - The pitch of the screw axis
/// # Returns
/// * A normalized screw axis described by the inputs
/// # Example
/// ```
/// use nalgebra::Vector3;
/// use nalgebra::Vector6;
/// use modern_robotics_rust::screw_to_axis;
/// let q = Vector3::new(3.0, 0.0, 0.0);
/// let s = Vector3::new(0.0, 0.0, 1.0);
/// let h = 2.0;
/// let screw_axis = screw_to_axis(q, s, h);
/// assert_eq!(screw_axis, Vector6::new(0.0, 0.0, 1.0, 0.0, -3.0, 2.0));
/// ```
pub fn screw_to_axis(q: Vector3<f64>, s: Vector3<f64>, h: f64) -> Vector6<f64> {
    let temp = q.cross(&s) + h * s;
    Vector6::new(s[0], s[1], s[2], temp[0], temp[1], temp[2])
}

/// Converts a 6-vector of exponential coordinates into screw axis-angle form
/// # Arguments
/// * `expc6` - A 6-vector of exponential coordinates for rigid-body motion
/// # Returns
/// * A tuple containing the corresponding normalized screw axis and the distance traveled along/about it
/// # Example
/// ```
/// use nalgebra::Vector6;
/// use modern_robotics_rust::axis_ang_6;
/// let expc6 = Vector6::new(1.0, 0.0, 0.0, 1.0, 2.0, 3.0);
/// let (s, theta) = axis_ang_6(expc6);
/// assert_eq!(s, Vector6::new(1.0, 0.0, 0.0, 1.0, 2.0, 3.0));
/// assert_eq!(theta, 1.0);
/// ```
pub fn axis_ang_6(expc6: Vector6<f64>) -> (Vector6<f64>, f64) {
    let theta = expc6.fixed_rows::<3>(0).norm();
    let theta = if near_zero(theta) {
        expc6.fixed_rows::<3>(3).norm()
    } else {
        theta
    };
    (expc6 / theta, theta)
}

/// Computes the matrix exponential of an se3 representation of exponential coordinates
/// # Arguments
/// * `se3mat` - A 4x4 matrix in se3
/// # Returns
/// * The matrix exponential of se3mat
/// # Example
/// ```
/// use nalgebra::SMatrix;
/// use nalgebra::Vector6;
/// use modern_robotics_rust::{vec_to_se3, matrix_exp6, near_zero};
/// use modern_robotics_rust::mat4;
/// let se3mat = mat4!(0.0, 0.0, 0.0, 0.0,
///                                     0.0, 0.0, -1.57079632, 2.35619449,
///                                     0.0, 1.57079632, 0.0, 2.35619449,
///                                     0.0, 0.0, 0.0, 0.0);
/// let T = matrix_exp6(se3mat);
/// assert!(near_zero((T - mat4!(1.0, 0.0, 0.0, 0.0,
///                                     0.0, 0.0, -1.0, 0.0,
///                                     0.0, 1.0, 0.0, 3.0,
///                                     0.0, 0.0, 0.0, 1.0)).abs().max()));
/// ```
pub fn matrix_exp6(se3mat: SMatrix<f64, 4, 4>) -> SMatrix<f64, 4, 4> {
    let omgtheta = so3_to_vec(<SMatrix<f64, 3, 3>>::from(se3mat.fixed_slice::<3, 3>(0, 0)));
    if near_zero(omgtheta.norm()) {
        rp_to_trans(
            SMatrix::<f64, 3, 3>::identity(),
            se3mat.fixed_slice::<3, 1>(0, 3).into(),
        )
    } else {
        let (omghat, theta) = axis_ang_3(omgtheta);
        let omgmat = vec_to_so3(omghat);
        let r = matrix_exp3(<SMatrix<f64, 3, 3>>::from(se3mat.fixed_slice::<3, 3>(0, 0)));
        let p = ((SMatrix::<f64, 3, 3>::identity() * theta
            + (1.0 - theta.cos()) * omgmat
            + (theta - theta.sin()) * (omgmat * omgmat))
            * se3mat.fixed_slice::<3, 1>(0, 3))
            / theta;
        rp_to_trans(r, p.into())
    }
}

// def MatrixLog6(T):
//     """Computes the matrix logarithm of a homogeneous transformation matrix
//
//     :param R: A matrix in SE3
//     :return: The matrix logarithm of R
//
//     Example Input:
//         T = np.array([[1, 0,  0, 0],
//                       [0, 0, -1, 0],
//                       [0, 1,  0, 3],
//                       [0, 0,  0, 1]])
//     Output:
//         np.array([[0,          0,           0,           0]
//                   [0,          0, -1.57079633,  2.35619449]
//                   [0, 1.57079633,           0,  2.35619449]
//                   [0,          0,           0,           0]])
//     """
//     R, p = TransToRp(T)
//     omgmat = MatrixLog3(R)
//     if np.array_equal(omgmat, np.zeros((3, 3))):
//         return np.r_[np.c_[np.zeros((3, 3)),
//                            [T[0][3], T[1][3], T[2][3]]],
//                      [[0, 0, 0, 0]]]
//     else:
//         theta = np.arccos((np.trace(R) - 1) / 2.0)
//         return np.r_[np.c_[omgmat,
//                            np.dot(np.eye(3) - omgmat / 2.0 \
//                            + (1.0 / theta - 1.0 / np.tan(theta / 2.0) / 2) \
//                               * np.dot(omgmat,omgmat) / theta,[T[0][3],
//                                                                T[1][3],
//                                                                T[2][3]])],
//                      [[0, 0, 0, 0]]]
/// Computes the matrix logarithm of a homogeneous transformation matrix
/// # Arguments
/// * `T` - A 4x4 homogeneous transformation matrix
/// # Returns
/// * The matrix logarithm of T
/// # Example
/// ```
/// use nalgebra::SMatrix;
/// use modern_robotics_rust::{matrix_log6, near_zero};
/// use modern_robotics_rust::mat4;
/// let T = mat4!(1, 0,  0, 0,
///                       0, 0, -1, 0,
///                       0, 1,  0, 3,
///                       0, 0,  0, 1);
/// let log_T = matrix_log6(T);
/// assert!(near_zero((log_T - mat4!(0.0, 0.0, 0.0, 0.0,
///                                     0.0, 0.0, -1.57079633, 2.35619449,
///                                     0.0, 1.57079633, 0.0, 2.35619449,
///                                     0.0, 0.0, 0.0, 0.0)).abs().max()));
/// ```
pub fn matrix_log6(t: SMatrix<f64, 4, 4>) -> SMatrix<f64, 4, 4> {
    let (r, p) = trans_to_rp(t);
    let omgmat = matrix_log3(r);
    if omgmat == SMatrix::<f64, 3, 3>::zeros() {
        vec_to_se3(Vector6::new(0.0, 0.0, 0.0, p[0], p[1], p[2]))
    } else {
        let theta = ((r.trace() - 1.0) / 2.0).acos();
        let omgmat_squared = omgmat * omgmat;
        let term2 = SMatrix::<f64, 3, 3>::identity() - omgmat / 2.0
            + (1.0 / theta - 1.0 / ((theta / 2.0).tan()) / 2.0) * omgmat_squared / theta;
        let term3 = term2 * Vector3::new(t[(0, 3)], t[(1, 3)], t[(2, 3)]);
        let mut res = SMatrix::<f64, 4, 4>::zeros();
        res.fixed_slice_mut::<3, 3>(0, 0).copy_from(&omgmat);
        res.fixed_slice_mut::<3, 1>(0, 3).copy_from(&term3);
        res
    }
}

/// Computes a projection of a matrix into SO(3) using singular-value decomposition
/// # Arguments
/// * `mat` - A 3x3 matrix near SO(3) to project to SO(3)
/// # Returns
/// * The closest matrix to mat that is in SO(3)
/// # Example
/// ```
/// use nalgebra::SMatrix;
/// use modern_robotics_rust::{near_zero, project_to_so3};
/// use modern_robotics_rust::mat3;
/// let mat = mat3!(0.675, 0.150, 0.720,
///                                     0.370, 0.771, -0.511,
///                                     -0.630, 0.619, 0.472);
/// let R = project_to_so3(mat);
/// assert!(near_zero((R - mat3!(0.67901136, 0.14894516, 0.71885945,
///                                     0.37320708, 0.77319584, -0.51272279,
///                                     -0.63218672, 0.61642804, 0.46942137)).abs().max()));
/// ```
pub fn project_to_so3(mat: SMatrix<f64, 3, 3>) -> SMatrix<f64, 3, 3> {
    let svd = mat.svd(true, true);
    let mut u = svd.u.unwrap();
    let v_t = svd.v_t.unwrap();
    let mut r = u * v_t;
    if r.determinant() < 0.0 {
        let u_clone = u.clone();
        u.fixed_slice_mut::<3, 1>(0, 2)
            .copy_from(&(-u_clone.column(2)));
        r = u * v_t;
    }
    r
}

/// Computes a projection of a matrix into SE(3) using singular-value decomposition
/// # Arguments
/// * `mat` - A 4x4 matrix near SE(3) to project to SE(3)
/// # Returns
/// * The closest matrix to mat that is in SE(3)
/// # Example
/// ```
/// use nalgebra::SMatrix;
/// use modern_robotics_rust::{near_zero, project_to_se3};
/// use modern_robotics_rust::mat4;
/// let mat = mat4!(0.675, 0.150, 0.720, 1.2,
///                                     0.370, 0.771, -0.511, 5.4,
///                                     -0.630, 0.619, 0.472, 3.6,
///                                     0.003, 0.002, 0.010, 0.9);
/// let T = project_to_se3(mat);
/// assert!(near_zero((T - mat4!(0.67901136, 0.14894516, 0.71885945, 1.2,
///                                     0.37320708, 0.77319584, -0.51272279, 5.4,
///                                     -0.63218672, 0.61642804, 0.46942137, 3.6,
///                                     0.0, 0.0, 0.0, 1.0)).abs().max()));
/// ```
pub fn project_to_se3(mat: SMatrix<f64, 4, 4>) -> SMatrix<f64, 4, 4> {
    let r = project_to_so3(mat.fixed_slice::<3, 3>(0, 0).into());
    rp_to_trans(r, Vector3::new(mat[(0, 3)], mat[(1, 3)], mat[(2, 3)]))
}

/// Computes the Frobenius norm to describe the distance of a matrix from the SO(3) manifold
/// # Arguments
/// * `mat` - A 3x3 matrix
/// # Returns
/// * A quantity describing the distance of mat from the SO(3) manifold
/// # Example
/// ```
/// use nalgebra::SMatrix;
/// use modern_robotics_rust::{distance_to_so3, near_zero};
/// use modern_robotics_rust::mat3;
/// let mat = mat3!(1.0, 0.0, 0.0, 0.0, 0.1, -0.95, 0.0, 1.0, 0.1);
/// let dist = distance_to_so3(mat);
/// assert!(near_zero(dist - 0.08835298523536149));
/// ```
pub fn distance_to_so3(mat: SMatrix<f64, 3, 3>) -> f64 {
    if mat.determinant() > 0.0 {
        (mat.transpose() * mat - SMatrix::<f64, 3, 3>::identity()).norm()
    } else {
        1e9
    }
}

/// Computes the Frobenius norm to describe the distance of a matrix from the SE(3) manifold
/// # Arguments
/// * `mat` - A 4x4 matrix
/// # Returns
/// * A quantity describing the distance of mat from the SE(3) manifold
/// # Example
/// ```
/// use nalgebra::SMatrix;
/// use modern_robotics_rust::{distance_to_se3, near_zero};
/// use modern_robotics_rust::mat4;
/// let mat = mat4!(1.0, 0.0, 0.0, 1.2, 0.0, 0.1, -0.95, 1.5, 0.0, 1.0, 0.1, -0.9, 0.0, 0.0, 0.1, 0.98);
/// println!("mat: {mat}");
/// let dist = distance_to_se3(mat);
/// println!("Distance to SE(3): {dist}");
/// assert!(near_zero(dist - 0.134931));
/// ```
pub fn distance_to_se3(mat: SMatrix<f64, 4, 4>) -> f64 {
    let mat_r = mat.fixed_slice::<3, 3>(0, 0);
    if mat_r.determinant() > 0.0 {
        (r_stack(
            &c_stack(&(mat_r.transpose() * mat_r).into(), &Vector3::zeros()),
            &mat.row(3).clone_owned(),
        ) - SMatrix::<f64, 4, 4>::identity())
        .norm()
    } else {
        1e9
    }
}

/// Returns true if a matrix is close to or on the manifold SO(3)
/// # Arguments
/// * `mat` - A 3x3 matrix
/// # Returns
/// * True if mat is very close to or in SO(3), false otherwise
/// # Example
/// ```
/// use nalgebra::SMatrix;
/// use modern_robotics_rust::{test_if_so3, near_zero};
/// use modern_robotics_rust::mat3;
/// let mat = mat3!(1.0, 0.0, 0.0, 0.0, 0.1, -0.95, 0.0, 1.0, 0.1);
/// let is_so3 = test_if_so3(mat);
/// assert_eq!(is_so3, false);
/// ```
pub fn test_if_so3(mat: SMatrix<f64, 3, 3>) -> bool {
    distance_to_so3(mat).abs() < 1e-3
}

/// Returns true if a matrix is close to or on the manifold SE(3)
/// # Arguments
/// * `mat` - A 4x4 matrix
/// # Returns
/// * True if mat is very close to or in SE(3), false otherwise
/// # Example
/// ```
/// use nalgebra::SMatrix;
/// use modern_robotics_rust::{test_if_se3, near_zero};
/// use modern_robotics_rust::mat4;
/// let mat = mat4!(1.0, 0.0, 0.0, 1.2, 0.0, 0.1, -0.95, 1.5, 0.0, 1.0, 0.1, -0.9, 0.0, 0.0, 0.1, 0.98);
/// let is_se3 = test_if_se3(mat);
/// assert_eq!(is_se3, false);
/// ```
pub fn test_if_se3(mat: SMatrix<f64, 4, 4>) -> bool {
    distance_to_se3(mat).abs() < 1e-3
}

// *** CHAPTER 4: FORWARD KINEMATICS ***

/// Computes forward kinematics in the body frame for an open chain robot
/// # Arguments
/// * `M` - The home configuration (position and orientation) of the end-effector
/// * `Blist` - The joint screw axes in the end-effector frame when the manipulator is at the home position, in the format of a matrix with axes as the columns
/// * `thetalist` - A list of joint coordinates
/// # Returns
/// * A homogeneous transformation matrix representing the end-effector frame when the joints are at the specified coordinates (i.t.o Body Frame)
/// # Example
/// ```
/// use nalgebra::SMatrix;
/// use nalgebra::Vector3;
/// use nalgebra::Vector6;
/// use modern_robotics_rust::{fkin_body, near_zero};
/// use modern_robotics_rust::vec_to_se3;
/// use modern_robotics_rust::mat4;
/// let M = mat4!(-1.0, 0.0, 0.0, 0.0,
///                                     0.0, 1.0, 0.0, 6.0,
///                                     0.0, 0.0, -1.0, 2.0,
///                                     0.0, 0.0, 0.0, 1.0);
/// let Blist = SMatrix::<f64, 3, 6>::from_row_slice(&[0.0, 0.0, -1.0, 2.0, 0.0, 0.0,
///                                     0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
///                                     0.0, 0.0, 1.0, 0.0, 0.0, 0.1]).transpose();
/// let thetalist = Vector3::new(std::f64::consts::PI / 2.0, 3.0, std::f64::consts::PI);
/// let T = fkin_body(M, Blist, thetalist);
/// assert!(near_zero((T- mat4!(0.0, 1.0, 0.0, -5.0,
///                                     1.0, 0.0, 0.0, 4.0,
///                                     0.0, 0.0, -1.0, 1.68584073,
///                                     0.0, 0.0, 0.0, 1.0)).abs().max()));
/// ```
// N is the number of joints, and the columns of Blist are the screw axes of the joints in the end-effector frame when the manipulator is at the home position
// therefore thetalist must have the N joint coordinates corresponding to the N screw axes in Blist
pub fn fkin_body<const N: usize>(
    m: SMatrix<f64, 4, 4>,
    blist: SMatrix<f64, 6, N>,
    thetalist: SVector<f64, N>,
) -> SMatrix<f64, 4, 4> {
    let mut t = m;
    for i in 0..thetalist.len() {
        t = t * matrix_exp6(vec_to_se3(blist.column(i) * thetalist[i]));
    }
    t
}

/// Computes forward kinematics in the space frame for an open chain robot
/// # Arguments
/// * `M` - The home configuration (position and orientation) of the end-effector
/// * `Slist` - The joint screw axes in the space frame when the manipulator is at the home position, in the format of a matrix with axes as the columns
/// * `thetalist` - A list of joint coordinates
/// # Returns
/// * A homogeneous transformation matrix representing the end-effector frame when the joints are at the specified coordinates (i.t.o Space Frame)
/// # Example
/// ```
/// use nalgebra::SMatrix;
/// use nalgebra::Vector3;
/// use nalgebra::Vector6;
/// use modern_robotics_rust::{fkin_space, near_zero};
/// use modern_robotics_rust::vec_to_se3;
/// use modern_robotics_rust::mat4;
/// let M = mat4!(-1.0, 0.0, 0.0, 0.0,
///                                     0.0, 1.0, 0.0, 6.0,
///                                     0.0, 0.0, -1.0, 2.0,
///                                     0.0, 0.0, 0.0, 1.0);
/// let Slist = SMatrix::<f64, 3, 6>::from_row_slice(&[0.0, 0.0, 1.0, 4.0, 0.0, 0.0,
///                                     0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
///                                     0.0, 0.0, -1.0, -6.0, 0.0, -0.1]).transpose();
/// let thetalist = Vector3::new(std::f64::consts::PI / 2.0, 3.0, std::f64::consts::PI);
/// let T = fkin_space(M, Slist, thetalist);
/// assert!(near_zero((T- mat4!(0.0, 1.0, 0.0, -5.0,
///                                     1.0, 0.0, 0.0, 4.0,
///                                     0.0, 0.0, -1.0, 1.68584073,
///                                     0.0, 0.0, 0.0, 1.0)).abs().max()));
/// ```
pub fn fkin_space<const N: usize>(
    m: SMatrix<f64, 4, 4>,
    slist: SMatrix<f64, 6, N>,
    thetalist: SVector<f64, N>,
) -> SMatrix<f64, 4, 4> {
    let mut t = m;
    for i in (0..thetalist.len()).rev() {
        t = matrix_exp6(vec_to_se3(slist.column(i) * thetalist[i])) * t;
    }
    t
}

// *** CHAPTER 5: VELOCITY KINEMATICS AND STATICS***

/// Computes the body Jacobian for an open chain robot
/// # Arguments
/// * `Blist` - The joint screw axes in the end-effector frame when the manipulator is at the home position, in the format of a matrix with axes as the columns
/// * `thetalist` - A list of joint coordinates
/// # Returns
/// * The body Jacobian corresponding to the inputs (6xn real numbers)
/// # Example
/// ```
/// use nalgebra::SMatrix;
/// use nalgebra::Vector4;
/// use nalgebra::Vector6;
/// use nalgebra::SVector;
/// use modern_robotics_rust::{jacobian_body, near_zero};
/// use modern_robotics_rust::vec_to_se3;
/// use modern_robotics_rust::mat4;
/// let Blist = SMatrix::<f64, 4, 6>::from_row_slice(&[0.0, 0.0, 1.0, 0.0, 0.2, 0.2,
///                                     1.0, 0.0, 0.0, 2.0, 0.0, 3.0,
///                                     0.0, 1.0, 0.0, 0.0, 2.0, 1.0,
///                                     1.0, 0.0, 0.0, 0.2, 0.3, 0.4]).transpose();
/// let thetalist = Vector4::new(0.2, 1.1, 0.1, 1.2);
/// let Jb = jacobian_body(Blist, thetalist);
/// assert!(near_zero((Jb - SMatrix::<f64, 6, 4>::from_row_slice(&[-0.04528405, 0.99500417, 0.0, 1.0,
///                                     0.74359313, 0.09304865, 0.36235775, 0.0,
///                                     -0.66709716, 0.03617541, -0.93203909, 0.0,
///                                     2.32586047, 1.66809, 0.56410831, 0.2,
///                                     -1.44321167, 2.94561275, 1.43306521, 0.3,
///                                     -2.06639565, 1.82881722, -1.58868628, 0.4])).abs().max()));
/// ```
pub fn jacobian_body<const N: usize>(
    blist: SMatrix<f64, 6, N>,
    thetalist: SVector<f64, N>,
) -> SMatrix<f64, 6, N> {
    let mut jb = blist;
    let mut t = SMatrix::<f64, 4, 4>::identity();
    for i in (0..thetalist.len() - 1).rev() {
        t = t * matrix_exp6(vec_to_se3(blist.column(i + 1) * -thetalist[i + 1]));
        jb.column_mut(i).copy_from(&(adjoint(t) * blist.column(i)));
    }
    jb
}

/// Computes the space Jacobian for an open chain robot
/// # Arguments
/// * `Slist` - The joint screw axes in the space frame when the manipulator is at the home position, in the format of a matrix with axes as the columns
/// * `thetalist` - A list of joint coordinates
/// # Returns
/// * The space Jacobian corresponding to the inputs (6xn real numbers)
/// # Example
/// ```
/// use nalgebra::SMatrix;
/// use nalgebra::Vector4;
/// use nalgebra::Vector6;
/// use nalgebra::SVector;
/// use modern_robotics_rust::{jacobian_space, near_zero};
/// use modern_robotics_rust::vec_to_se3;
/// use modern_robotics_rust::mat4;
/// let Slist = SMatrix::<f64, 4, 6>::from_row_slice(&[0., 0., 1.,   0., 0.2, 0.2,
///                           1., 0., 0.,   2.,   0.,   3.,
///                           0., 1., 0.,   0.,   2.,   1.,
///                           1., 0., 0., 0.2, 0.3, 0.4]).transpose();
/// let thetalist = Vector4::new(0.2, 1.1, 0.1, 1.2);
/// let Js = jacobian_space(Slist, thetalist);
/// println!("Js: {Js}");
/// assert!(near_zero((Js - SMatrix::<f64, 6, 4>::from_row_slice(&[0.0, 0.98006658, -0.09011564, 0.95749426,
///                                     0.0, 0.19866933, 0.4445544, 0.28487557,
///                                     1.0, 0.0, 0.89120736, -0.04528405,
///                                     0.0, 1.95218638, -2.21635216, -0.51161537,
///                                     0.2, 0.43654132, -2.43712573, 2.77535713,
///                                     0.2, 2.96026613, 3.23573065, 2.22512443])).abs().max()));
/// ```
pub fn jacobian_space<const N: usize>(
    slist: SMatrix<f64, 6, N>,
    thetalist: SVector<f64, N>,
) -> SMatrix<f64, 6, N> {
    let mut js = slist;
    let mut t = SMatrix::<f64, 4, 4>::identity();
    for i in 1..thetalist.len() {
        t = t * matrix_exp6(vec_to_se3(slist.column(i - 1) * thetalist[i - 1]));
        js.column_mut(i).copy_from(&(adjoint(t) * slist.column(i)));
    }
    js
}

// *** CHAPTER 6: INVERSE KINEMATICS ***

/// Computes inverse kinematics in the body frame for an open chain robot
/// # Arguments
/// * `Blist` - The joint screw axes in the end-effector frame when the manipulator is at the home position, in the format of a matrix with axes as the columns
/// * `M` - The home configuration of the end-effector
/// * `T` - The desired end-effector configuration Tsd
/// * `thetalist0` - An initial guess of joint angles that are close to satisfying Tsd
/// * `eomg` - A small positive tolerance on the end-effector orientation error. The returned joint angles must give an end-effector orientation error less than eom
/// * `ev` - A small positive tolerance on the end-effector linear position error. The returned joint angles must give an end-effector position error less than ev
/// # Returns
/// * A tuple containing thetalist, which are joint angles that achieve T within the specified tolerances, and success, which is a logical value where TRUE means that the function found a solution and FALSE means that it ran through the set number of maximum iterations without finding a solution within the tolerances eomg and ev.
/// # Example
/// ```
/// use nalgebra::SMatrix;
/// use nalgebra::Vector3;
/// use nalgebra::Vector6;
/// use nalgebra::SVector;
/// use modern_robotics_rust::{ikin_body, near_zero};
/// use modern_robotics_rust::vec_to_se3;
/// use modern_robotics_rust::mat4;
/// let Blist = SMatrix::<f64, 3, 6>::from_row_slice(&[0.0, 0.0, -1.0, 2.0, 0.0, 0.0,
///                                     0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
///                                     0.0, 0.0, 1.0, 0.0, 0.0, 0.1,
///  ]).transpose();
/// let M = mat4!(-1.0, 0.0, 0.0, 0.0,
///                                    0.0, 1.0, 0.0, 6.0,
///                                   0.0, 0.0, -1.0, 2.0,
///                                  0.0, 0.0, 0.0, 1.0);
/// let T = mat4!(0.0, 1.0, 0.0, -5.0,
///                                     1.0, 0.0, 0.0, 4.0,
///                                     0.0, 0.0, -1.0, 1.68584073,
///                                     0.0, 0.0, 0.0, 1.0);
/// let thetalist0 = Vector3::new(1.5, 2.5, 3.0);
/// let eomg = 0.01;
/// let ev = 0.001;
/// let (thetalist, success) = ikin_body(Blist, M, T, thetalist0, eomg, ev);
/// println!("thetalist: {thetalist}");
/// assert!(success);
/// assert!(near_zero((thetalist - Vector3::new(1.5707380419173325, 2.9996671257213467, 3.1415349462497315)).abs().max()));
/// ```
pub fn ikin_body<const N: usize>(
    blist: SMatrix<f64, 6, N>,
    m: SMatrix<f64, 4, 4>,
    t: SMatrix<f64, 4, 4>,
    thetalist0: SVector<f64, N>,
    eomg: f64,
    ev: f64,
) -> (SVector<f64, N>, bool) {
    let mut thetalist = thetalist0;
    let max_iterations = 20;
    let mut i = 0;
    let mut v_b = se3_to_vec(matrix_log6(trans_inv(fkin_body(m, blist, thetalist)) * t));
    let mut err = v_b.fixed_rows::<3>(0).norm() > eomg || v_b.fixed_rows::<3>(3).norm() > ev;
    while err && i < max_iterations {
        println!("{}", &smat_to_dmat(&jacobian_body(blist, thetalist,)));
        thetalist += dmat_to_smat(&pinv_lapack(&smat_to_dmat(&jacobian_body(
            blist, thetalist,
        )))) * v_b;
        i += 1;
        v_b = se3_to_vec(matrix_log6(trans_inv(fkin_body(m, blist, thetalist)) * t));
        err = v_b.fixed_rows::<3>(0).norm() > eomg || v_b.fixed_rows::<3>(3).norm() > ev;
    }
    (thetalist, !err)
}

/// Computes inverse kinematics in the space frame for an open chain robot
/// # Arguments
/// * `Slist` - The joint screw axes in the space frame when the manipulator is at the home position, in the format of a matrix with axes as the columns
/// * `M` - The home configuration of the end-effector
/// * `T` - The desired end-effector configuration Tsd
/// * `thetalist0` - An initial guess of joint angles that are close to satisfying Tsd
/// * `eomg` - A small positive tolerance on the end-effector orientation error. The returned joint angles must give an end-effector orientation error less than eom
/// * `ev` - A small positive tolerance on the end-effector linear position error. The returned joint angles must give an end-effector position error less than ev
/// # Returns
/// * A tuple containing thetalist, which are joint angles that achieve T within the specified tolerances, and success, which is a logical value where TRUE means that the function found a solution and FALSE means that it ran through the set number of maximum iterations without finding a solution within the tolerances eomg and ev.
/// Uses an iterative Newton-Raphson root-finding method.
/// The maximum number of iterations before the algorithm is terminated has been hardcoded in as a variable called maxiterations. It is set to 20 at the start of the function, but can be changed if needed.
/// # Example
/// ```
/// use nalgebra::SMatrix;
/// use nalgebra::Vector3;
/// use nalgebra::Vector6;
/// use nalgebra::SVector;
/// use modern_robotics_rust::{ikin_space, near_zero};
/// use modern_robotics_rust::vec_to_se3;
/// use modern_robotics_rust::mat4;
/// let s_list = SMatrix::<f64, 3, 6>::from_row_slice(&[0.0, 0.0, 1.0, 4.0, 0.0, 0.0,
///                                     0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
///                                     0.0, 0.0, -1.0, -6.0, 0.0, -0.1,
///  ]).transpose();
/// let m = mat4!(-1.0, 0.0, 0.0, 0.0,
///                                    0.0, 1.0, 0.0, 6.0,
///                                   0.0, 0.0, -1.0, 2.0,
///                                  0.0, 0.0, 0.0, 1.0);
/// let t = mat4!(0.0, 1.0, 0.0, -5.0,
///                                     1.0, 0.0, 0.0, 4.0,
///                                     0.0, 0.0, -1.0, 1.68584073,
///                                     0.0, 0.0, 0.0, 1.0);
/// let thetalist0 = Vector3::new(1.5, 2.5, 3.0);
/// let eomg = 0.01;
/// let ev = 0.001;
/// let (thetalist, success) = ikin_space(s_list, m, t, thetalist0, eomg, ev);
/// println!("thetalist: {thetalist}");
/// assert!(success);
/// assert!(near_zero((thetalist - Vector3::new(1.57073783, 2.99966384, 3.1415342)).abs().max()));
/// ```
pub fn ikin_space<const N: usize>(
    slist: SMatrix<f64, 6, N>,
    m: SMatrix<f64, 4, 4>,
    t: SMatrix<f64, 4, 4>,
    thetalist0: SVector<f64, N>,
    eomg: f64,
    ev: f64,
) -> (SVector<f64, N>, bool) {
    let mut thetalist = thetalist0;
    let max_iterations = 20;
    let mut i = 0;
    let mut t_sb = fkin_space(m, slist, thetalist);
    let mut v_s = adjoint(t_sb) * se3_to_vec(matrix_log6(trans_inv(t_sb) * t));
    let mut err = v_s.fixed_rows::<3>(0).norm() > eomg || v_s.fixed_rows::<3>(3).norm() > ev;
    while err && i < max_iterations {
        thetalist += dmat_to_smat(&pinv_lapack(&smat_to_dmat(&jacobian_space(
            slist, thetalist,
        )))) * v_s;
        i += 1;
        t_sb = fkin_space(m, slist, thetalist);
        v_s = adjoint(t_sb) * se3_to_vec(matrix_log6(trans_inv(t_sb) * t));
        err = v_s.fixed_rows::<3>(0).norm() > eomg || v_s.fixed_rows::<3>(3).norm() > ev;
    }
    (thetalist, !err)
}
