use nalgebra::SMatrix;
use crate::matrix_log6;

/// Computes s(t) for a cubic time scaling
/// # Arguments
/// * `tf` - Total time of the motion in seconds from rest to rest
/// * `t` - The current time t satisfying 0 < t < Tf
/// # Returns
/// The path parameter s(t) corresponding to a third-order
/// polynomial motion that begins and ends at zero velocity
/// # Example
/// ```
/// use modern_robotics_rust::near_zero;
/// use modern_robotics_rust::trajectory_gen::cubic_time_scaling;
/// let tf = 2.;
/// let t = 0.6;
/// let s = cubic_time_scaling(tf, t);
/// assert!(near_zero(s-0.216))
/// ```
pub fn cubic_time_scaling(tf: f64, t: f64) -> f64 {
    3. * (t / tf).powi(2) - 2. * (t / tf).powi(3)
}


/// Computes s(t) for a quintic time scaling
/// # Arguments
/// * `tf` - Total time of the motion in seconds from rest to rest
/// * `t` - The current time t satisfying 0 < t < Tf
/// # Returns
/// The path parameter s(t) corresponding to a fifth-order
/// polynomial motion that begins and ends at zero velocity and zero acceleration
/// # Example
/// ```
/// use modern_robotics_rust::near_zero;
/// use modern_robotics_rust::trajectory_gen::quintic_time_scaling;
/// let tf = 2.;
/// let t = 0.6;
/// let s = quintic_time_scaling(tf, t);
/// assert!(near_zero(s - 0.16308))
/// ```
pub fn quintic_time_scaling(tf: f64, t: f64) -> f64 {
    10. * (t / tf).powi(3) - 15. * (t / tf).powi(4) + 6. * (t / tf).powi(5)
}



/// Computes a straight-line trajectory in joint space
/// # Arguments
/// * `thetastart` - The initial joint variables
/// * `thetaend` - The final joint variables
/// * `tf` - Total time of the motion in seconds from rest to rest
/// * `n` - The number of points n > 1 (Start and stop) in the discrete
///           representation of the trajectory
/// * `method` - The time-scaling method, where 3 indicates cubic (third-
///              order polynomial) time scaling and 5 indicates quintic
///              (fifth-order polynomial) time scaling
/// # Returns
/// A trajectory as an n x m matrix, where each row is an m-vector
/// of joint variables at an instant in time. The first row is
/// thetastart and the nth row is thetaend . The elapsed time
/// between each row is tf / (n - 1)
/// # Example
/// ```
/// use modern_robotics_rust::trajectory_gen::joint_trajectory;
/// use nalgebra::SMatrix;
/// use modern_robotics_rust::near_zero;
/// let thetastart = nalgebra::SMatrix::<f64, 8, 1>::from_row_slice(&[1., 0., 0., 1., 1., 0.2, 0., 1.]);
/// let thetaend = SMatrix::<f64, 8, 1>::from_row_slice(&[1.2, 0.5, 0.6, 1.1, 2., 2., 0.9, 1.]);
/// let tf = 4.;
/// let n = 6;
/// let method = 3;
/// let traj = joint_trajectory(&thetastart, &thetaend, tf, n, method);
/// assert!(near_zero((traj - SMatrix::<f64, 6, 8>::from_row_slice(&[
///     1., 0., 0., 1., 1., 0.2, 0., 1.,
///     1.0208, 0.052, 0.0624, 1.0104, 1.104, 0.3872, 0.0936, 1.,
///     1.0704, 0.176, 0.2112, 1.0352, 1.352, 0.8336, 0.3168, 1.,
///     1.1296, 0.324, 0.3888, 1.0648, 1.648, 1.3664, 0.5832, 1.,
///     1.1792, 0.448, 0.5376, 1.0896, 1.896, 1.8128, 0.8064, 1.,
///     1.2, 0.5, 0.6, 1.1, 2., 2., 0.9, 1.,])).abs().max()))
/// ```
pub fn joint_trajectory<const M: usize, const N: usize>(
    thetastart: &SMatrix<f64, M, 1>,
    thetaend: &SMatrix<f64, M, 1>,
    tf: f64,
    n: usize,
    method: usize,
) -> SMatrix<f64, N, M>
{
    let timegap = tf / (n - 1) as f64;
    let mut traj = SMatrix::<f64, M, N>::zeros();
    for i in 0..n {
        let s = if method == 3 {
            cubic_time_scaling(tf, timegap * i as f64)
        } else {
            quintic_time_scaling(tf, timegap * i as f64)
        };
        traj.set_column(i, &(s * thetaend + (1. - s) * thetastart));
    }
    traj.transpose()
}



/// Computes a trajectory as a list of N SE(3) matrices corresponding to
/// the screw motion about a space screw axis
/// # Arguments
/// * `Xstart` - The initial end-effector configuration
/// * `Xend` - The final end-effector configuration
/// * `tf` - Total time of the motion in seconds from rest to rest
/// * `n` - The number of points n > 1 (Start and stop) in the discrete
///           representation of the trajectory
/// * `method` - The time-scaling method, where 3 indicates cubic (third-
///              order polynomial) time scaling and 5 indicates quintic
///              (fifth-order polynomial) time scaling
/// # Returns
/// The discretized trajectory as a list of N matrices in SE(3)
/// separated in time by Tf/(N-1). The first in the list is Xstart
/// and the Nth is Xend
/// # Example
/// ```
/// use modern_robotics_rust::trajectory_gen::screw_trajectory;
/// use nalgebra::SMatrix;
/// use modern_robotics_rust::{mat4, near_zero};
/// let x_start = SMatrix::<f64, 4, 4>::from_row_slice(&[
///     1., 0., 0., 1.,
///     0., 1., 0., 0.,
///     0., 0., 1., 1.,
///     0., 0., 0., 1.,
/// ]);
/// let x_end = SMatrix::<f64, 4, 4>::from_row_slice(&[
///     0., 0., 1., 0.1,
///     1., 0., 0., 0.,
///     0., 1., 0., 4.1,
///     0., 0., 0., 1.,
/// ]);
/// let tf = 5.;
/// let n = 4;
/// let method = 3;
/// let traj = screw_trajectory::<4>(&x_start, &x_end, tf, n, method);
/// println!("{}", traj[1]);
/// assert!(near_zero((traj[0] - x_start).abs().max()));
/// assert!(near_zero((traj[1] - SMatrix::<f64, 4, 4>::from_row_slice(&[
///     0.904, -0.25, 0.346, 0.441,
///     0.346, 0.904, -0.25, 0.529,
///     -0.25, 0.346, 0.904, 1.601,
///     0., 0., 0., 1.,
/// ])).abs().max() / 2000.));
/// assert!(near_zero((traj[2] - mat4!([0.346, -0.25, 0.904, -0.117],
///                    [0.904, 0.346, -0.25,  0.473],
///                    [-0.25, 0.904, 0.346,  3.274],
///                    [    0,     0,     0,      1])).abs().max() / 2000.));
/// assert!(near_zero((traj[3] - x_end).abs().max()));
/// ```
pub fn screw_trajectory<const N: usize>(
    x_start: &SMatrix<f64, 4, 4>,
    x_end: &SMatrix<f64, 4, 4>,
    tf: f64,
    n: usize,
    method: usize,
) -> [SMatrix<f64, 4, 4>; N]
{
    let timegap = tf / (n - 1) as f64;
    let mut traj = [SMatrix::<f64, 4, 4>::zeros(); N];
    for i in 0..n {
        let s = if method == 3 {
            cubic_time_scaling(tf, timegap * i as f64)
        } else {
            quintic_time_scaling(tf, timegap * i as f64)
        };
        traj[i] = x_start * (crate::matrix_exp6(
            matrix_log6(x_start.try_inverse().unwrap() * x_end) * s,
        ));
    }
    traj
}