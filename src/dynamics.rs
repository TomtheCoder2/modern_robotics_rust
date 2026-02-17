use crate::utils::{c_stack, r_stack};
use crate::{adjoint, matrix_exp6, trans_inv, vec_to_se3, vec_to_so3};

/// Calculate the 6x6 matrix [adV] of the given 6-vector
/// # Arguments
/// * `V` - A 6-vector spatial velocity
/// # Returns
/// The corresponding 6x6 matrix [adV]
/// Used to calculate the Lie bracket [V1, V2] = [adV1]V2
/// # Example
/// ```
/// use nalgebra::Vector6;
/// use modern_robotics_rust::dynamics::ad;
/// use modern_robotics_rust::near_zero;
/// let V = Vector6::new(1., 2., 3., 4., 5., 6.);
/// let adV = ad(&V);
/// assert!(near_zero((&adV - nalgebra::SMatrix::<f64, 6, 6>::new(
///     0., -3., 2., 0., 0., 0.,
///     3., 0., -1., 0., 0., 0.,
///     -2., 1., 0., 0., 0., 0.,
///     0., -6., 5., 0., -3., 2.,
///     6., 0., -4., 3., 0., -1.,
///     -5., 4., 0., -2., 1., 0.,
/// )).abs().max()));
/// ```
pub fn ad(v: &nalgebra::Vector6<f64>) -> nalgebra::SMatrix<f64, 6, 6> {
    let omgmat = vec_to_so3(v.fixed_rows::<3>(0).clone_owned().into());
    r_stack(
        &c_stack(&omgmat, &nalgebra::SMatrix::<f64, 3, 3>::zeros()),
        &c_stack(
            &vec_to_so3(v.fixed_rows::<3>(3).clone_owned().into()),
            &omgmat,
        ),
    )
}

pub const MAX_N: usize = 200;
/// Computes inverse dynamics in the space frame for an open chain robot
/// # Arguments
/// * `thetalist` - n-vector of joint variables
/// * `dthetalist` - n-vector of joint rates
/// * `ddthetalist` - n-vector of joint accelerations
/// * `g` - Gravity vector g
/// * `Ftip` - Spatial force applied by the end-effector expressed in frame
///             {n+1}
/// * `Mlist` - List of link frames {i} relative to {i-1} at the home
///              position
/// * `Glist` - Spatial inertia matrices Gi of the links
/// * `Slist` - Screw axes Si of the joints in a space frame, in the format
///              of a matrix with axes as the columns
/// # Returns
/// The n-vector of required joint forces/torques
/// This function uses forward-backward Newton-Euler iterations to solve the
/// equation:
/// taulist = Mlist(thetalist)ddthetalist + c(thetalist,dthetalist) \
///           + g(thetalist) + Jtr(thetalist)Ftip
/// # Example
/// ```
/// use nalgebra::{SMatrix, Vector3, Vector6};
/// use modern_robotics_rust::dynamics::inverse_dynamics;
/// use modern_robotics_rust::utils::*;
/// use modern_robotics_rust::mat4;
/// use modern_robotics_rust::near_zero;
/// let thetalist = Vector3::new(0.1, 0.1, 0.1);
/// let dthetalist = Vector3::new(0.1, 0.2, 0.3);
/// let ddthetalist = Vector3::new(2., 1.5, 1.);
/// let g = Vector3::new(0., 0., -9.8);
/// let Ftip = Vector6::new(1., 1., 1., 1., 1., 1.);
/// let M01 = mat4!(
///     [1., 0., 0.,        0.],
///     [0., 1., 0.,        0.],
///     [0., 0., 1., 0.089159],
///     [0., 0., 0.,        1.],
/// );
/// let M12 = mat4!(
///     [ 0., 0., 1.,    0.28],
///     [ 0., 1., 0., 0.13585],
///     [-1., 0., 0.,       0.],
///     [ 0., 0., 0.,       1.],
/// );
/// let M23 = mat4!(
///     [1., 0., 0.,       0.],
///     [0., 1., 0., -0.1197],
///     [0., 0., 1.,   0.395],
///     [0., 0., 0.,       1.],
/// );
/// let M34 = mat4!(
///     [1., 0., 0.,       0.],
///     [0., 1., 0.,       0.],
///     [0., 0., 1., 0.14225],
///     [0., 0., 0.,       1.],
/// );
/// let G1 = nalgebra::SMatrix::<f64, 6, 6>::from_diagonal(&Vector6::new(0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7));
/// let G2 = nalgebra::SMatrix::<f64, 6, 6>::from_diagonal(&Vector6::new(0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393));
/// let G3 = nalgebra::SMatrix::<f64, 6, 6>::from_diagonal(&Vector6::new(0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275));
/// let Glist = [G1, G2, G3];
/// let Mlist = [M01, M12, M23, M34];
/// let Slist = SMatrix::<f64, 3, 6>::from_row_slice(&[
///     1., 0., 1.,      0., 1.,     0.,
///                           0., 1., 0., -0.089, 0.,     0.,
///                           0., 1., 0., -0.089, 0., 0.425]).transpose();
/// let tau = inverse_dynamics(&thetalist, &dthetalist, &ddthetalist, &g, &Ftip, &Mlist, &Glist, &Slist);
/// println!("{}", tau);
/// assert!(near_zero((&tau - Vector3::new(74.69616155, -33.06766016, -3.23057314)).abs().max()));
/// ```
pub fn inverse_dynamics<const N: usize>(
    thetalist: &nalgebra::SVector<f64, N>,
    dthetalist: &nalgebra::SVector<f64, N>,
    ddthetalist: &nalgebra::SVector<f64, N>,
    g: &nalgebra::Vector3<f64>,
    f_tip: &nalgebra::Vector6<f64>,
    m_list: &[nalgebra::SMatrix<f64, 4, 4>],
    g_list: &[nalgebra::SMatrix<f64, 6, 6>],
    s_list: &nalgebra::SMatrix<f64, 6, N>,
) -> nalgebra::SVector<f64, N>
where
    [(); MAX_N - N]: Sized,
    [();  N + 1]: Sized,
{
    let mut m_i = nalgebra::SMatrix::<f64, 4, 4>::identity();
    let mut a_i = nalgebra::SMatrix::<f64, 6, N>::zeros();
    let mut ad_t_i = vec![nalgebra::SMatrix::<f64, 6, 6>::zeros(); N + 1];
    let mut v_i = nalgebra::SMatrix::<f64, 6, { N + 1 }>::zeros();
    let mut vd_i = nalgebra::SMatrix::<f64, 6, { N + 1 }>::zeros();
    vd_i.fixed_columns_mut::<1>(0)
        .copy_from(&nalgebra::SVector::<f64, 6>::new(
            0., 0., 0., -g.x, -g.y, -g.z,
        ));
    ad_t_i[N] = adjoint(trans_inv(m_list[N]));
    let mut f_i = f_tip.clone();
    let mut tau = nalgebra::SVector::<f64, N>::zeros();
    for i in 0..N {
        m_i = m_i * m_list[i];
        a_i.fixed_columns_mut::<1>(i)
            .copy_from(&(adjoint(trans_inv(m_i)) * s_list.fixed_columns::<1>(i)));
        ad_t_i[i] = adjoint(
            matrix_exp6(vec_to_se3(a_i.fixed_columns::<1>(i) * -thetalist[i]))
                * trans_inv(m_list[i]),
        );
        let temp = v_i.fixed_columns::<1>(i).clone_owned();
        v_i.fixed_columns_mut::<1>(i + 1)
            .copy_from(&(ad_t_i[i] * temp + a_i.fixed_columns::<1>(i) * dthetalist[i]));
        let temp = vd_i.fixed_columns::<1>(i).clone_owned();
        vd_i.fixed_columns_mut::<1>(i + 1).copy_from(
            &(ad_t_i[i] * temp
                + a_i.fixed_columns::<1>(i) * ddthetalist[i]
                + ad(&v_i.fixed_columns::<1>(i + 1).clone_owned())
                    * a_i.fixed_columns::<1>(i)
                    * dthetalist[i]),
        );
    }
    for i in (0..N).rev() {
        let temp = f_i.clone();
        f_i = ad_t_i[i + 1].transpose() * temp + g_list[i] * vd_i.fixed_columns::<1>(i + 1)
            - ad(&v_i.fixed_columns::<1>(i + 1).clone_owned()).transpose()
                * (g_list[i] * v_i.fixed_columns::<1>(i + 1));
        tau[i] = (f_i.transpose() * a_i.fixed_columns::<1>(i)).x;
    }
    tau
}

/// Computes the mass matrix of an open chain robot based on the given configuration
/// # Arguments
/// * `thetalist` - A list of joint variables
/// * `Mlist` - List of link frames i relative to i-1 at the home position
/// * `Glist` - Spatial inertia matrices Gi of the links
/// * `Slist` - Screw axes Si of the joints in a space frame, in the format of a matrix with axes as the columns
/// # Returns
/// The numerical inertia matrix M(thetalist) of an n-joint serial chain at the given configuration thetalist
/// This function calls InverseDynamics n times, each time passing a ddthetalist vector with a single element equal to one and all other inputs set to zero.
/// Each call of InverseDynamics generates a single column, and these columns are assembled to create the inertia matrix.
/// # Example
/// ```
/// use nalgebra::{SMatrix, Vector3};
/// use modern_robotics_rust::dynamics::mass_matrix;
/// use modern_robotics_rust::utils::*;
/// use modern_robotics_rust::mat4;
/// use modern_robotics_rust::mat3;
/// use modern_robotics_rust::near_zero;
/// use nalgebra::Vector6;
/// let thetalist = Vector3::new(0.1, 0.1, 0.1);
/// let M01 = mat4!(
///     [1., 0., 0.,        0.],
///     [0., 1., 0.,        0.],
///     [0., 0., 1., 0.089159],
///     [0., 0., 0.,        1.],
/// );
/// let M12 = mat4!(
///     [ 0., 0., 1.,    0.28],
///     [ 0., 1., 0., 0.13585],
///     [-1., 0., 0.,       0.],
///     [ 0., 0., 0.,       1.],
/// );
/// let M23 = mat4!(
///     [1., 0., 0.,       0.],
///     [0., 1., 0., -0.1197],
///     [0., 0., 1.,   0.395],
///     [0., 0., 0.,       1.],
/// );
/// let M34 = mat4!(
///     [1., 0., 0.,       0.],
///     [0., 1., 0.,       0.],
///     [0., 0., 1., 0.14225],
///     [0., 0., 0.,       1.],
/// );
/// let G1 = nalgebra::SMatrix::<f64, 6, 6>::from_diagonal(&Vector6::new(0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7));
/// let G2 = nalgebra::SMatrix::<f64, 6, 6>::from_diagonal(&Vector6::new(0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393));
/// let G3 = nalgebra::SMatrix::<f64, 6, 6>::from_diagonal(&Vector6::new(0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275));
/// let Glist = [G1, G2, G3];
/// let Mlist = [M01, M12, M23, M34];
/// let Slist = SMatrix::<f64, 3, 6>::from_row_slice(&[
///     1., 0., 1.,      0., 1.,     0.,
///                           0., 1., 0., -0.089, 0.,     0.,
///                           0., 1., 0., -0.089, 0., 0.425]).transpose();
/// let M = mass_matrix(&thetalist, &Mlist, &Glist, &Slist);
/// println!("{}", M);
/// assert!(near_zero((&M - mat3!(
///     22.543338, -0.307146754, -0.00718426391,
///     -0.307146754, 1.96850717, 0.432157368,
///     -0.00718426391, 0.432157368, 0.191630858
/// )).abs().max()));
/// ```
pub fn mass_matrix<const N: usize>(
    thetalist: &nalgebra::SVector<f64, N>,
    m_list: &[nalgebra::SMatrix<f64, 4, 4>],
    g_list: &[nalgebra::SMatrix<f64, 6, 6>],
    s_list: &nalgebra::SMatrix<f64, 6, N>,
) -> nalgebra::SMatrix<f64, N, N>
where
    [(); MAX_N - N]: Sized,
    [();  N + 1]: Sized,
{
    let mut m = nalgebra::SMatrix::<f64, N, N>::zeros();
    for i in 0..N {
        let mut ddthetalist = nalgebra::SVector::<f64, N>::zeros();
        ddthetalist[i] = 1.;
        m.fixed_columns_mut::<1>(i).copy_from(&inverse_dynamics(
            thetalist,
            &nalgebra::SVector::<f64, N>::zeros(),
            &ddthetalist,
            &nalgebra::Vector3::zeros(),
            &nalgebra::Vector6::zeros(),
            m_list,
            g_list,
            s_list,
        ));
    }
    m
}

/// Computes the Coriolis and centripetal terms in the inverse dynamics of an open chain robot
/// # Arguments
/// * `thetalist` - A list of joint variables,
/// * `dthetalist` - A list of joint rates,
/// * `Mlist` - List of link frames i relative to i-1 at the home position,
/// * `Glist` - Spatial inertia matrices Gi of the links,
/// * `Slist` - Screw axes Si of the joints in a space frame, in the format of a matrix with axes as the columns.
/// # Returns
/// The vector c(thetalist,dthetalist) of Coriolis and centripal terms for a given thetalist and dthetalist.
/// This function calls InverseDynamics with g = 0, Ftip = 0, and ddthetalist = 0
/// # Example
/// ```
/// use nalgebra::{SMatrix, Vector3};
/// use modern_robotics_rust::dynamics::vel_quadratic_forces;
/// use modern_robotics_rust::utils::*;
/// use modern_robotics_rust::mat4;
/// use modern_robotics_rust::mat3;
/// use modern_robotics_rust::near_zero;
/// use nalgebra::Vector6;
/// let thetalist = Vector3::new(0.1, 0.1, 0.1);
/// let dthetalist = Vector3::new(0.1, 0.2, 0.3);
/// let M01 = mat4!(
///     [1., 0., 0.,        0.],
///     [0., 1., 0.,        0.],
///     [0., 0., 1., 0.089159],
///     [0., 0., 0.,        1.],
/// );
/// let M12 = mat4!(
///     [ 0., 0., 1.,    0.28],
///     [ 0., 1., 0., 0.13585],
///     [-1., 0., 0.,       0.],
///     [ 0., 0., 0.,       1.],
/// );
/// let M23 = mat4!(
///     [1., 0., 0.,       0.],
///     [0., 1., 0., -0.1197],
///     [0., 0., 1.,   0.395],
///     [0., 0., 0.,       1.],
/// );
/// let M34 = mat4!(
///     [1., 0., 0.,       0.],
///     [0., 1., 0.,       0.],
///     [0., 0., 1., 0.14225],
///     [0., 0., 0.,       1.],
/// );
/// let G1 = nalgebra::SMatrix::<f64, 6, 6>::from_diagonal(&Vector6::new(0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7));
/// let G2 = nalgebra::SMatrix::<f64, 6, 6>::from_diagonal(&Vector6::new(0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393));
/// let G3 = nalgebra::SMatrix::<f64, 6, 6>::from_diagonal(&Vector6::new(0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275));
/// let Glist = [G1, G2, G3];
/// let Mlist = [M01, M12, M23, M34];
/// let Slist = SMatrix::<f64, 3, 6>::from_row_slice(&[
///     1., 0., 1.,      0., 1.,     0.,
///                           0., 1., 0., -0.089, 0.,     0.,
///                           0., 1., 0., -0.089, 0., 0.425]).transpose();
/// let c = vel_quadratic_forces(&thetalist, &dthetalist, &Mlist, &Glist, &Slist);
/// println!("{}", c);
/// assert!(near_zero((&c - Vector3::new(0.26453118, -0.05505157, -0.00689132)).abs().max()));
/// ```
pub fn vel_quadratic_forces<const N: usize>(
    thetalist: &nalgebra::SVector<f64, N>,
    dthetalist: &nalgebra::SVector<f64, N>,
    m_list: &[nalgebra::SMatrix<f64, 4, 4>],
    g_list: &[nalgebra::SMatrix<f64, 6, 6>],
    s_list: &nalgebra::SMatrix<f64, 6, N>,
) -> nalgebra::SVector<f64, N>
where
    [(); MAX_N - N]: Sized,
    [();  N + 1]: Sized,
{
    inverse_dynamics(
        thetalist,
        dthetalist,
        &nalgebra::SVector::<f64, N>::zeros(),
        &nalgebra::Vector3::zeros(),
        &nalgebra::Vector6::zeros(),
        m_list,
        g_list,
        s_list,
    )
}

/// Computes the joint forces/torques an open chain robot requires to overcome gravity at its configuration
/// # Arguments
/// * `thetalist` - A list of joint variables,
/// * `g` - 3-vector for gravitational acceleration,
/// * `Mlist` - List of link frames i relative to i-1 at the home position,
/// * `Glist` - Spatial inertia matrices Gi of the links,
/// * `Slist` - Screw axes Si of the joints in a space frame, in the format of a matrix with axes as the columns.
/// # Returns
/// The joint forces/torques required to overcome gravity at thetalist
/// This function calls InverseDynamics with Ftip = 0, dthetalist = 0, and ddthetalist = 0.
/// # Example
/// ```
/// use nalgebra::{SMatrix, Vector3};
/// use modern_robotics_rust::dynamics::gravity_forces;
/// use modern_robotics_rust::utils::*;
/// use modern_robotics_rust::mat4;
/// use modern_robotics_rust::mat3;
/// use modern_robotics_rust::near_zero;
/// use nalgebra::Vector6;
/// let thetalist = Vector3::new(0.1, 0.1, 0.1);
/// let g = Vector3::new(0., 0., -9.8);
/// let M01 = mat4!(
///     [1., 0., 0.,        0.],
///     [0., 1., 0.,        0.],
///     [0., 0., 1., 0.089159],
///     [0., 0., 0.,        1.],
/// );
/// let M12 = mat4!(
///     [ 0., 0., 1.,    0.28],
///     [ 0., 1., 0., 0.13585],
///     [-1., 0., 0.,       0.],
///     [ 0., 0., 0.,       1.],
/// );
/// let M23 = mat4!(
///     [1., 0., 0.,       0.],
///     [0., 1., 0., -0.1197],
///     [0., 0., 1.,   0.395],
///     [0., 0., 0.,       1.],
/// );
/// let M34 = mat4!(
///     [1., 0., 0.,       0.],
///     [0., 1., 0.,       0.],
///     [0., 0., 1., 0.14225],
///     [0., 0., 0.,       1.],
/// );
/// let G1 = nalgebra::SMatrix::<f64, 6, 6>::from_diagonal(&Vector6::new(0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7));
/// let G2 = nalgebra::SMatrix::<f64, 6, 6>::from_diagonal(&Vector6::new(0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393));
/// let G3 = nalgebra::SMatrix::<f64, 6, 6>::from_diagonal(&Vector6::new(0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275));
/// let Glist = [G1, G2, G3];
/// let Mlist = [M01, M12, M23, M34];
/// let Slist = SMatrix::<f64, 3, 6>::from_row_slice(&[
///     1., 0., 1.,      0., 1.,     0.,
///                           0., 1., 0., -0.089, 0.,     0.,
///                           0., 1., 0., -0.089, 0., 0.425]).transpose();
/// let grav = gravity_forces(&thetalist, &g, &Mlist, &Glist, &Slist);
/// println!("{}", grav);
/// assert!(near_zero((&grav - Vector3::new(28.40331262, -37.64094817, -5.4415892)).abs().max()));
/// ```
pub fn gravity_forces<const N: usize>(
    thetalist: &nalgebra::SVector<f64, N>,
    g: &nalgebra::Vector3<f64>,
    m_list: &[nalgebra::SMatrix<f64, 4, 4>],
    g_list: &[nalgebra::SMatrix<f64, 6, 6>],
    s_list: &nalgebra::SMatrix<f64, 6, N>,
) -> nalgebra::SVector<f64, N>
where
    [(); MAX_N - N]: Sized,
    [();  N + 1]: Sized,
{
    inverse_dynamics(
        thetalist,
        &nalgebra::SVector::<f64, N>::zeros(),
        &nalgebra::SVector::<f64, N>::zeros(),
        g,
        &nalgebra::Vector6::zeros(),
        m_list,
        g_list,
        s_list,
    )
}

/// Computes the joint forces/torques an open chain robot requires only to create the end-effector force Ftip
/// # Arguments
/// * `thetalist` - A list of joint variables,
/// * `Ftip` - Spatial force applied by the end-effector expressed in frame {n+1},
/// * `Mlist` - List of link frames i relative to i-1 at the home position,
/// * `Glist` - Spatial inertia matrices Gi of the links,
/// * `Slist` - Screw axes Si of the joints in a space frame, in the format of a matrix with axes as the columns.
/// # Returns
/// The joint forces and torques required only to create the end-effector force Ftip
/// This function calls InverseDynamics with g = 0, dthetalist = 0, and ddthetalist = 0.
/// # Example
/// ```
/// use nalgebra::{SMatrix, Vector3};
/// use modern_robotics_rust::dynamics::end_effector_forces;
/// use modern_robotics_rust::utils::*;
/// use modern_robotics_rust::mat4;
/// use modern_robotics_rust::mat3;
/// use modern_robotics_rust::near_zero;
/// use nalgebra::Vector6;
/// let thetalist = Vector3::new(0.1, 0.1, 0.1);
/// let Ftip = Vector6::new(1., 1., 1., 1., 1., 1.);
/// let M01 = mat4!(
///     [1., 0., 0.,        0.],
///     [0., 1., 0.,        0.],
///     [0., 0., 1., 0.089159],
///     [0., 0., 0.,        1.],
/// );
/// let M12 = mat4!(
///     [ 0., 0., 1.,    0.28],
///     [ 0., 1., 0., 0.13585],
///     [-1., 0., 0.,       0.],
///     [ 0., 0., 0.,       1.],
/// );
/// let M23 = mat4!(
///     [1., 0., 0.,       0.],
///     [0., 1., 0., -0.1197],
///     [0., 0., 1.,   0.395],
///     [0., 0., 0.,       1.],
/// );
/// let M34 = mat4!(
///     [1., 0., 0.,       0.],
///     [0., 1., 0.,       0.],
///     [0., 0., 1., 0.14225],
///     [0., 0., 0.,       1.],
/// );
/// let G1 = nalgebra::SMatrix::<f64, 6, 6>::from_diagonal(&Vector6::new(0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7));
/// let G2 = nalgebra::SMatrix::<f64, 6, 6>::from_diagonal(&Vector6::new(0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393));
/// let G3 = nalgebra::SMatrix::<f64, 6, 6>::from_diagonal(&Vector6::new(0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275));
/// let Glist = [G1, G2, G3];
/// let Mlist = [M01, M12, M23, M34];
/// let Slist = SMatrix::<f64, 3, 6>::from_row_slice(&[
///     1., 0., 1.,      0., 1.,     0.,
///                           0., 1., 0., -0.089, 0.,     0.,
///                           0., 1., 0., -0.089, 0., 0.425]).transpose();
/// let end_effector = end_effector_forces(&thetalist, &Ftip, &Mlist, &Glist, &Slist);
/// println!("{}", end_effector);
/// assert!(near_zero((&end_effector - Vector3::new(1.40954608, 1.85771497, 1.392409)).abs().max()));
/// ```
pub fn end_effector_forces<const N: usize>(
    thetalist: &nalgebra::SVector<f64, N>,
    f_tip: &nalgebra::Vector6<f64>,
    m_list: &[nalgebra::SMatrix<f64, 4, 4>],
    g_list: &[nalgebra::SMatrix<f64, 6, 6>],
    s_list: &nalgebra::SMatrix<f64, 6, N>,
) -> nalgebra::SVector<f64, N>
where
    [(); MAX_N - N]: Sized,
    [();  N + 1]: Sized,
{
    inverse_dynamics(
        thetalist,
        &nalgebra::SVector::<f64, N>::zeros(),
        &nalgebra::SVector::<f64, N>::zeros(),
        &nalgebra::Vector3::zeros(),
        f_tip,
        m_list,
        g_list,
        s_list,
    )
}

/// Computes the forward dynamics of an open chain robot
/// # Arguments
/// * `thetalist` - A list of joint variables,
/// * `dthetalist` - A list of joint rates,
/// * `taulist` - An n-vector of joint forces/torques,
/// * `g` - Gravity vector g,
/// * `Ftip` - Spatial force applied by the end-effector expressed in frame {n+1},
/// * `Mlist` - List of link frames i relative to i-1 at the home position,
/// * `Glist` - Spatial inertia matrices Gi of the links,
/// * `Slist` - Screw axes Si of the joints in a space frame, in the format of a matrix with axes as the columns.
/// # Returns
/// The resulting joint accelerations
/// This function computes ddthetalist by solving:
/// Mlist(thetalist) * ddthetalist = taulist - c(thetalist,dthetalist) - g(thetalist) - Jtr(thetalist) * Ftip
/// # Example
/// ```
/// use nalgebra::{SMatrix, Vector3};
/// use modern_robotics_rust::dynamics::forward_dynamics;
/// use modern_robotics_rust::utils::*;
/// use modern_robotics_rust::mat4;
/// use modern_robotics_rust::mat3;
/// use modern_robotics_rust::near_zero;
/// use nalgebra::Vector6;
/// let thetalist = Vector3::new(0.1, 0.1, 0.1);
/// let dthetalist = Vector3::new(0.1, 0.2, 0.3);
/// let taulist = Vector3::new(0.5, 0.6, 0.7);
/// let g = Vector3::new(0., 0., -9.8);
/// let Ftip = Vector6::new(1., 1., 1., 1., 1., 1.);
/// let M01 = mat4!(
///     [1., 0., 0.,        0.],
///     [0., 1., 0.,        0.],
///     [0., 0., 1., 0.089159],
///     [0., 0., 0.,        1.],
/// );
/// let M12 = mat4!(
///     [ 0., 0., 1.,    0.28],
///     [ 0., 1., 0., 0.13585],
///     [-1., 0., 0.,       0.],
///     [ 0., 0., 0.,       1.],
/// );
/// let M23 = mat4!(
///     [1., 0., 0.,       0.],
///     [0., 1., 0., -0.1197],
///     [0., 0., 1.,   0.395],
///     [0., 0., 0.,       1.],
/// );
/// let M34 = mat4!(
///     [1., 0., 0.,       0.],
///     [0., 1., 0.,       0.],
///     [0., 0., 1., 0.14225],
///     [0., 0., 0.,       1.],
/// );
/// let G1 = nalgebra::SMatrix::<f64, 6, 6>::from_diagonal(&Vector6::new(0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7));
/// let G2 = nalgebra::SMatrix::<f64, 6, 6>::from_diagonal(&Vector6::new(0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393));
/// let G3 = nalgebra::SMatrix::<f64, 6, 6>::from_diagonal(&Vector6::new(0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275));
/// let Glist = [G1, G2, G3];
/// let Mlist = [M01, M12, M23, M34];
/// let Slist = SMatrix::<f64, 3, 6>::from_row_slice(&[
///     1., 0., 1.,      0., 1.,     0.,
///                           0., 1., 0., -0.089, 0.,     0.,
///                           0., 1., 0., -0.089, 0., 0.425]).transpose();
/// let ddthetalist = forward_dynamics(&thetalist, &dthetalist, &taulist, &g, &Ftip, &Mlist, &Glist, &Slist);
/// println!("{}", ddthetalist);
/// assert!(near_zero((&ddthetalist - Vector3::new(-0.97392907, 25.58466784, -32.91499212)).abs().max()));
/// ```
pub fn forward_dynamics<const N: usize>(
    thetalist: &nalgebra::SVector<f64, N>,
    dthetalist: &nalgebra::SVector<f64, N>,
    taulist: &nalgebra::SVector<f64, N>,
    g: &nalgebra::Vector3<f64>,
    f_tip: &nalgebra::Vector6<f64>,
    m_list: &[nalgebra::SMatrix<f64, 4, 4>],
    g_list: &[nalgebra::SMatrix<f64, 6, 6>],
    s_list: &nalgebra::SMatrix<f64, 6, N>,
) -> nalgebra::SVector<f64, N>
where
    [(); MAX_N - N]: Sized,
    [();  N + 1]: Sized,
{
    let m = mass_matrix(thetalist, m_list, g_list, s_list);
    let c = vel_quadratic_forces(thetalist, dthetalist, m_list, g_list, s_list);
    let grav = gravity_forces(thetalist, g, m_list, g_list, s_list);
    let end_eff = end_effector_forces(thetalist, f_tip, m_list, g_list, s_list);
    m.try_inverse()
        .unwrap_or_else(|| panic!("Mass matrix is singular at thetalist = {}", thetalist))
        * (taulist - c - grav - end_eff)
}

/// Computes the joint angles and velocities at the next timestep using first order Euler integration
/// # Arguments
/// * `thetalist` - n-vector of joint variables
/// * `dthetalist` - n-vector of joint rates
/// * `ddthetalist` - n-vector of joint accelerations
/// * `dt` - The timestep delta t
/// # Returns
/// * `thetalist_next` - Vector of joint variables after dt from first order Euler integration
/// * `dthetalist_next` - Vector of joint rates after dt from first order Euler integration
/// # Example
/// ```
/// use nalgebra::Vector3;
/// use modern_robotics_rust::dynamics::euler_step;
/// let thetalist = Vector3::new(0.1, 0.1, 0.1);
/// let dthetalist = Vector3::new(0.1, 0.2, 0.3);
/// let ddthetalist = Vector3::new(2., 1.5, 1.);
/// let dt = 0.1;
/// let (thetalist_next, dthetalist_next) = euler_step(&thetalist, &dthetalist, &ddthetalist, dt);
/// println!("thetalist_next:\n{}", thetalist_next);
/// println!("dthetalist_next:\n{}", dthetalist_next);
/// assert!((thetalist_next - Vector3::new(0.11, 0.12, 0.13)).abs().max() < 1e-6);
/// assert!((dthetalist_next - Vector3::new(0.3, 0.35, 0.4)).abs().max() < 1e-6);
/// ```
pub fn euler_step<const N: usize>(
    thetalist: &nalgebra::SVector<f64, N>,
    dthetalist: &nalgebra::SVector<f64, N>,
    ddthetalist: &nalgebra::SVector<f64, N>,
    dt: f64,
) -> (nalgebra::SVector<f64, N>, nalgebra::SVector<f64, N>)
where
    [(); MAX_N - N]: Sized,
    [();  N + 1]: Sized,
{
    (thetalist + dt * dthetalist, dthetalist + dt * ddthetalist)
}

/// Calculates the joint forces/torques required to move the serial chain
/// along the given trajectory using inverse dynamics
/// # Arguments
/// * `thetamat` - An N x n matrix of robot joint variables
/// * `dthetamat` - An N x n matrix of robot joint velocities
/// * `ddthetamat` - An N x n matrix of robot joint accelerations
/// * `g` - Gravity vector g
/// * `Ftipmat` - An N x 6 matrix of spatial forces applied by the end-effector (If there are no tip forces the user should
///                 input a zero and a zero matrix will be used)
/// * `Mlist` - List of link frames i relative to i-1 at the home position
/// * `Glist` - Spatial inertia matrices Gi of the links
/// * `Slist` - Screw axes Si of the joints in a space frame, in the format of a matrix with axes as the columns
/// # Returns
/// The N x n matrix of joint forces/torques for the specified trajectory, where each of the N rows is the vector of joint forces/torques at each time step
pub fn inverse_dynamics_trajectory<const N: usize>(
    thetamat: &[nalgebra::SVector<f64, N>],
    dthetamat: &[nalgebra::SVector<f64, N>],
    ddthetamat: &[nalgebra::SVector<f64, N>],
    g: &nalgebra::Vector3<f64>,
    f_tipmat: &[nalgebra::Vector6<f64>],
    m_list: &[nalgebra::SMatrix<f64, 4, 4>],
    g_list: &[nalgebra::SMatrix<f64, 6, 6>],
    s_list: &nalgebra::SMatrix<f64, 6, N>,
) -> Vec<nalgebra::SVector<f64, N>>
where
    [(); MAX_N - N]: Sized,
    [(); N + 1]: Sized,
{
    thetamat
        .iter()
        .zip(dthetamat.iter())
        .zip(ddthetamat.iter())
        .zip(f_tipmat.iter())
        .map(|(((thetalist, dthetalist), ddthetalist), f_tip)| {
            inverse_dynamics(
                thetalist,
                dthetalist,
                ddthetalist,
                g,
                f_tip,
                m_list,
                g_list,
                s_list,
            )
        })
        .collect()
}

/// Simulates the motion of a serial chain given an open-loop history of joint forces/torques
/// # Arguments
/// * `thetalist` - n-vector of initial joint variables
/// * `dthetalist` - n-vector of initial joint rates
/// * `taumat` - An N x n matrix of joint forces/torques, where each row is the joint effort at any time step
/// * `g` - Gravity vector g
/// * `Ftipmat` - An N x 6 matrix of spatial forces applied by the end-effector (If there are no tip forces the user should input a zero and a zero matrix will be used)
/// * `Mlist` - List of link frames {i} relative to {i-1} at the home position
/// * `Glist` - Spatial inertia matrices Gi of the links
/// * `Slist` - Screw axes Si of the joints in a space frame, in the format of a matrix with axes as the columns
/// * `dt` - The timestep between consecutive joint forces/torques
/// * `int_res` - Integration resolution is the number of times integration (Euler) takes places between each time step. Must be an integer value greater than or equal to 1
/// # Returns
/// * `thetamat` - The N x n matrix of robot joint angles resulting from the specified joint forces/torques
/// * `dthetamat` - The N x n matrix of robot joint velocities
/// This function calls a numerical integration procedure that uses ForwardDynamics.
pub fn forward_dynamics_trajectory<const N: usize>(
    thetalist: &nalgebra::SVector<f64, N>,
    dthetalist: &nalgebra::SVector<f64, N>,
    taumat: &[nalgebra::SVector<f64, N>],
    g: &nalgebra::Vector3<f64>,
    f_tipmat: &[nalgebra::Vector6<f64>],
    m_list: &[nalgebra::SMatrix<f64, 4, 4>; N + 1],
    g_list: &[nalgebra::SMatrix<f64, 6, 6>; N],
    s_list: &nalgebra::SMatrix<f64, 6, N>,
    dt: f64,
    int_res: usize,
) -> (
    Vec<nalgebra::SVector<f64, N>>,
    Vec<nalgebra::SVector<f64, N>>,
)
where
    [(); MAX_N - N]: Sized,
    [(); N + 1]: Sized,
{
    let mut thetamat = vec![nalgebra::SVector::<f64, N>::zeros(); taumat.len()];
    let mut dthetamat = vec![nalgebra::SVector::<f64, N>::zeros(); taumat.len()];
    thetamat[0] = *thetalist;
    dthetamat[0] = *dthetalist;
    for i in 0..taumat.len() - 1 {
        let mut thetalist = thetamat[i];
        let mut dthetalist = dthetamat[i];
        for _ in 0..int_res {
            let ddthetalist = forward_dynamics(
                &thetalist,
                &dthetalist,
                &taumat[i],
                g,
                &f_tipmat[i],
                m_list,
                g_list,
                s_list,
            );
            let (thetalist_next, dthetalist_next) =
                euler_step(&thetalist, &dthetalist, &ddthetalist, dt / int_res as f64);
            thetalist = thetalist_next;
            dthetalist = dthetalist_next;
        }
        thetamat[i + 1] = thetalist;
        dthetamat[i + 1] = dthetalist;
    }
    (thetamat, dthetamat)
}
