use crate::dynamics::{euler_step, MAX_N};

/// Computes the joint control torques at a particular time instant
/// # Arguments
/// * `thetalist` - n-vector of joint variables
/// * `dthetalist` - n-vector of joint rates
/// * `eint` - n-vector of the time-integral of joint errors
/// * `g` - Gravity vector g
/// * `Mlist` - List of link frames {i} relative to {i-1} at the home
///             position
/// * `Glist` - Spatial inertia matrices Gi of the links
/// * `Slist` - Screw axes Si of the joints in a space frame, in the format
///             of a matrix with axes as the columns
/// * `thetalistd` - n-vector of reference joint variables
/// * `dthetalistd` - n-vector of reference joint velocities
/// * `ddthetalistd` - n-vector of reference joint accelerations
/// * `Kp` - The feedback proportional gain (identical for each joint)
/// * `Ki` - The feedback integral gain (identical for each joint)
/// * `Kd` - The feedback derivative gain (identical for each joint)
/// # Returns
/// The vector of joint forces/torques computed by the feedback
/// linearizing controller at the current instant
/// # Example
/// ```
/// use modern_robotics_rust::robot_control::computed_torque;
/// use nalgebra::SVector;
/// use nalgebra::SMatrix;
/// use nalgebra::Vector6;
/// use modern_robotics_rust::utils::*;
/// use modern_robotics_rust::m4;
/// use modern_robotics_rust::near_zero;
/// let thetalist = col3(0.1, 0.1, 0.1);
/// let dthetalist = col3(0.1, 0.2, 0.3);
/// let eint = col3(0.2, 0.2, 0.2);
/// let g = col3(0., 0., -9.8);
/// let M01 = m4!(
///     [1., 0., 0., 0.],
///     [0., 1., 0., 0.],
///     [0., 0., 1., 0.089159],
///     [0., 0., 0., 1.],
/// );
/// let M12 = m4!(
///     [0., 0., 1., 0.28],
///     [0., 1., 0., 0.13585],
///     [-1., 0., 0., 0.],
///     [0., 0., 0., 1.],
/// );
/// let M23 = m4!(
///     [1., 0., 0., 0.],
///     [0., 1., 0., -0.1197],
///     [0., 0., 1., 0.395],
///     [0., 0., 0., 1.],
/// );
/// let M34 = m4!(
///     [1., 0., 0., 0.],
///     [0., 1., 0., 0.],
///     [0., 0., 1., 0.14225],
///     [0., 0., 0., 1.],
/// );
/// let G1 = nalgebra::Matrix6::from_diagonal(&Vector6::new(0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7));
/// let G2 = nalgebra::Matrix6::from_diagonal(&Vector6::new(0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393));
/// let G3 = nalgebra::Matrix6::from_diagonal(&Vector6::new(0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275));
/// let Glist = [G1, G2, G3];
/// let Mlist = [M01, M12, M23, M34];
/// let Slist = SMatrix::<f64, 3, 6>::from_row_slice(&[
///     1., 0., 1., 0., 1., 0.,
///     0., 1., 0., -0.089, 0., 0.,
///     0., 1., 0., -0.089, 0., 0.425,
/// ]).transpose();
/// let thetalistd = col3(1., 1., 1.);
/// let dthetalistd = col3(2., 1.2, 2.);
/// let ddthetalistd = col3(0.1, 0.1, 0.1);
/// let Kp = 1.3;
/// let Ki = 1.2;
/// let Kd = 1.1;
/// let tau = computed_torque(
///     &thetalist,
///     &dthetalist,
///     &eint,
///     &g,
///     (&Mlist).into(),
///     &Glist,
///     &Slist,
///     &thetalistd,
///     &dthetalistd,
///     &ddthetalistd,
///     Kp,
///     Ki,
///     Kd,
/// );
/// println!("{tau}");
/// assert!(near_zero((tau - col3(133.00525246, -29.94223324, -3.03276856)).abs().max()));
/// ```
pub fn computed_torque<const N: usize>(
    thetalist: &nalgebra::SVector<f64, N>,
    dthetalist: &nalgebra::SVector<f64, N>,
    eint: &nalgebra::SVector<f64, N>,
    g: &nalgebra::SVector<f64, 3>,
    m_list: &[nalgebra::SMatrix<f64, 4, 4>; N + 1],
    g_list: &[nalgebra::Matrix6<f64>; N],
    s_list: &nalgebra::SMatrix<f64, 6, N>,
    thetalistd: &nalgebra::SVector<f64, N>,
    dthetalistd: &nalgebra::SVector<f64, N>,
    ddthetalistd: &nalgebra::SVector<f64, N>,
    k_p: f64,
    k_i: f64,
    k_d: f64,
) -> nalgebra::SVector<f64, N>
where
    [(); N + 1 ]: Sized, [(); MAX_N - N]: Sized
{
    let e = thetalistd - thetalist;
    let m = crate::dynamics:: mass_matrix(thetalist, m_list, g_list, s_list);
    let inv_dyn = crate::dynamics::inverse_dynamics(
        thetalist,
        dthetalist,
        ddthetalistd,
        g,
        &nalgebra::SVector::<f64, 6>::zeros(),
        m_list,
        g_list,
        s_list,
    );
    m * (k_p * e + k_i * (eint + e) + k_d * (dthetalistd - dthetalist)) + inv_dyn
}

// todo: debug
/// Simulates the computed torque controller over a given desired
/// trajectory
/// # Arguments
/// * `thetalist` - n-vector of initial joint variables
/// * `dthetalist` - n-vector of initial joint velocities
/// * `g` - Actual gravity vector g
/// * `f_tip_mat` - An N x 6 matrix of spatial forces applied by the end-
///                 effector (If there are no tip forces the user should
///                 input a zero and a zero matrix will be used)
/// * `m_list` - Actual list of link frames i relative to i-1 at the home
///              position
/// * `g_list` - Actual spatial inertia matrices Gi of the links
/// * `s_list` - Screw axes Si of the joints in a space frame, in the format
///              of a matrix with axes as the columns
/// * `thetamatd` - An Nxn matrix of desired joint variables from the
///                 reference trajectory
/// * `dthetamatd` - An Nxn matrix of desired joint velocities
/// * `ddthetamatd` - An Nxn matrix of desired joint accelerations
/// * `gtilde` - The gravity vector based on the model of the actual robot
///              (actual values given above)
/// * `m_tilde_list` - The link frame locations based on the model of the
///                    actual robot (actual values given above)
/// * `g_tilde_list` - The link spatial inertias based on the model of the
///                    actual robot (actual values given above)
/// * `Kp` - The feedback proportional gain (identical for each joint)
/// * `Ki` - The feedback integral gain (identical for each joint)
/// * `Kd` - The feedback derivative gain (identical for each joint)
/// * `dt` - The timestep between points on the reference trajectory
/// * `int_res` - Integration resolution is the number of times integration
///               (Euler) takes places between each time step. Must be an
///               integer value greater than or equal to 1
/// # Returns
/// * `tau_mat` - An Nxn matrix of the controllers commanded joint forces/
///               torques, where each row of n forces/torques corresponds
///               to a single time instant
/// * `theta_mat` - An Nxn matrix of actual joint angles
/// The end of this function plots all the actual and desired joint angles
/// using miniplot
/// # Example
/// ```
/// use modern_robotics_rust::robot_control::simulate_control;
/// use nalgebra::SVector;
/// use nalgebra::SMatrix;
/// use nalgebra::Vector6;
/// use modern_robotics_rust::utils::*;
/// use modern_robotics_rust::m4;
/// use modern_robotics_rust::near_zero;
/// let thetalist = col3(0.1, 0.1, 0.1);
/// let dthetalist = col3(0.1, 0.2, 0.3);
/// let g = col3(0., 0., -9.8);
/// let M01 = m4!(
///     [1., 0., 0., 0.],
///     [0., 1., 0., 0.],
///     [0., 0., 1., 0.089159],
///     [0., 0., 0., 1.],
/// );
/// let M12 = m4!(
///     [0., 0., 1., 0.28],
///     [0., 1., 0., 0.13585],
///     [-1., 0., 0., 0.],
///     [0., 0., 0., 1.],
/// );
/// let M23 = m4!(
///     [1., 0., 0., 0.],
///     [0., 1., 0., -0.1197],
///     [0., 0., 1., 0.395],
///     [0., 0., 0., 1.],
/// );
/// let M34 = m4!(
///     [1., 0., 0., 0.],
///     [0., 1., 0., 0.],
///     [0., 0., 1., 0.14225],
///     [0., 0., 0., 1.],
/// );
/// let G1 = nalgebra::Matrix6::from_diagonal(&Vector6::new(0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7));
/// let G2 = nalgebra::Matrix6::from_diagonal(&Vector6::new(0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393));
/// let G3 = nalgebra::Matrix6::from_diagonal(&Vector6::new(0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275));
/// let Glist = [G1, G2, G3];
/// let Mlist = [M01, M12, M23, M34];
/// let Slist = SMatrix::<f64, 3, 6>::from_row_slice(&[
///     1., 0., 1., 0., 1., 0.,
///     0., 1., 0., -0.089, 0., 0.,
///     0., 1., 0., -0.089, 0., 0.425,
/// ]).transpose();
/// const DT: f64 = 0.01;
/// // Create a trajectory to follow
/// let thetaend = col3(std::f64::consts::PI / 2., std::f64::consts::PI, 1.5 * std::f64::consts::PI);
/// const TF: f64 = 1.;
/// const N: usize = (1.0 * TF / DT) as usize;
/// let method = 5;
/// let traj: SMatrix<f64, N, 3> = modern_robotics_rust::trajectory_gen::joint_trajectory(&thetalist, &thetaend, TF, N, method);
/// let thetamatd = traj.clone();
/// let mut dthetamatd = [nalgebra::SVector::<f64, 3>::zeros(); N];
/// let mut ddthetamatd = [nalgebra::SVector::<f64, 3>::zeros(); N];
/// for i in 0..N - 1 {
///     dthetamatd[i + 1] = (thetamatd.row(i + 1) - thetamatd.row(i)).transpose() / DT;
///     ddthetamatd[i + 1] = (dthetamatd[i + 1] - dthetamatd[i]) / DT;
/// }
/// // Possibly wrong robot description (Example with 3 links)
/// let gtilde = col3(0.8, 0.2, -8.8);
/// let Mhat01 = m4!(
///     [1., 0., 0., 0.],
///     [0., 1., 0., 0.],
///     [0., 0., 1., 0.1],
///     [0., 0., 0., 1.],
/// );
/// let Mhat12 = m4!(
///     [0., 0., 1., 0.3],
///     [0., 1., 0., 0.2],
///     [-1., 0., 0., 0.],
///     [0., 0., 0., 1.],
/// );
/// let Mhat23 = m4!(
///     [1., 0., 0., 0.],
///     [0., 1., 0., -0.2],
///     [0., 0., 1., 0.4],
///     [0., 0., 0., 1.],
/// );
/// let Mhat34 = m4!(
///     [1., 0., 0., 0.],
///     [0., 1., 0., 0.],
///     [0., 0., 1., 0.2],
///     [0., 0., 0., 1.],
/// );
/// let Ghat1 = nalgebra::Matrix6::from_diagonal(&Vector6::new(0.1, 0.1, 0.1, 4., 4., 4.));
/// let Ghat2 = nalgebra::Matrix6::from_diagonal(&Vector6::new(0.3, 0.3, 0.1, 9., 9., 9.));
/// let Ghat3 = nalgebra::Matrix6::from_diagonal(&Vector6::new(0.1, 0.1, 0.1, 3., 3., 3.));
/// let gtildelist = [Ghat1, Ghat2, Ghat3];
/// let mtildelist = [Mhat01, Mhat12, Mhat23, Mhat34];
/// let ftipmat = [nalgebra::SVector::<f64, 6>::from_element(1.); N];
/// let k_p = 20.;
/// let k_i = 10.;
/// let k_d = 18.;
/// let int_res = 8;
/// let (tau_mat, theta_mat) = simulate_control(
///     &thetalist,
///     &dthetalist,
///     &g,
///     &ftipmat,
///     (&Mlist).into(),
///     &Glist,
///     &Slist,
///     &thetamatd,
///     &dthetamatd,
///     &ddthetamatd,
///     &gtilde,
///     (&mtildelist).into(),
///     &gtildelist,
///     k_p,
///     k_i,
///     k_d,
///     DT,
///     int_res,
/// );
/// //println!("{tau_mat}");
/// //println!("{theta_mat}");
/// ```
// N is the number of joints, T is the number of time steps in the reference trajectory
pub fn simulate_control<const N: usize, const T: usize>(
    thetalist: &nalgebra::SVector<f64, N>,
    dthetalist: &nalgebra::SVector<f64, N>,
    g: &nalgebra::SVector<f64, 3>,
    f_tip_mat: &[nalgebra::SVector<f64, 6>; T],
    m_list: &[nalgebra::SMatrix<f64, 4, 4>; N + 1],
    g_list: &[nalgebra::Matrix6<f64>; N],
    s_list: &nalgebra::SMatrix<f64, 6, N>,
    thetamatd: &nalgebra::SMatrix<f64, T, N>,
    dthetamatd: &[nalgebra::SVector<f64, N>; T],
    ddthetamatd: &[nalgebra::SVector<f64, N>; T],
    gtilde: &nalgebra::SVector<f64, 3>,
    m_tilde_list: &[nalgebra::SMatrix<f64, 4, 4>; N + 1],
    g_tilde_list: &[nalgebra::Matrix6<f64>; N],
    k_p: f64,
    k_i: f64,
    k_d: f64,
    dt: f64,
    int_res: usize,
) -> (Vec<nalgebra::SVector<f64, N>>, Vec<nalgebra::SVector<f64, N>>)
where
    [(); N + 1 ]: Sized, [(); MAX_N - N]: Sized
{
    let m = thetamatd.ncols();
    let mut thetacurrent = thetalist.clone();
    let mut dthetacurrent = dthetalist.clone();
    let mut eint = vec![0.; m];
    let mut tau_mat = vec![nalgebra::SVector::<f64, N>::zeros(); m];
    let mut theta_mat = vec![nalgebra::SVector::<f64, N>::zeros(); m];
    for i in 0..m {
        let tau_list = computed_torque(
            &thetacurrent,
            &dthetacurrent,
            &nalgebra::SVector::<f64, N>::from_column_slice(&eint),
            gtilde,
            m_tilde_list,
            g_tilde_list,
            s_list,
            &thetamatd.row(i).transpose(),
            &dthetamatd[i],
            &ddthetamatd[i],
            k_p,
            k_i,
            k_d,
        );
        for _ in 0..int_res {
            let ddthetalist = crate::dynamics::forward_dynamics(
                &thetacurrent,
                &dthetacurrent,
                &tau_list,
                g,
                &f_tip_mat[i],
                m_list,
                g_list,
                s_list,
            );
            let (theta_next, dtheta_next) = euler_step(&thetacurrent, &dthetacurrent, &ddthetalist, dt / int_res as f64);
            thetacurrent = theta_next;
            dthetacurrent = dtheta_next;
        }
        tau_mat[i] = tau_list;
        theta_mat[i] = thetacurrent;
        for j in 0..eint.len() {
            eint[j] += dt * (thetamatd.row(i)
                [j] -
                thetacurrent[j]);
        }
    }
    #[cfg(feature = "miniplot")]
    {
        // links = np.array(thetamat).shape[0]
        //         N = np.array(thetamat).shape[1]
        //         Tf = N * dt
        //         timestamp = np.linspace(0, Tf, N)
        //         for i in range(links):
        //             col = [np.random.uniform(0, 1), np.random.uniform(0, 1),
        //                    np.random.uniform(0, 1)]
        //             plt.plot(timestamp, thetamat[i, :], "-", color=col, \
        //                      label = ("ActualTheta" + str(i + 1)))
        //             plt.plot(timestamp, thetamatd[i, :], ".", color=col, \
        //                      label = ("DesiredTheta" + str(i + 1)))
        //         plt.legend(loc = 'upper left')
        //         plt.xlabel("Time")
        //         plt.ylabel("Joint Angles")
        //         plt.title("Plot of Actual and Desired Joint Angles")
        //         plt.show()
        let links = theta_mat[0].len();
        let timestamp: Vec<f64> = (0..T).map(|i| i as f64 * dt).collect();
        let mut plot = miniplot::MiniPlot::new("Joint Angles");
        for i in 0..links {
            let actual_theta: Vec<f64> = theta_mat.iter().map(|row| row[i]).collect();
            let desired_theta: Vec<f64> = thetamatd.column(i).iter().cloned().collect();
            let actuail_theta_points = actual_theta.iter().zip(timestamp.iter()).map(|(theta, t)| [*t, *theta]).collect::<Vec<[f64; 2]>>();
            let desired_theta_points = desired_theta.iter().zip(timestamp.iter()).map(|(theta, t)| [*t, *theta]).collect::<Vec<[f64; 2]>>();
            plot = plot.plot_points(actuail_theta_points).name(&format!("ActualTheta{}", i + 1));
            plot = plot.plot_points(desired_theta_points).name(&format!("DesiredTheta{}", i + 1));
        }
        plot = plot.xlabel("Time");
        plot = plot.ylabel("Joint Angles");
        // plot.set_title("Plot of Actual and Desired Joint Angles");
        plot.show();
    }
    (tau_mat, theta_mat)
}