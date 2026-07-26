"""EKF update step for the MSCKF: chi-square outlier gating on the null-space
projected residual, optional QR measurement compression, and the Joseph-form
covariance/state update.

Error-state convention matches msckf_state.py/measurement_model.py exactly:
delta_theta is a GLOBAL perturbation with R_true = (I + [delta_theta]x) @ R_hat.
"""
import numpy as np
from scipy.stats import chi2

from imu_propagation import GRAVITY_WORLD
from msckf_state import MSCKFState, N_CLONE_ERROR, N_IMU_ERROR
from quaternion_utils import axis_angle_to_quat, quat_multiply, quaternion_to_rotation_matrix, skew_symmetric

DEFAULT_CONFIDENCE = 0.95


def chi_square_threshold(dof, confidence=DEFAULT_CONFIDENCE):
    """Chi-square critical value for `dof` degrees of freedom at the given confidence level."""
    return chi2.ppf(confidence, dof)


def mahalanobis_distance(r_o, H_o, P, observation_noise_std):
    """Squared Mahalanobis distance of the (null-space-projected) residual under the current covariance.

    observation_noise_std is in normalized image-plane units (bearing units,
    not raw pixels) -- roughly pixel_noise_std / focal_length.
    """
    S = H_o @ P @ H_o.T + observation_noise_std ** 2 * np.eye(len(r_o))
    return r_o @ np.linalg.solve(S, r_o)


def passes_chi_square_gate(r_o, H_o, P, observation_noise_std, confidence=DEFAULT_CONFIDENCE):
    """True if the residual is statistically consistent with the current state estimate (not an outlier)."""
    gamma = mahalanobis_distance(r_o, H_o, P, observation_noise_std)
    return gamma <= chi_square_threshold(len(r_o), confidence)


def compress_measurement(r_o, H_o):
    """QR-compress an over-tall (rows > columns) measurement system for a cheaper update.

    Mathematically lossless for the EKF update: the discarded component lies
    in the orthogonal complement of H_o's range and (if the model is correct)
    is pure noise uncorrelated with the state, carrying no information about
    delta_x. No-ops when H_o isn't over-tall.
    """
    n_rows, n_cols = H_o.shape
    if n_rows <= n_cols:
        return r_o, H_o
    Q, R = np.linalg.qr(H_o, mode="reduced")
    return Q.T @ r_o, R


def ekf_update(state, r_o, H_o, observation_noise_std, min_variance=None):
    """Apply one EKF measurement update (Joseph form). Returns a new MSCKFState.

    min_variance, if given, is a length-N_IMU_ERROR array of minimum variances
    (std^2) for the current IMU state's 15 diagonal entries (theta, b_g, v,
    b_a, p, in that order) -- a covariance floor, additive so P stays exactly
    PSD (adding a nonnegative diagonal matrix can only raise eigenvalues, never
    lower them). This is regularization in the classic Tikhonov/ridge sense:
    it exists to stop P from becoming falsely overconfident (collapsing below
    what the filter can actually justify) after a long run of individually
    plausible-looking updates, each of which is locally well-justified but
    whose cumulative shrinkage leaves no "correction budget" once real drift
    needs correcting -- see demos/demo_divergence_analysis.py, which measured
    exactly this: bias/velocity covariance collapsing to a near-zero floor
    well before real error started to diverge.
    """
    r_o, H_o = compress_measurement(r_o, H_o)

    P = state.P
    n = P.shape[0]
    R_noise = observation_noise_std ** 2 * np.eye(len(r_o))
    S = H_o @ P @ H_o.T + R_noise
    K = P @ H_o.T @ np.linalg.inv(S)

    # measurement_model.py's H is d(r)/d(x) = d(obs - h)/d(x) = -dh/dx, the
    # opposite sign from the textbook Kalman-gain convention (H = dh/dx).
    # K/S/the Joseph-form P_new below are all invariant to this sign choice
    # (they only ever appear as the product K@H), but delta_x = K@r would
    # come out exactly negated without this correction.
    delta_x = -(K @ r_o)

    I_KH = np.eye(n) - K @ H_o
    P_new = I_KH @ P @ I_KH.T + K @ R_noise @ K.T
    P_new = 0.5 * (P_new + P_new.T)

    if min_variance is not None:
        current_diag = np.diag(P_new)[:N_IMU_ERROR]
        deficit = np.maximum(0.0, min_variance - current_diag)
        if np.any(deficit > 0):
            P_new[:N_IMU_ERROR, :N_IMU_ERROR] += np.diag(deficit)

    return _apply_error_state(state, delta_x, P_new)


def zero_velocity_update(state, zupt_noise_std, min_variance=None):
    """Apply a synthetic "velocity = 0" measurement (a ZUPT, zero-velocity update).

    Standard technique for stationary periods: when the platform isn't
    moving, the camera gets zero parallax on everything it sees, so vision
    updates are uninformative (or worse -- see msckf_pipeline.py's stationary
    detector docstring for why a vision-based parallax check isn't a safe
    substitute for this). Unaided IMU dead-reckoning then just integrates
    whatever residual accel bias error remains, drifting position with no
    corrective signal available. A ZUPT sidesteps this by asserting the one
    thing independently known to be true while stationary -- velocity is
    zero -- directly, without relying on triangulation at all.

    h(x) = v (identity observation of velocity); using this module's H = d(r)/dx
    = -dh/dx convention (see ekf_update's docstring above), the velocity
    columns of H_o are -I and everything else is zero; r_o = 0 - v_hat = -v_hat.

    Deliberately NOT a plain call to ekf_update: the mean correction (delta_x)
    is restricted to the velocity block only, discarding whatever a full EKF
    update's Kalman gain would otherwise apply to every other state through
    P's cross-correlations (P's covariance update itself is left standard --
    only the mean is restricted). Found necessary empirically: a real incident
    on this dataset had a single confident ZUPT application drag gyro bias by
    ~0.06 rad/s over a few seconds, via a b_g-v correlation built up earlier
    during ordinary vision-based operation that (like everything else this
    filter's self-referential-triangulation problem touches) wasn't well
    calibrated. That's actively dangerous here specifically: the stationarity
    detector itself bias-corrects raw gyro using this same state.b_g, so a
    corrupted b_g made the detector see "false rotation" during a period that
    was, per ground truth, still genuinely stationary -- causing ZUPT to shut
    itself off for the ~13s it was needed most. A ZUPT should only ever be
    allowed to assert what it actually measures.
    """
    n = state.P.shape[0]
    H_o = np.zeros((3, n))
    H_o[:, 6:9] = -np.eye(3)
    r_o = -state.v

    P = state.P
    R_noise = zupt_noise_std ** 2 * np.eye(3)
    S = H_o @ P @ H_o.T + R_noise
    K = P @ H_o.T @ np.linalg.inv(S)
    delta_x = np.zeros(n)
    delta_x[6:9] = -(K @ r_o)[6:9]

    I_KH = np.eye(n) - K @ H_o
    P_new = I_KH @ P @ I_KH.T + K @ R_noise @ K.T
    P_new = 0.5 * (P_new + P_new.T)

    if min_variance is not None:
        current_diag = np.diag(P_new)[:N_IMU_ERROR]
        deficit = np.maximum(0.0, min_variance - current_diag)
        if np.any(deficit > 0):
            P_new[:N_IMU_ERROR, :N_IMU_ERROR] += np.diag(deficit)

    return _apply_error_state(state, delta_x, P_new)


def zero_angular_rate_update(state, gyro_measured, zaru_noise_std, min_variance=None):
    """Apply a synthetic "gyro bias = raw gyro reading" measurement (a ZARU, zero
    angular-rate update) -- the standard companion to a ZUPT for stationary periods.

    When the platform truly isn't rotating, the true angular velocity is zero, so
    the raw gyro reading is pure bias plus noise (gyro_measured = 0 + b_g + noise)
    -- a direct, independent measurement of b_g. Unlike zero_velocity_update, this
    doesn't need its mean restricted to one block: it's not leaning on any
    cross-correlation to reach bias, it *is* a bias measurement, so there's
    nothing spurious for it to launder through P.

    h(x) = b_g (identity observation); using this module's H = d(r)/dx = -dh/dx
    convention, the b_g columns of H_o are -I and everything else is zero;
    r_o = gyro_measured - b_g_hat.
    """
    n = state.P.shape[0]
    H_o = np.zeros((3, n))
    H_o[:, 3:6] = -np.eye(3)
    r_o = np.asarray(gyro_measured) - state.b_g
    return ekf_update(state, r_o, H_o, zaru_noise_std, min_variance=min_variance)


def gravity_alignment_update(state, accel_measured, tilt_noise_std, min_variance=None):
    """Apply a synthetic "measured specific force matches gravity" measurement --
    the standard static-alignment/leveling update for stationary periods,
    completing the ZUPT+ZARU trio.

    When the platform truly has no linear acceleration, the raw accelerometer
    reading is fixed by orientation and bias alone (see imu_propagation.py's
    propagate_step, which this inverts):
        accel_measured = R(q)^T @ (-GRAVITY_WORLD) + b_a + noise
    This directly measures roll/pitch (rotation about the gravity vector itself
    changes nothing here, so yaw stays correctly unobservable) and accel bias --
    again a direct measurement, not something inferred through a correlation.

    Jacobian (re-derived here, not transcribed, given this project's history of
    sign bugs in exactly this kind of derivation -- verified against a finite-
    difference check in tests/test_msckf_update.py): using R_true = (I + [dtheta]x)
    R_hat, h_true - h_hat = -R_hat^T @ [GRAVITY_WORLD]x @ dtheta + delta_b_a, so
    dh/d(dtheta) = -R_hat^T @ skew(GRAVITY_WORLD) and dh/d(delta_b_a) = I. Using
    this module's H = -dh/dx convention: H_o's theta columns are
    R_hat^T @ skew(GRAVITY_WORLD), and its b_a columns are -I.
    """
    R_hat = quaternion_to_rotation_matrix(state.q)
    h = R_hat.T @ (-GRAVITY_WORLD) + state.b_a
    r_o = np.asarray(accel_measured) - h

    n = state.P.shape[0]
    H_o = np.zeros((3, n))
    H_o[:, 0:3] = R_hat.T @ skew_symmetric(GRAVITY_WORLD)
    H_o[:, 9:12] = -np.eye(3)
    return ekf_update(state, r_o, H_o, tilt_noise_std, min_variance=min_variance)


def _apply_error_state(state, delta_x, P_new):
    d_theta, d_bg, d_v, d_ba, d_p = (
        delta_x[0:3], delta_x[3:6], delta_x[6:9], delta_x[9:12], delta_x[12:15])

    q_new = quat_multiply(axis_angle_to_quat(d_theta), state.q)
    q_new = q_new / np.linalg.norm(q_new)

    clone_positions, clone_orientations = [], []
    for i, (q_cam, p_cam) in enumerate(zip(state.clone_orientations, state.clone_positions)):
        offset = N_IMU_ERROR + N_CLONE_ERROR * i
        d_theta_cam = delta_x[offset:offset + 3]
        d_p_cam = delta_x[offset + 3:offset + 6]
        q_cam_new = quat_multiply(axis_angle_to_quat(d_theta_cam), q_cam)
        clone_orientations.append(q_cam_new / np.linalg.norm(q_cam_new))
        clone_positions.append(p_cam + d_p_cam)

    return MSCKFState(state.p + d_p, state.v + d_v, q_new, state.b_g + d_bg, state.b_a + d_ba,
                       clone_positions, clone_orientations, P_new)
