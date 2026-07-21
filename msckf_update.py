"""EKF update step for the MSCKF: chi-square outlier gating on the null-space
projected residual, optional QR measurement compression, and the Joseph-form
covariance/state update.

Error-state convention matches msckf_state.py/measurement_model.py exactly:
delta_theta is a GLOBAL perturbation with R_true = (I + [delta_theta]x) @ R_hat.
"""
import numpy as np
from scipy.stats import chi2

from msckf_state import MSCKFState, N_CLONE_ERROR, N_IMU_ERROR
from quaternion_utils import axis_angle_to_quat, quat_multiply

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


def ekf_update(state, r_o, H_o, observation_noise_std):
    """Apply one EKF measurement update (Joseph form). Returns a new MSCKFState."""
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

    return _apply_error_state(state, delta_x, P_new)


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
