"""Error-state (indirect) EKF scaffolding for the MSCKF: state representation,
IMU error-state propagation, and camera-clone state augmentation.

Error-state convention (kept consistent throughout this module, and matching
quaternion_utils.quat_error_vector/axis_angle_to_quat exactly): orientation
error delta_theta is a GLOBAL/world-frame perturbation, defined by
    R_true = (I + [delta_theta]x) @ R_hat
i.e. q_true = axis_angle_to_quat(delta_theta) (x) q_hat for small delta_theta.
All other errors (velocity, position, biases) are plain differences,
x_true = x_hat + delta_x. (Note: this is the sign-opposite of the original
MSCKF paper's stated convention; the algebra here was re-derived and verified
against it directly via the finite-difference tests in tests/test_msckf_state.py
rather than transcribed, since the sign is easy to get backwards either way.)

Per-IMU-state error vector layout (15,): [theta(3), b_g(3), v(3), b_a(3), p(3)]
Per-camera-clone error vector layout (6,): [theta(3), p(3)]
"""
import os

import numpy as np
import yaml

from frames import body_to_sensor_position, body_to_sensor_quaternion
from imu_propagation import propagate_step
from quaternion_utils import quaternion_to_rotation_matrix, skew_symmetric

MAV0_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "machine_hall", "MH_01_easy", "MH_01_easy", "mav0")

N_IMU_ERROR = 15
N_CLONE_ERROR = 6


def load_imu_noise_params(mav0_dir=MAV0_DIR):
    """Continuous-time IMU noise densities from imu0/sensor.yaml."""
    path = os.path.join(mav0_dir, "imu0", "sensor.yaml")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return dict(
        gyro_noise=cfg["gyroscope_noise_density"],
        gyro_bias_walk=cfg["gyroscope_random_walk"],
        accel_noise=cfg["accelerometer_noise_density"],
        accel_bias_walk=cfg["accelerometer_random_walk"],
    )


class MSCKFState:
    """Nominal IMU state + a list of camera-pose clones, plus their joint error covariance P."""

    def __init__(self, p, v, q, b_g, b_a, clone_positions, clone_orientations, P):
        self.p = np.asarray(p, dtype=float)
        self.v = np.asarray(v, dtype=float)
        self.q = np.asarray(q, dtype=float)
        self.b_g = np.asarray(b_g, dtype=float)
        self.b_a = np.asarray(b_a, dtype=float)
        self.clone_positions = list(clone_positions)
        self.clone_orientations = list(clone_orientations)
        self.P = np.asarray(P, dtype=float)

        expected_dim = N_IMU_ERROR + N_CLONE_ERROR * len(self.clone_positions)
        assert self.P.shape == (expected_dim, expected_dim), \
            f"P shape {self.P.shape} inconsistent with {len(self.clone_positions)} clones"

    @property
    def n_clones(self):
        return len(self.clone_positions)

    @classmethod
    def initialize(cls, p, v, q, b_g, b_a, P0):
        """Fresh filter state with no camera clones yet. P0 is the initial 15x15 IMU covariance."""
        return cls(p, v, q, b_g, b_a, [], [], P0)


def imu_error_jacobians(q_hat, omega_hat, a_hat):
    """Continuous-time error-state Jacobians F (15x15) and G (15x12) at the current linearization point.

    omega_hat = gyro measurement - b_g_hat (bias-corrected angular velocity)
    a_hat = accel measurement - b_a_hat (bias-corrected specific force)
    G's noise columns are ordered [n_g(3), n_wg(3), n_a(3), n_wa(3)].
    """
    R_hat = quaternion_to_rotation_matrix(q_hat)

    F = np.zeros((N_IMU_ERROR, N_IMU_ERROR))
    F[0:3, 0:3] = -skew_symmetric(omega_hat)
    F[0:3, 3:6] = np.eye(3)
    F[6:9, 0:3] = -skew_symmetric(R_hat @ a_hat)
    F[6:9, 9:12] = -R_hat
    F[12:15, 6:9] = np.eye(3)

    G = np.zeros((N_IMU_ERROR, 12))
    G[0:3, 0:3] = np.eye(3)
    G[3:6, 3:6] = np.eye(3)
    G[6:9, 6:9] = -R_hat
    G[9:12, 9:12] = np.eye(3)

    return F, G


def discretize(F, G, Qc, dt):
    """First-order discretization: Phi ~= I + F dt, Qd ~= Phi (G Qc G^T) Phi^T dt."""
    Phi = np.eye(N_IMU_ERROR) + F * dt
    Qd = Phi @ (G @ Qc @ G.T) @ Phi.T * dt
    return Phi, Qd


def propagate(state, gyro, accel, dt, noise_params):
    """Propagate the nominal state and the full (IMU + clones) covariance by one IMU step.

    Zero bias correction: biases are carried through unchanged (their own
    dynamics are pure random walk, handled by Qd) since nothing estimates them yet.
    """
    omega_hat = gyro - state.b_g
    a_hat = accel - state.b_a

    F, G = imu_error_jacobians(state.q, omega_hat, a_hat)
    Qc = np.diag(
        [noise_params["gyro_noise"] ** 2] * 3
        + [noise_params["gyro_bias_walk"] ** 2] * 3
        + [noise_params["accel_noise"] ** 2] * 3
        + [noise_params["accel_bias_walk"] ** 2] * 3
    )
    Phi, Qd = discretize(F, G, Qc, dt)

    # Phi_full = [[Phi, 0], [0, I]] (clones are frozen copies, so they don't evolve
    # under IMU propagation) -- applying it densely as an n x n matmul, n up to 135+,
    # is O(n^3) for a result that's mostly a no-op. Block form below is exact (expand
    # Phi_full @ P @ Phi_full.T + Qd_full and the identity blocks cancel out) and
    # O(N_IMU_ERROR * n) instead: only the IMU/clone cross-covariance actually needs
    # touching, and the clone-clone block passes through completely unchanged.
    P = state.P
    P_II = P[:N_IMU_ERROR, :N_IMU_ERROR]
    P_IC = P[:N_IMU_ERROR, N_IMU_ERROR:]

    new_P = P.copy()
    new_P[:N_IMU_ERROR, :N_IMU_ERROR] = Phi @ P_II @ Phi.T + Qd
    new_P[:N_IMU_ERROR, N_IMU_ERROR:] = Phi @ P_IC
    new_P[N_IMU_ERROR:, :N_IMU_ERROR] = new_P[:N_IMU_ERROR, N_IMU_ERROR:].T
    new_P = 0.5 * (new_P + new_P.T)  # symmetrize away floating-point drift

    p_new, v_new, q_new = propagate_step(state.p, state.v, state.q, gyro, accel, dt,
                                          bias_gyro=state.b_g, bias_accel=state.b_a)

    return MSCKFState(p_new, v_new, q_new, state.b_g.copy(), state.b_a.copy(),
                       state.clone_positions, state.clone_orientations, new_P)


def augment(state, T_BS):
    """Clone the current IMU pose into a new camera-pose clone (nominal state + covariance).

    The camera-IMU extrinsic T_BS is treated as exactly known (no calibration
    error / no extra noise added), so the new clone's error is a deterministic
    linear function of the current state error: delta_theta_cam = delta_theta_body,
    delta_p_cam = -[R_hat @ t_BS]x @ delta_theta_body + delta_p_body (rigid-body
    lever-arm coupling).
    """
    q_cam = body_to_sensor_quaternion(state.q, T_BS)
    p_cam = body_to_sensor_position(state.p, state.q, T_BS)

    R_hat = quaternion_to_rotation_matrix(state.q)
    lever_arm_world = R_hat @ T_BS[:3, 3]

    n = state.P.shape[0]
    J = np.zeros((N_CLONE_ERROR, n))
    J[0:3, 0:3] = np.eye(3)                           # d(theta_cam) / d(theta_body)
    J[3:6, 0:3] = -skew_symmetric(lever_arm_world)    # d(p_cam) / d(theta_body)
    J[3:6, 12:15] = np.eye(3)                         # d(p_cam) / d(p_body)

    cross = J @ state.P
    corner = cross @ J.T

    new_n = n + N_CLONE_ERROR
    P_aug = np.zeros((new_n, new_n))
    P_aug[:n, :n] = state.P
    P_aug[n:, :n] = cross
    P_aug[:n, n:] = cross.T
    P_aug[n:, n:] = corner
    P_aug = 0.5 * (P_aug + P_aug.T)

    return MSCKFState(state.p, state.v, state.q, state.b_g, state.b_a,
                       state.clone_positions + [p_cam],
                       state.clone_orientations + [q_cam],
                       P_aug)


def marginalize_clones(state, indices_to_remove):
    """Drop the given camera clones (0-based indices into clone_positions/orientations) from the state.

    Plain deletion: the corresponding rows/columns are dropped from P
    outright, with no Schmidt-filter-style prior-fixing step. That's valid
    precisely because a clone is only ever marginalized after every feature
    it observed has already triggered its EKF update (stage 6) -- by then the
    clone carries no information that hasn't already been folded into the
    surviving state, so simply deleting it loses nothing.
    """
    indices_to_remove = sorted(set(indices_to_remove))
    if not indices_to_remove:
        return state

    keep_clone_idx = [i for i in range(state.n_clones) if i not in indices_to_remove]

    keep_cols = list(range(N_IMU_ERROR))
    for i in keep_clone_idx:
        off = N_IMU_ERROR + N_CLONE_ERROR * i
        keep_cols.extend(range(off, off + N_CLONE_ERROR))

    P_new = state.P[np.ix_(keep_cols, keep_cols)]

    return MSCKFState(state.p, state.v, state.q, state.b_g, state.b_a,
                       [state.clone_positions[i] for i in keep_clone_idx],
                       [state.clone_orientations[i] for i in keep_clone_idx],
                       P_new)


def marginalize_clone(state, index):
    """Drop a single camera clone by index."""
    return marginalize_clones(state, [index])


def marginalize_oldest_clone(state):
    """Drop the oldest (first-added, index 0) camera clone -- clones are always appended, so index 0 is oldest."""
    return marginalize_clone(state, 0)


def enforce_sliding_window(state, max_clones):
    """Marginalize the oldest clone(s), if needed, so at most max_clones remain."""
    n_to_remove = state.n_clones - max_clones
    if n_to_remove <= 0:
        return state
    return marginalize_clones(state, range(n_to_remove))
