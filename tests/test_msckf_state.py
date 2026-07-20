"""Finite-difference validation of msckf_state.py's error-state Jacobians.

These are the single most bug-prone piece of an MSCKF: a sign error in F or
the augmentation Jacobian J will not crash anything, it will just silently
make the filter diverge. Each test perturbs one component of the true
error-state, propagates the *nonlinear* true and nominal trajectories
independently, and checks the actual resulting error against what the
analytic Jacobian predicts to first order.
"""
import numpy as np
import pytest

from frames import body_to_sensor_position, body_to_sensor_quaternion, load_T_BS
from imu_propagation import MAV0_DIR, _load_imu_measurements, propagate_step
from msckf_state import MSCKFState, N_IMU_ERROR, augment, imu_error_jacobians, load_imu_noise_params, propagate
from quaternion_utils import axis_angle_to_quat, quat_error_vector, quat_multiply

EPS = 1e-6


def _random_unit_quaternion(rng):
    v = rng.normal(size=4)
    return v / np.linalg.norm(v)


def _perturb_state(p, v, q, b_g, b_a, dim, eps):
    """Apply +eps to error-state dimension `dim` (order: theta, b_g, v, b_a, p)."""
    delta = np.zeros(N_IMU_ERROR)
    delta[dim] = eps
    d_theta, d_bg, d_v, d_ba, d_p = delta[0:3], delta[3:6], delta[6:9], delta[9:12], delta[12:15]

    q_pert = quat_multiply(axis_angle_to_quat(d_theta), q)
    q_pert = q_pert / np.linalg.norm(q_pert)
    return p + d_p, v + d_v, q_pert, b_g + d_bg, b_a + d_ba


def _state_error(p_true, v_true, q_true, b_g_true, b_a_true, p_hat, v_hat, q_hat, b_g_hat, b_a_hat):
    return np.concatenate([
        quat_error_vector(q_true, q_hat),
        b_g_true - b_g_hat,
        v_true - v_hat,
        b_a_true - b_a_hat,
        p_true - p_hat,
    ])


@pytest.mark.parametrize("dim", range(N_IMU_ERROR))
def test_imu_error_jacobian_matches_nonlinear_propagation(dim):
    rng = np.random.default_rng(1234 + dim)

    p_hat = rng.normal(size=3)
    v_hat = rng.normal(size=3) * 0.5
    q_hat = _random_unit_quaternion(rng)
    b_g_hat = rng.normal(size=3) * 1e-3
    b_a_hat = rng.normal(size=3) * 1e-2

    gyro = rng.normal(size=3) * 0.5
    accel = rng.normal(size=3) * 2.0 + np.array([0.0, 0.0, 9.81])
    dt = 0.005

    F, _ = imu_error_jacobians(q_hat, gyro - b_g_hat, accel - b_a_hat)
    Phi = np.eye(N_IMU_ERROR) + F * dt

    p_true0, v_true0, q_true0, b_g_true0, b_a_true0 = _perturb_state(
        p_hat, v_hat, q_hat, b_g_hat, b_a_hat, dim, EPS)
    delta_x0 = np.zeros(N_IMU_ERROR)
    delta_x0[dim] = EPS

    p_hat1, v_hat1, q_hat1 = propagate_step(p_hat, v_hat, q_hat, gyro, accel, dt,
                                             bias_gyro=b_g_hat, bias_accel=b_a_hat)
    p_true1, v_true1, q_true1 = propagate_step(p_true0, v_true0, q_true0, gyro, accel, dt,
                                                bias_gyro=b_g_true0, bias_accel=b_a_true0)
    # biases have no dynamics of their own here (pure random walk / unmodeled)
    b_g_true1, b_a_true1 = b_g_true0, b_a_true0

    actual = _state_error(p_true1, v_true1, q_true1, b_g_true1, b_a_true1,
                           p_hat1, v_hat1, q_hat1, b_g_hat, b_a_hat)
    predicted = Phi @ delta_x0

    # first-order linearization: error should be O(EPS^2); a loose relative
    # tolerance on top of EPS itself catches sign/wiring bugs without being
    # sensitive to the quadratic remainder
    assert np.linalg.norm(actual - predicted) < 50 * EPS ** 1.5, (
        f"dim={dim}: actual={actual}, predicted={predicted}")


@pytest.mark.parametrize("dim", range(N_IMU_ERROR))
def test_augmentation_jacobian_matches_nonlinear_clone(dim):
    rng = np.random.default_rng(5678 + dim)

    p_hat = rng.normal(size=3)
    v_hat = rng.normal(size=3)
    q_hat = _random_unit_quaternion(rng)
    b_g_hat = rng.normal(size=3) * 1e-3
    b_a_hat = rng.normal(size=3) * 1e-2

    T_BS = load_T_BS(f"{MAV0_DIR}/cam0/sensor.yaml")

    # augment() only ever exposes J indirectly (through P_cross = J @ P), so
    # recover J itself by augmenting a state whose P0 is the identity: then
    # P_cross == J directly
    state = MSCKFState.initialize(p_hat, v_hat, q_hat, b_g_hat, b_a_hat, np.eye(N_IMU_ERROR))
    augmented = augment(state, T_BS)
    J = augmented.P[N_IMU_ERROR:, :N_IMU_ERROR]

    p_true, v_true, q_true, b_g_true, b_a_true = _perturb_state(
        p_hat, v_hat, q_hat, b_g_hat, b_a_hat, dim, EPS)
    delta_x = np.zeros(N_IMU_ERROR)
    delta_x[dim] = EPS

    q_cam_hat = body_to_sensor_quaternion(q_hat, T_BS)
    p_cam_hat = body_to_sensor_position(p_hat, q_hat, T_BS)
    q_cam_true = body_to_sensor_quaternion(q_true, T_BS)
    p_cam_true = body_to_sensor_position(p_true, q_true, T_BS)

    actual = np.concatenate([quat_error_vector(q_cam_true, q_cam_hat), p_cam_true - p_cam_hat])
    predicted = J @ delta_x

    assert np.linalg.norm(actual - predicted) < 50 * EPS ** 1.5, (
        f"dim={dim}: actual={actual}, predicted={predicted}")


def test_covariance_stays_symmetric_positive_semidefinite():
    p0 = np.zeros(3)
    v0 = np.zeros(3)
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    b_g0 = np.zeros(3)
    b_a0 = np.zeros(3)
    P0 = np.eye(N_IMU_ERROR) * 1e-4

    state = MSCKFState.initialize(p0, v0, q0, b_g0, b_a0, P0)
    noise_params = load_imu_noise_params()
    T_BS = load_T_BS(f"{MAV0_DIR}/cam0/sensor.yaml")

    timestamps, gyro, accel = _load_imu_measurements()
    dt = 0.005
    for i in range(100):
        state = propagate(state, gyro[i], accel[i], dt, noise_params)
        if i % 20 == 0:
            state = augment(state, T_BS)

        assert np.allclose(state.P, state.P.T, atol=1e-9)
        eigenvalues = np.linalg.eigvalsh(state.P)
        assert eigenvalues.min() > -1e-9, f"step {i}: P not PSD, min eigenvalue {eigenvalues.min()}"

    assert state.n_clones == 5
