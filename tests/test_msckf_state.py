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
from msckf_state import (
    MSCKFState,
    N_CLONE_ERROR,
    N_IMU_ERROR,
    augment,
    enforce_sliding_window,
    imu_error_jacobians,
    load_imu_noise_params,
    marginalize_clone,
    marginalize_clones,
    marginalize_oldest_clone,
    propagate,
)
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


def _distinguishable_state(n_clones):
    """A state whose clones/P entries are all individually identifiable, for testing marginalization bookkeeping."""
    clone_positions = [np.array([float(i), 0.0, 0.0]) for i in range(n_clones)]
    clone_orientations = [axis_angle_to_quat(np.array([0.0, 0.0, 0.1 * (i + 1)])) for i in range(n_clones)]
    clone_orientations = [q / np.linalg.norm(q) for q in clone_orientations]

    n = N_IMU_ERROR + N_CLONE_ERROR * n_clones
    P = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            P[i, j] = 1000 * i + j  # every entry uniquely identifies its (row, col)

    return MSCKFState(np.zeros(3), np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), np.zeros(3),
                       clone_positions, clone_orientations, P)


def test_marginalize_clone_drops_correct_clone_and_index_range():
    state = _distinguishable_state(4)  # clones 0,1,2,3 -> P column ranges [15,21),[21,27),[27,33),[33,39)
    updated = marginalize_clone(state, 1)

    assert updated.n_clones == 3
    assert [p[0] for p in updated.clone_positions] == [0.0, 2.0, 3.0]  # clone 1 (x=1.0) dropped
    assert updated.P.shape == (state.P.shape[0] - N_CLONE_ERROR, state.P.shape[0] - N_CLONE_ERROR)

    # hand-picked (not re-derived from the implementation's own loop) column ranges to keep
    expected_cols = list(range(15)) + list(range(15, 21)) + list(range(27, 33)) + list(range(33, 39))
    expected_P = state.P[np.ix_(expected_cols, expected_cols)]
    assert np.array_equal(updated.P, expected_P)


def test_marginalize_oldest_clone_matches_marginalize_clone_zero():
    state = _distinguishable_state(3)
    via_oldest = marginalize_oldest_clone(state)
    via_index = marginalize_clone(state, 0)

    assert np.array_equal(via_oldest.P, via_index.P)
    assert via_oldest.n_clones == 2
    assert via_oldest.clone_positions[0][0] == 1.0  # clone 0 (x=0.0) dropped, clone 1 (x=1.0) now first


def test_marginalize_clones_handles_noncontiguous_indices():
    state = _distinguishable_state(5)
    updated = marginalize_clones(state, [0, 2])
    assert updated.n_clones == 3
    assert [p[0] for p in updated.clone_positions] == [1.0, 3.0, 4.0]


def test_marginalize_clones_empty_list_is_a_noop():
    state = _distinguishable_state(3)
    updated = marginalize_clones(state, [])
    assert updated.n_clones == 3
    assert np.array_equal(updated.P, state.P)


def test_enforce_sliding_window_keeps_most_recent_clones():
    state = _distinguishable_state(10)
    updated = enforce_sliding_window(state, max_clones=4)
    assert updated.n_clones == 4
    assert [p[0] for p in updated.clone_positions] == [6.0, 7.0, 8.0, 9.0]  # oldest 6 dropped


def test_enforce_sliding_window_is_a_noop_when_already_under_the_limit():
    state = _distinguishable_state(3)
    updated = enforce_sliding_window(state, max_clones=10)
    assert updated.n_clones == 3
    assert np.array_equal(updated.P, state.P)


def test_sliding_window_stays_bounded_over_many_propagate_augment_steps():
    p0, v0, q0 = np.zeros(3), np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
    P0 = np.eye(N_IMU_ERROR) * 1e-4
    state = MSCKFState.initialize(p0, v0, q0, np.zeros(3), np.zeros(3), P0)
    noise_params = load_imu_noise_params()
    T_BS = load_T_BS(f"{MAV0_DIR}/cam0/sensor.yaml")
    timestamps, gyro, accel = _load_imu_measurements()

    max_clones = 8
    dt = 0.005
    for i in range(300):
        state = propagate(state, gyro[i], accel[i], dt, noise_params)
        if i % 10 == 0:
            state = augment(state, T_BS)
            state = enforce_sliding_window(state, max_clones)

        assert state.n_clones <= max_clones
        expected_dim = N_IMU_ERROR + N_CLONE_ERROR * state.n_clones
        assert state.P.shape == (expected_dim, expected_dim)
        assert np.allclose(state.P, state.P.T, atol=1e-9)
        eigenvalues = np.linalg.eigvalsh(state.P)
        assert eigenvalues.min() > -1e-9, f"step {i}: P not PSD, min eigenvalue {eigenvalues.min()}"

    assert state.n_clones == max_clones  # window should have filled up and stayed capped
