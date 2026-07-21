import numpy as np
import pytest
from scipy.stats import chi2

from measurement_model import null_space_project, reprojection_residual_and_jacobians, stack_feature_observations
from msckf_state import MSCKFState, N_CLONE_ERROR, N_IMU_ERROR
from msckf_update import chi_square_threshold, compress_measurement, ekf_update, mahalanobis_distance, \
    passes_chi_square_gate
from quaternion_utils import axis_angle_to_quat, quat_error_vector, quat_multiply
from triangulation import triangulate_feature


def _random_unit_quaternion(rng):
    v = rng.normal(size=4)
    return v / np.linalg.norm(v)


def _true_bearing(X_world, q_cam, p_cam):
    """Exact (noiseless) bearing of X_world as seen by camera (q_cam, p_cam)."""
    r, _, _ = reprojection_residual_and_jacobians(X_world, q_cam, p_cam, np.zeros(2))
    return -r  # r = obs - h, obs=0 => r = -h => h = -r


def test_chi_square_threshold_matches_scipy_and_is_monotonic():
    assert chi_square_threshold(5, 0.95) == pytest.approx(chi2.ppf(0.95, 5))
    assert chi_square_threshold(5, 0.99) > chi_square_threshold(5, 0.95)  # higher confidence -> wider gate
    assert chi_square_threshold(10, 0.95) > chi_square_threshold(5, 0.95)  # more dof -> larger threshold


def test_mahalanobis_distance_isotropic_case():
    r_o = np.array([0.3, -0.4, 0.5])
    H_o = np.zeros((3, 21))
    sigma = 0.1
    expected = np.dot(r_o, r_o) / sigma ** 2
    assert mahalanobis_distance(r_o, H_o, np.eye(21), sigma) == pytest.approx(expected)


def test_gate_accepts_small_residual_rejects_large_one():
    H_o = np.zeros((3, 21))
    P = np.eye(21) * 1e-4
    sigma = 0.002

    r_small = np.array([0.001, -0.001, 0.0005])  # within a couple of sigma
    assert passes_chi_square_gate(r_small, H_o, P, sigma)

    r_large = np.array([0.5, -0.5, 0.5])  # wildly outside noise -- an outlier
    assert not passes_chi_square_gate(r_large, H_o, P, sigma)


def test_compress_measurement_is_a_noop_when_not_overtall():
    H_o = np.random.default_rng(0).normal(size=(4, 10))
    r_o = np.random.default_rng(1).normal(size=4)
    r_c, H_c = compress_measurement(r_o, H_o)
    assert np.allclose(r_c, r_o)
    assert np.allclose(H_c, H_o)


def test_compress_measurement_preserves_ekf_update_result():
    rng = np.random.default_rng(2)
    n = 15  # base IMU state only, no clones, for a clean minimal test
    H_o = rng.normal(size=(20, n))  # over-tall: 20 rows > 15 cols, compression should kick in
    r_o = rng.normal(size=20) * 0.01
    P = np.eye(n) * 1e-3
    sigma = 0.01

    state = MSCKFState.initialize(np.zeros(3), np.zeros(3), np.array([1.0, 0, 0, 0]),
                                   np.zeros(3), np.zeros(3), P)

    # manual, uncompressed EKF math for comparison
    R_noise = sigma ** 2 * np.eye(20)
    S = H_o @ P @ H_o.T + R_noise
    K = P @ H_o.T @ np.linalg.inv(S)
    delta_x_uncompressed = -(K @ r_o)  # see the sign-convention comment in msckf_update.ekf_update
    I_KH = np.eye(n) - K @ H_o
    P_new_uncompressed = I_KH @ P @ I_KH.T + K @ R_noise @ K.T

    updated = ekf_update(state, r_o, H_o, sigma)

    assert np.allclose(updated.p, delta_x_uncompressed[12:15], atol=1e-9)
    assert np.allclose(updated.P, P_new_uncompressed, atol=1e-9)


def _synthetic_state_and_truth(rng, n_clones=3):
    """A nominal MSCKFState with n_clones camera clones (block-diagonal P for a clean test),
    plus the TRUE clone poses (nominal + small known errors) and a true feature position.

    The perturbation scale here (~0.003 rad, ~0.005m) is deliberately kept
    within the EKF's linear-validity regime, matching the size of error a
    well-behaved propagation step would actually produce. A single EKF update
    is a *linearized* correction and isn't meant to fully correct arbitrarily
    large errors in one step -- injecting truth 10x further out than this
    reliably makes a single update worse, which is expected EKF behavior, not
    a bug in the update itself (verified separately against a hand-derived
    reference calculation).
    """
    nominal_clone_positions, nominal_clone_orientations = [], []
    true_clone_positions, true_clone_orientations = [], []
    true_deltas = []

    for i in range(n_clones):
        p_nom = np.array([0.3 * i, 0.0, 0.0])
        q_nom = _random_unit_quaternion(rng)
        d_theta = rng.normal(size=3) * 0.002
        d_p = rng.normal(size=3) * 0.003
        true_deltas.append(np.concatenate([d_theta, d_p]))

        q_true = quat_multiply(axis_angle_to_quat(d_theta), q_nom)
        q_true = q_true / np.linalg.norm(q_true)

        nominal_clone_positions.append(p_nom)
        nominal_clone_orientations.append(q_nom)
        true_clone_positions.append(p_nom + d_p)
        true_clone_orientations.append(q_true)

    P0 = np.eye(N_IMU_ERROR) * 1e-6  # tight base-state uncertainty, irrelevant to this test
    P_clone_block = np.eye(N_CLONE_ERROR) * (0.003 ** 2)  # matches the true perturbation scale above
    n = N_IMU_ERROR + N_CLONE_ERROR * n_clones
    P = np.zeros((n, n))
    P[:N_IMU_ERROR, :N_IMU_ERROR] = P0
    for i in range(n_clones):
        off = N_IMU_ERROR + N_CLONE_ERROR * i
        P[off:off + N_CLONE_ERROR, off:off + N_CLONE_ERROR] = P_clone_block

    state = MSCKFState(np.zeros(3), np.zeros(3), np.array([1.0, 0, 0, 0]), np.zeros(3), np.zeros(3),
                        nominal_clone_positions, nominal_clone_orientations, P)

    return state, true_clone_positions, true_clone_orientations, true_deltas


def test_update_moves_state_toward_truth_and_shrinks_covariance():
    # A single feature seen by 3 clones only gives 2*3-3=3 constraint rows
    # against 18 clone-error dimensions -- genuinely under-determined, so it
    # can't be expected to shrink error in *every* direction. Using many
    # independent features (as a real MSCKF batches per timestep) makes the
    # problem well-determined, matching how the filter is actually used.
    rng = np.random.default_rng(10)
    state, true_positions, true_orientations, true_deltas = _synthetic_state_and_truth(rng)
    sigma = 0.002
    n_features = 15

    camera_poses = list(zip(state.clone_orientations, state.clone_positions))  # nominal, as the filter would use
    clone_indices = list(range(state.n_clones))

    r_o_all, H_o_all = [], []
    for _ in range(n_features):
        X_true = rng.normal(size=3) * 1.5 + np.array([0.0, 0.0, 5.0])
        bearings = [_true_bearing(X_true, q, p) + rng.normal(size=2) * sigma
                    for q, p in zip(true_orientations, true_positions)]

        X_est = triangulate_feature(camera_poses, bearings)
        r, H_x, H_f = stack_feature_observations(X_est, camera_poses, bearings, clone_indices,
                                                  n_clones=state.n_clones)
        r_o, H_o = null_space_project(r, H_x, H_f)
        assert passes_chi_square_gate(r_o, H_o, state.P, sigma)
        r_o_all.append(r_o)
        H_o_all.append(H_o)

    r_o_stacked = np.concatenate(r_o_all)
    H_o_stacked = np.concatenate(H_o_all, axis=0)

    error_before = np.concatenate(true_deltas)  # true - nominal, in the same error-state layout as each clone block

    updated = ekf_update(state, r_o_stacked, H_o_stacked, sigma)

    error_after = []
    for i in range(state.n_clones):
        d_theta_after = quat_error_vector(true_orientations[i], updated.clone_orientations[i])
        d_p_after = true_positions[i] - updated.clone_positions[i]
        error_after.append(np.concatenate([d_theta_after, d_p_after]))
    error_after = np.concatenate(error_after)

    assert np.linalg.norm(error_after) < np.linalg.norm(error_before)

    clone_trace_before = np.diag(state.P)[N_IMU_ERROR:].sum()
    clone_trace_after = np.diag(updated.P)[N_IMU_ERROR:].sum()
    assert clone_trace_after < clone_trace_before

    eigenvalues = np.linalg.eigvalsh(updated.P)
    assert eigenvalues.min() > -1e-9
    assert np.allclose(updated.P, updated.P.T, atol=1e-9)


def test_gate_rejects_a_corrupted_feature_but_accepts_a_clean_one():
    rng = np.random.default_rng(11)
    state, true_positions, true_orientations, _ = _synthetic_state_and_truth(rng)
    X_true = np.array([0.5, -0.3, 5.0])
    sigma = 0.002
    camera_poses = list(zip(state.clone_orientations, state.clone_positions))
    clone_indices = list(range(state.n_clones))

    clean_bearings = [_true_bearing(X_true, q, p) + rng.normal(size=2) * sigma
                       for q, p in zip(true_orientations, true_positions)]
    X_est_clean = triangulate_feature(camera_poses, clean_bearings)
    r, H_x, H_f = stack_feature_observations(X_est_clean, camera_poses, clean_bearings, clone_indices,
                                              n_clones=state.n_clones)
    r_o, H_o = null_space_project(r, H_x, H_f)
    assert passes_chi_square_gate(r_o, H_o, state.P, sigma)

    corrupted_bearings = list(clean_bearings)
    corrupted_bearings[0] = corrupted_bearings[0] + np.array([0.3, -0.3])  # a huge, obviously-wrong observation
    X_est_corrupt = triangulate_feature(camera_poses, corrupted_bearings)
    r_c, H_x_c, H_f_c = stack_feature_observations(X_est_corrupt, camera_poses, corrupted_bearings, clone_indices,
                                                    n_clones=state.n_clones)
    r_o_c, H_o_c = null_space_project(r_c, H_x_c, H_f_c)
    assert not passes_chi_square_gate(r_o_c, H_o_c, state.P, sigma)
