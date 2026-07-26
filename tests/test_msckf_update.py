import numpy as np
import pytest
from scipy.stats import chi2

from measurement_model import null_space_project, reprojection_residual_and_jacobians, stack_feature_observations
from msckf_state import MSCKFState, N_CLONE_ERROR, N_IMU_ERROR
from imu_propagation import GRAVITY_WORLD
from msckf_update import chi_square_threshold, compress_measurement, ekf_update, gravity_alignment_update, \
    mahalanobis_distance, passes_chi_square_gate, zero_angular_rate_update, zero_velocity_update
from quaternion_utils import axis_angle_to_quat, quat_error_vector, quat_multiply, quaternion_to_rotation_matrix, \
    skew_symmetric
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


def test_ekf_update_covariance_floor_prevents_collapse():
    # A deliberately extreme, highly-informative update (tiny sigma, H directly
    # observing every IMU dimension) that would otherwise crush P far below a
    # chosen floor -- checks the floor actually engages, not just that it's
    # trivially satisfied already.
    n = N_IMU_ERROR
    P0 = np.eye(n) * 1e-8
    sigma = 1e-6
    state = MSCKFState.initialize(np.zeros(3), np.zeros(3), np.array([1.0, 0, 0, 0]),
                                   np.zeros(3), np.zeros(3), P0)
    H_o = np.eye(n)
    r_o = np.zeros(n)
    min_variance = np.full(n, 1e-4)  # far larger than P0's already-tiny variance

    updated_unfloored = ekf_update(state, r_o, H_o, sigma)
    updated_floored = ekf_update(state, r_o, H_o, sigma, min_variance=min_variance)

    diag_unfloored = np.diag(updated_unfloored.P)
    diag_floored = np.diag(updated_floored.P)

    assert np.all(diag_unfloored < min_variance)  # sanity check: without the floor, it really does collapse
    assert np.all(diag_floored >= min_variance - 1e-15)
    assert np.allclose(diag_floored, min_variance, atol=1e-12)  # floor is tight: deficit exactly made up

    eigenvalues = np.linalg.eigvalsh(updated_floored.P)
    assert eigenvalues.min() > -1e-12
    assert np.allclose(updated_floored.P, updated_floored.P.T, atol=1e-12)


def test_ekf_update_covariance_floor_is_a_noop_when_already_above_floor():
    rng = np.random.default_rng(12)
    state, true_positions, true_orientations, _ = _synthetic_state_and_truth(rng)
    X_true = np.array([0.5, -0.3, 5.0])
    sigma = 0.002
    camera_poses = list(zip(state.clone_orientations, state.clone_positions))
    clone_indices = list(range(state.n_clones))

    bearings = [_true_bearing(X_true, q, p) + rng.normal(size=2) * sigma
                for q, p in zip(true_orientations, true_positions)]
    X_est = triangulate_feature(camera_poses, bearings)
    r, H_x, H_f = stack_feature_observations(X_est, camera_poses, bearings, clone_indices, n_clones=state.n_clones)
    r_o, H_o = null_space_project(r, H_x, H_f)

    tiny_floor = np.full(N_IMU_ERROR, 1e-20)  # far below anything this update could produce
    updated_unfloored = ekf_update(state, r_o, H_o, sigma)
    updated_floored = ekf_update(state, r_o, H_o, sigma, min_variance=tiny_floor)

    assert np.allclose(updated_unfloored.P, updated_floored.P, atol=1e-15)


def test_zero_velocity_update_pulls_velocity_toward_zero_and_shrinks_its_covariance():
    P0 = np.eye(N_IMU_ERROR) * 1e-2
    state = MSCKFState.initialize(np.zeros(3), np.array([0.4, -0.2, 0.1]), np.array([1.0, 0, 0, 0]),
                                   np.zeros(3), np.zeros(3), P0)
    v_before_norm = np.linalg.norm(state.v)

    updated = zero_velocity_update(state, zupt_noise_std=1e-3)

    assert np.linalg.norm(updated.v) < v_before_norm  # pulled toward the asserted v=0
    assert np.linalg.norm(updated.v) < 0.05  # a tight ZUPT noise should correct this almost fully

    var_v_before = np.diag(state.P)[6:9]
    var_v_after = np.diag(updated.P)[6:9]
    assert np.all(var_v_after < var_v_before)  # a confident, informative measurement should shrink P

    eigenvalues = np.linalg.eigvalsh(updated.P)
    assert eigenvalues.min() > -1e-9
    assert np.allclose(updated.P, updated.P.T, atol=1e-9)

    # only velocity should move to first order -- other IMU blocks are untouched by this measurement
    assert np.allclose(updated.p, state.p, atol=1e-9)
    assert np.allclose(updated.b_g, state.b_g, atol=1e-9)
    assert np.allclose(updated.b_a, state.b_a, atol=1e-9)


def test_zero_velocity_update_does_not_leak_into_bias_through_cross_correlation():
    # a deliberately correlated P (unlike the diagonal one above) -- checks that the
    # restriction to the velocity-only mean update is a real, active choice and not just
    # trivially true because P had no cross-terms to leak through in the first place.
    # This directly guards against a real incident found on this dataset: a b_g-v
    # correlation from ordinary earlier operation dragged gyro bias by ~0.06 rad/s over a
    # few seconds of ZUPT, corrupting bias despite the "measurement" only ever asserting
    # something about velocity.
    n = N_IMU_ERROR
    P = np.eye(n) * 1e-4
    P[3:6, 6:9] = np.eye(3) * 5e-5  # strong b_g <-> v correlation
    P[6:9, 3:6] = np.eye(3) * 5e-5
    state = MSCKFState.initialize(np.zeros(3), np.array([0.3, -0.1, 0.05]), np.array([1.0, 0, 0, 0]),
                                   np.array([0.02, -0.01, 0.05]), np.zeros(3), P)

    updated = zero_velocity_update(state, zupt_noise_std=1e-3)

    assert np.allclose(updated.b_g, state.b_g, atol=1e-9)  # bias mean must not move, despite the correlation
    assert np.linalg.norm(updated.v) < np.linalg.norm(state.v)  # velocity should still be corrected


def test_zero_angular_rate_update_pulls_bias_toward_raw_gyro_reading():
    P0 = np.eye(N_IMU_ERROR) * 1e-2
    state = MSCKFState.initialize(np.zeros(3), np.zeros(3), np.array([1.0, 0, 0, 0]),
                                   np.array([0.02, -0.01, 0.03]), np.zeros(3), P0)
    true_reading = np.array([0.001, 0.0005, -0.0008])  # what the raw gyro reads while at rest

    updated = zero_angular_rate_update(state, true_reading, zaru_noise_std=1e-3)

    assert np.linalg.norm(updated.b_g - true_reading) < np.linalg.norm(state.b_g - true_reading)
    assert np.all(np.diag(updated.P)[3:6] < np.diag(state.P)[3:6])  # a confident measurement shrinks P

    eigenvalues = np.linalg.eigvalsh(updated.P)
    assert eigenvalues.min() > -1e-9
    assert np.allclose(updated.P, updated.P.T, atol=1e-9)

    # P0 is diagonal here, so no other block should move to first order
    assert np.allclose(updated.v, state.v, atol=1e-9)
    assert np.allclose(updated.p, state.p, atol=1e-9)
    assert np.allclose(updated.b_a, state.b_a, atol=1e-9)


def test_gravity_alignment_update_jacobian_matches_finite_difference():
    # re-derived (not transcribed) given this project's history of sign bugs in exactly this
    # kind of Jacobian -- trust this numerical check over the analytic derivation in the
    # docstring, per the project's established practice.
    rng = np.random.default_rng(20)
    q_hat = axis_angle_to_quat(rng.normal(size=3) * 0.5)
    q_hat = q_hat / np.linalg.norm(q_hat)
    b_a = rng.normal(size=3) * 0.05

    def h(q, b_a_val):
        R = quaternion_to_rotation_matrix(q)
        return R.T @ (-GRAVITY_WORLD) + b_a_val

    h0 = h(q_hat, b_a)
    eps = 1e-6

    dh_dtheta_fd = np.zeros((3, 3))
    for i in range(3):
        dtheta = np.zeros(3)
        dtheta[i] = eps
        q_pert = quat_multiply(axis_angle_to_quat(dtheta), q_hat)
        q_pert = q_pert / np.linalg.norm(q_pert)
        dh_dtheta_fd[:, i] = (h(q_pert, b_a) - h0) / eps

    dh_dba_fd = np.zeros((3, 3))
    for i in range(3):
        db = np.zeros(3)
        db[i] = eps
        dh_dba_fd[:, i] = (h(q_hat, b_a + db) - h0) / eps

    R_hat = quaternion_to_rotation_matrix(q_hat)
    dh_dtheta_analytic = -R_hat.T @ skew_symmetric(GRAVITY_WORLD)

    assert np.allclose(dh_dtheta_analytic, dh_dtheta_fd, atol=1e-4)
    assert np.allclose(np.eye(3), dh_dba_fd, atol=1e-6)


def test_gravity_alignment_update_corrects_accel_bias_and_tilt():
    q_true = axis_angle_to_quat(np.array([0.02, -0.015, 0.0]))  # small roll/pitch error, no yaw
    q_true = q_true / np.linalg.norm(q_true)
    b_a_true = np.array([0.03, -0.02, 0.01])

    P0 = np.eye(N_IMU_ERROR) * 1e-2
    state = MSCKFState.initialize(np.zeros(3), np.zeros(3), np.array([1.0, 0, 0, 0]),
                                   np.zeros(3), np.zeros(3), P0)  # wrong (identity) orientation, zero bias

    R_true = quaternion_to_rotation_matrix(q_true)
    accel_measured = R_true.T @ (-GRAVITY_WORLD) + b_a_true

    updated = gravity_alignment_update(state, accel_measured, tilt_noise_std=1e-2)

    assert np.linalg.norm(updated.b_a - b_a_true) < np.linalg.norm(state.b_a - b_a_true)

    theta_err_before = np.linalg.norm(quat_error_vector(q_true, state.q))
    theta_err_after = np.linalg.norm(quat_error_vector(q_true, updated.q))
    assert theta_err_after < theta_err_before

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
