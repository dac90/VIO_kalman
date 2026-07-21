"""Finite-difference validation of measurement_model.py.

Same rationale as tests/test_msckf_state.py: a sign error in H_clone/H_f or
in the null-space projection wouldn't crash anything, it would just silently
feed the EKF update a wrong constraint. Each Jacobian test perturbs one
component of the true error-state/feature-position, recomputes the
*nonlinear* residual independently, and checks it against what the analytic
Jacobian predicts to first order.
"""
import numpy as np
import pytest

import ground_truth as gt
from feature_tracker import FeatureTracker, iter_cam0_frames
from frames import body_to_sensor_position, load_T_BS
from measurement_model import null_space_project, reprojection_residual_and_jacobians, stack_feature_observations
from msckf_state import N_CLONE_ERROR
from quaternion_utils import axis_angle_to_quat, quat_multiply
from triangulation import load_cam0_intrinsics, triangulate_feature, undistort_normalized

EPS = 1e-6


def _random_unit_quaternion(rng):
    v = rng.normal(size=4)
    return v / np.linalg.norm(v)


def _perturb(X_world, q_cam, p_cam, dim, eps):
    """Apply +eps to combined error dim (order: theta_cam(3), p_cam(3), X_world(3))."""
    delta = np.zeros(9)
    delta[dim] = eps
    d_theta, d_p_cam, d_X = delta[0:3], delta[3:6], delta[6:9]

    q_pert = quat_multiply(axis_angle_to_quat(d_theta), q_cam)
    q_pert = q_pert / np.linalg.norm(q_pert)
    return X_world + d_X, q_pert, p_cam + d_p_cam


@pytest.mark.parametrize("dim", range(9))
def test_reprojection_jacobians_match_nonlinear_perturbation(dim):
    rng = np.random.default_rng(1000 + dim)
    X_world = rng.normal(size=3) * 2 + np.array([0.0, 0.0, 5.0])  # keep in front of the camera
    q_cam = _random_unit_quaternion(rng)
    p_cam = rng.normal(size=3)
    bearing_obs = rng.normal(size=2) * 0.05  # arbitrary "observed" bearing, doesn't need to match X_world

    r0, H_clone, H_f = reprojection_residual_and_jacobians(X_world, q_cam, p_cam, bearing_obs)
    H = np.concatenate([H_clone, H_f], axis=1)  # (2, 9): [theta_cam, p_cam, X_world]

    X_pert, q_pert, p_pert = _perturb(X_world, q_cam, p_cam, dim, EPS)
    r1, _, _ = reprojection_residual_and_jacobians(X_pert, q_pert, p_pert, bearing_obs)

    delta = np.zeros(9)
    delta[dim] = EPS
    predicted = r0 + H @ delta

    assert np.linalg.norm(r1 - predicted) < 50 * EPS ** 1.5, f"dim={dim}: r1={r1}, predicted={predicted}"


def test_H_clone_position_block_equals_negative_H_f():
    # exact algebraic identity (not just first-order): translating the camera
    # is geometrically equivalent to translating the feature the other way
    rng = np.random.default_rng(42)
    X_world = rng.normal(size=3) * 2 + np.array([0.0, 0.0, 5.0])
    q_cam = _random_unit_quaternion(rng)
    p_cam = rng.normal(size=3)
    _, H_clone, H_f = reprojection_residual_and_jacobians(X_world, q_cam, p_cam, np.zeros(2))
    assert np.allclose(H_clone[:, 3:6], -H_f, atol=1e-12)


def _synthetic_track(rng, n_views=4):
    X_world = np.array([0.2, -0.3, 5.0])
    camera_poses = []
    for i in range(n_views):
        p = np.array([0.3 * i, 0.1 * i, 0.0]) + rng.normal(size=3) * 0.02
        q = axis_angle_to_quat(rng.normal(size=3) * 0.05)
        camera_poses.append((q, p))
    bearings = []
    for q, p in camera_poses:
        r, _, _ = reprojection_residual_and_jacobians(X_world, q, p, np.zeros(2))
        bearings.append(-r)  # r = obs - h with obs=0, so h = -r
    return X_world, camera_poses, bearings


def test_null_space_projection_shapes():
    rng = np.random.default_rng(2)
    X_world, camera_poses, bearings = _synthetic_track(rng, n_views=5)
    clone_indices = list(range(5))
    r, H_x, H_f = stack_feature_observations(X_world, camera_poses, bearings, clone_indices, n_clones=5)
    assert r.shape == (10,)
    assert H_x.shape == (10, 15 + 6 * 5)
    assert H_f.shape == (10, 3)

    r_o, H_o = null_space_project(r, H_x, H_f)
    assert r_o.shape == (7,)  # 2*5 - 3
    assert H_o.shape == (7, 15 + 6 * 5)


def test_null_space_projection_requires_at_least_two_observations():
    rng = np.random.default_rng(3)
    X_world, camera_poses, bearings = _synthetic_track(rng, n_views=1)
    r, H_x, H_f = stack_feature_observations(X_world, camera_poses, bearings, [0], n_clones=1)
    with pytest.raises(ValueError):
        null_space_project(r, H_x, H_f)


@pytest.mark.parametrize("eps", [1e-3, 1e-4])
def test_null_space_projection_eliminates_feature_position_dependence(eps):
    rng = np.random.default_rng(4)
    X_world, camera_poses, bearings = _synthetic_track(rng, n_views=6)
    clone_indices = list(range(6))
    r0, H_x0, H_f0 = stack_feature_observations(X_world, camera_poses, bearings, clone_indices, n_clones=6)

    Q, _ = np.linalg.qr(H_f0, mode="complete")
    null_basis = Q[:, 3:]  # same basis null_space_project would compute from H_f0

    direction = rng.normal(size=3)
    direction = direction / np.linalg.norm(direction)
    X_perturbed = X_world + direction * eps

    r1, _, _ = stack_feature_observations(X_perturbed, camera_poses, bearings, clone_indices, n_clones=6)

    projected_diff = np.linalg.norm(null_basis.T @ (r1 - r0))
    # first-order feature-position dependence should be exactly eliminated;
    # only the O(eps^2) nonlinear remainder should survive
    assert projected_diff < 50 * eps ** 1.5, f"eps={eps}: projected_diff={projected_diff}"


def test_null_space_projection_remainder_scales_quadratically():
    # confirms the above isn't just a loose tolerance: halving eps should
    # shrink the remainder by ~4x (quadratic), not ~2x (which would mean a
    # real bug leaking first-order feature dependence through)
    rng = np.random.default_rng(5)
    X_world, camera_poses, bearings = _synthetic_track(rng, n_views=6)
    clone_indices = list(range(6))
    r0, _, H_f0 = stack_feature_observations(X_world, camera_poses, bearings, clone_indices, n_clones=6)
    Q, _ = np.linalg.qr(H_f0, mode="complete")
    null_basis = Q[:, 3:]

    direction = rng.normal(size=3)
    direction = direction / np.linalg.norm(direction)

    diffs = []
    for eps in [8e-4, 4e-4]:
        r1, _, _ = stack_feature_observations(X_world + direction * eps, camera_poses, bearings,
                                               clone_indices, n_clones=6)
        diffs.append(np.linalg.norm(null_basis.T @ (r1 - r0)))

    ratio = diffs[0] / diffs[1]
    assert 3.0 < ratio < 5.0, f"expected ~4x (quadratic) shrinkage, got {ratio:.2f}x"


def test_measurement_model_on_real_triangulated_track():
    K, dist_coeffs = load_cam0_intrinsics()
    T_BS_cam0 = load_T_BS(f"{gt.MAV0_DIR}/cam0/sensor.yaml")

    gt_timestamps, _, _, _ = gt._load_state_ground_truth()
    cam0_frames = gt.load_cam0_timestamps()
    start_frame = int(np.searchsorted(cam0_frames, gt_timestamps[0])) + 30

    tracker = FeatureTracker(max_features=150)
    for timestamp, image in iter_cam0_frames(max_frames=15, start_frame=start_frame):
        tracker.process_frame(image, timestamp)

    track_id = max(tracker.tracks, key=lambda tid: len(tracker.tracks[tid]))
    observations = tracker.tracks[track_id]
    assert len(observations) >= 10

    camera_poses, bearings = [], []
    for timestamp, (u, v) in observations:
        q_cam = gt.interpolate_cam0_orientation(timestamp)
        p_cam = body_to_sensor_position(gt.interpolate_ground_truth_position(timestamp),
                                         gt.interpolate_ground_truth_orientation(timestamp), T_BS_cam0)
        camera_poses.append((q_cam, p_cam))
        bearings.append(undistort_normalized([[u, v]], K, dist_coeffs)[0])

    X_world = triangulate_feature(camera_poses, bearings)
    n = len(observations)
    r, H_x, H_f = stack_feature_observations(X_world, camera_poses, bearings, list(range(n)), n_clones=n)
    r_o, H_o = null_space_project(r, H_x, H_f)

    assert r_o.shape == (2 * n - 3,)
    # ground-truth poses + a refined triangulation should fit almost exactly
    assert np.linalg.norm(r_o) / len(r_o) < 1e-2
