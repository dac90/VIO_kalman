import numpy as np

import ground_truth as gt
from feature_tracker import FeatureTracker, iter_cam0_frames
from frames import body_to_sensor_position, load_T_BS
from quaternion_utils import axis_angle_to_quat, quat_conjugate, rotate_vector_by_quaternion
from triangulation import (
    linear_triangulate,
    load_cam0_intrinsics,
    refine_triangulation,
    reprojection_residuals,
    triangulate_feature,
    undistort_normalized,
)


def _project_true(X_world, q_world_cam, p_world_cam):
    """Independent (not reusing triangulation.py's own _project) ground-truth projection for tests."""
    q_cam_world = quat_conjugate(q_world_cam)
    Xc = rotate_vector_by_quaternion(q_cam_world, X_world - p_world_cam)
    return np.array([Xc[0] / Xc[2], Xc[1] / Xc[2]])


def _synthetic_cameras(rng, n=4):
    """A handful of camera poses with real baseline/parallax, orbiting roughly toward the origin."""
    poses = []
    for i in range(n):
        p = np.array([2.0 * np.cos(i * 0.7), 2.0 * np.sin(i * 0.7), 0.3 * i]) + rng.normal(size=3) * 0.05
        q = axis_angle_to_quat(rng.normal(size=3) * 0.1)  # mild rotation variety, not just pure translation
        poses.append((q, p))
    return poses


def test_linear_triangulate_recovers_point_exactly_noiseless():
    rng = np.random.default_rng(0)
    X_true = np.array([0.3, -0.2, 4.0])
    cameras = _synthetic_cameras(rng)
    bearings = [_project_true(X_true, q, p) for q, p in cameras]

    X_est = linear_triangulate(cameras, bearings)
    assert np.allclose(X_est, X_true, atol=1e-8)


def test_triangulate_feature_recovers_point_exactly_noiseless():
    rng = np.random.default_rng(1)
    X_true = np.array([-1.0, 0.5, 6.0])
    cameras = _synthetic_cameras(rng)
    bearings = [_project_true(X_true, q, p) for q, p in cameras]

    X_est = triangulate_feature(cameras, bearings)
    assert np.allclose(X_est, X_true, atol=1e-9)


def test_refine_triangulation_reduces_reprojection_error_under_noise():
    rng = np.random.default_rng(2)
    X_true = np.array([0.5, 0.5, 5.0])
    cameras = _synthetic_cameras(rng, n=6)
    bearings = [_project_true(X_true, q, p) + rng.normal(size=2) * 1e-3 for q, p in cameras]

    X_linear = linear_triangulate(cameras, bearings)
    X_refined = refine_triangulation(X_linear, cameras, bearings)

    error_linear = np.linalg.norm(reprojection_residuals(X_linear, cameras, bearings))
    error_refined = np.linalg.norm(reprojection_residuals(X_refined, cameras, bearings))
    assert error_refined <= error_linear
    assert np.linalg.norm(X_refined - X_true) < 0.05  # noise is tiny, should recover close to truth


def test_triangulate_feature_recovers_point_under_more_noise():
    rng = np.random.default_rng(3)
    X_true = np.array([1.2, -0.8, 8.0])
    cameras = _synthetic_cameras(rng, n=8)
    bearings = [_project_true(X_true, q, p) + rng.normal(size=2) * 5e-3 for q, p in cameras]

    X_est = triangulate_feature(cameras, bearings)
    assert np.linalg.norm(X_est - X_true) < 0.5


def test_undistort_normalized_matches_manual_pinhole_for_zero_distortion():
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    zero_dist = np.zeros(4)
    pixel = np.array([[420.0, 340.0]])  # 100px right/down of principal point
    normalized = undistort_normalized(pixel, K, zero_dist)
    assert np.allclose(normalized[0], [100.0 / 500.0, 100.0 / 500.0], atol=1e-6)


def test_load_cam0_intrinsics_matches_known_sensor_yaml_values():
    K, dist_coeffs = load_cam0_intrinsics()
    assert np.isclose(K[0, 0], 458.654)
    assert np.isclose(K[1, 1], 457.296)
    assert np.isclose(K[0, 2], 367.215)
    assert np.isclose(K[1, 2], 248.375)
    assert len(dist_coeffs) == 4
    assert np.isclose(dist_coeffs[0], -0.28340811)


def test_triangulate_real_track_gives_plausible_positive_depth():
    K, dist_coeffs = load_cam0_intrinsics()
    T_BS_cam0 = load_T_BS(f"{gt.MAV0_DIR}/cam0/sensor.yaml")

    # ground truth doesn't cover cam0's first ~1s (see ground_truth.py), so
    # start well past that so every observation has a real (non-clamped) pose
    gt_timestamps, _, _, _ = gt._load_state_ground_truth()
    cam0_frames = gt.load_cam0_timestamps()
    start_frame = int(np.searchsorted(cam0_frames, gt_timestamps[0])) + 30

    tracker = FeatureTracker(max_features=150)
    for timestamp, image in iter_cam0_frames(max_frames=15, start_frame=start_frame):
        tracker.process_frame(image, timestamp)

    # pick a track that survived the whole window, for a real multi-view baseline
    track_id = max(tracker.tracks, key=lambda tid: len(tracker.tracks[tid]))
    observations = tracker.tracks[track_id]
    assert len(observations) >= 10

    cameras = []
    bearings = []
    for timestamp, (u, v) in observations:
        q_body = gt.interpolate_ground_truth_orientation(timestamp)
        p_body = gt.interpolate_ground_truth_position(timestamp)
        q_cam = gt.interpolate_cam0_orientation(timestamp)
        p_cam = body_to_sensor_position(p_body, q_body, T_BS_cam0)
        cameras.append((q_cam, p_cam))
        bearings.append(undistort_normalized([[u, v]], K, dist_coeffs)[0])

    X_world = triangulate_feature(cameras, bearings)

    # cheirality: the point should be in front of every observing camera
    for q_cam, p_cam in cameras:
        q_world_cam_conj = quat_conjugate(q_cam)
        Xc = rotate_vector_by_quaternion(q_world_cam_conj, X_world - p_cam)
        assert Xc[2] > 0, "triangulated point ended up behind a camera that observed it"
        assert 0.1 < Xc[2] < 50.0, f"depth {Xc[2]} implausible for an indoor machine-hall scene"
