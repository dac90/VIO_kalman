import numpy as np

import ground_truth as gt
from frames import sensor_to_body_quaternion


def test_load_cam0_timestamps_sorted_and_reasonable_length():
    timestamps = gt.load_cam0_timestamps()
    assert len(timestamps) > 1000
    assert np.all(np.diff(timestamps) > 0)


def test_load_state_ground_truth_shapes_and_sorted():
    timestamps, positions, quaternions, velocities = gt._load_state_ground_truth()
    n = len(timestamps)
    assert positions.shape == (n, 3)
    assert quaternions.shape == (n, 4)
    assert velocities.shape == (n, 3)
    assert np.all(np.diff(timestamps) > 0)
    # ground-truth quaternions should already be close to unit-norm (not just
    # any 4-vector); tolerance is loose since the CSV only stores 6 decimal
    # digits, which alone accounts for deviations up to a few 1e-6
    assert np.allclose(np.linalg.norm(quaternions, axis=1), 1.0, atol=1e-4)


def test_interpolate_ground_truth_position_matches_raw_at_exact_timestamp():
    timestamps, positions, _, _ = gt._load_state_ground_truth()
    gt._gt_position_interpolator = None  # reset module cache so this test is order-independent
    t = int(timestamps[10])
    result = gt.interpolate_ground_truth_position(t)
    assert np.allclose(result, positions[10])


def test_interpolate_ground_truth_orientation_matches_raw_at_exact_timestamp():
    timestamps, _, quaternions, _ = gt._load_state_ground_truth()
    gt._gt_orientation_interpolator = None
    t = int(timestamps[10])
    result = gt.interpolate_ground_truth_orientation(t)
    if np.dot(result, quaternions[10]) < 0:
        result = -result
    assert np.allclose(result, quaternions[10], atol=1e-9)


def test_interpolate_ground_truth_velocity_matches_raw_at_exact_timestamp():
    timestamps, _, _, velocities = gt._load_state_ground_truth()
    gt._gt_velocity_interpolator = None
    t = int(timestamps[10])
    result = gt.interpolate_ground_truth_velocity(t)
    assert np.allclose(result, velocities[10])


def test_interpolate_cam0_orientation_round_trips_to_body_orientation():
    gt._gt_orientation_interpolator = None
    gt._cam0_T_BS = None
    timestamps, _, quaternions, _ = gt._load_state_ground_truth()
    t = int(timestamps[100])

    q_cam0 = gt.interpolate_cam0_orientation(t)
    assert np.isclose(np.linalg.norm(q_cam0), 1.0)

    q_body_recovered = sensor_to_body_quaternion(q_cam0, gt._cam0_T_BS)
    expected = quaternions[100]
    if np.dot(q_body_recovered, expected) < 0:
        q_body_recovered = -q_body_recovered
    assert np.allclose(q_body_recovered, expected, atol=1e-9)


def test_cam0_timestamps_mostly_within_ground_truth_range():
    # ground truth starts/ends slightly inside cam0's own range (see the
    # __main__ trimming logic), but the two ranges should still overlap
    # substantially, not be disjoint
    cam0_ts = gt.load_cam0_timestamps()
    gt_ts, _, _, _ = gt._load_state_ground_truth()
    in_range = (cam0_ts >= gt_ts[0]) & (cam0_ts <= gt_ts[-1])
    assert in_range.sum() > 0.9 * len(cam0_ts)
