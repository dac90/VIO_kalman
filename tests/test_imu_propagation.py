import numpy as np

from imu_propagation import GRAVITY_WORLD, _load_imu_measurements, propagate_imu_trajectory, propagate_step
from quaternion_utils import axis_angle_to_quat, quat_multiply

IDENTITY_Q = np.array([1.0, 0.0, 0.0, 0.0])


def test_propagate_step_free_fall_from_rest():
    # zero specific force (e.g. sensor in free fall / a drop test): the only
    # acceleration is gravity, so velocity and position should follow simple
    # kinematics exactly
    p0, v0, q0 = np.zeros(3), np.zeros(3), IDENTITY_Q
    dt = 0.1
    p1, v1, q1 = propagate_step(p0, v0, q0, gyro=np.zeros(3), accel=np.zeros(3), dt=dt)

    assert np.allclose(v1, GRAVITY_WORLD * dt)
    assert np.allclose(p1, 0.5 * GRAVITY_WORLD * dt ** 2)
    assert np.allclose(q1, IDENTITY_Q)  # no rotation


def test_propagate_step_stationary_accel_cancels_gravity():
    # accelerometer reading that exactly cancels gravity's reaction (as if
    # sitting still on a table) should leave velocity/position unchanged
    p0, v0, q0 = np.array([1.0, 2.0, 3.0]), np.zeros(3), IDENTITY_Q
    accel = -GRAVITY_WORLD  # world-frame accel = R@accel + GRAVITY_WORLD = 0 when q=identity
    p1, v1, q1 = propagate_step(p0, v0, q0, gyro=np.zeros(3), accel=accel, dt=0.05)
    assert np.allclose(v1, np.zeros(3), atol=1e-12)
    assert np.allclose(p1, p0, atol=1e-12)


def test_propagate_step_pure_rotation_matches_axis_angle():
    p0, v0, q0 = np.zeros(3), np.zeros(3), IDENTITY_Q
    gyro = np.array([0.0, 0.0, np.pi / 2])  # 90 deg/s about Z
    dt = 1.0
    _, _, q1 = propagate_step(p0, v0, q0, gyro=gyro, accel=-GRAVITY_WORLD, dt=dt)
    expected_q = quat_multiply(axis_angle_to_quat(gyro * dt), q0)
    if np.dot(q1, expected_q) < 0:
        expected_q = -expected_q
    assert np.allclose(q1, expected_q, atol=1e-9)


def test_propagate_step_respects_bias_correction():
    p0, v0, q0 = np.zeros(3), np.zeros(3), IDENTITY_Q
    bias_gyro = np.array([0.1, 0.0, 0.0])
    bias_accel = -GRAVITY_WORLD  # chosen so raw accel=0 cancels gravity once bias-corrected accel is added back
    # raw measurement equals the bias exactly -> bias-corrected omega/accel are both zero
    _, v1, q1 = propagate_step(p0, v0, q0, gyro=bias_gyro, accel=bias_accel, dt=0.2,
                                bias_gyro=bias_gyro, bias_accel=bias_accel)
    assert np.allclose(q1, IDENTITY_Q, atol=1e-12)
    assert np.allclose(v1, GRAVITY_WORLD * 0.2)  # bias-corrected accel is zero, so only gravity acts


def test_propagate_step_quaternion_stays_unit_norm():
    rng = np.random.default_rng(0)
    p, v, q = np.zeros(3), np.zeros(3), IDENTITY_Q
    for _ in range(200):
        gyro = rng.normal(size=3)
        accel = rng.normal(size=3) * 5
        p, v, q = propagate_step(p, v, q, gyro, accel, dt=0.005)
        assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-9)


def test_propagate_imu_trajectory_applies_constant_bias_when_given():
    imu_timestamps, gyro, accel = _load_imu_measurements()
    start_idx = 50
    t0 = int(imu_timestamps[start_idx])
    p0, v0, q0 = np.zeros(3), np.zeros(3), IDENTITY_Q
    bias_gyro = np.array([0.01, -0.02, 0.03])
    bias_accel = np.array([0.1, 0.0, -0.1])

    _, _, _, quaternions_biased = propagate_imu_trajectory(p0, v0, q0, t0, bias_gyro=bias_gyro, bias_accel=bias_accel)
    _, _, _, quaternions_zero_bias = propagate_imu_trajectory(p0, v0, q0, t0)

    # a nonzero bias should visibly change the propagated trajectory...
    assert not np.allclose(quaternions_biased[10], quaternions_zero_bias[10])

    # ...and should match manually replaying propagate_step with that same bias
    p, v, q = p0, v0, q0
    for i in range(5):
        dt = (imu_timestamps[start_idx + i + 1] - imu_timestamps[start_idx + i]) / 1e9
        p, v, q = propagate_step(p, v, q, gyro[start_idx + i], accel[start_idx + i], dt,
                                  bias_gyro=bias_gyro, bias_accel=bias_accel)
    _, positions_biased, velocities_biased, _ = propagate_imu_trajectory(
        p0, v0, q0, t0, bias_gyro=bias_gyro, bias_accel=bias_accel)
    assert np.allclose(positions_biased[5], p)
    assert np.allclose(velocities_biased[5], v)


def test_propagate_imu_trajectory_starts_at_given_initial_state():
    p0, v0, q0 = np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]), IDENTITY_Q
    imu_timestamps, _, _ = _load_imu_measurements()
    t0 = int(imu_timestamps[100])  # an exact sample timestamp, mid-sequence

    timestamps, positions, velocities, quaternions = propagate_imu_trajectory(p0, v0, q0, t0)
    assert timestamps[0] == t0
    assert np.allclose(positions[0], p0)
    assert np.allclose(velocities[0], v0)
    assert np.allclose(quaternions[0], q0)
    assert np.all(np.diff(timestamps) > 0)


def test_propagate_imu_trajectory_matches_manual_step_loop():
    imu_timestamps, gyro, accel = _load_imu_measurements()
    start_idx = 50
    t0 = int(imu_timestamps[start_idx])
    p0, v0, q0 = np.zeros(3), np.zeros(3), IDENTITY_Q

    timestamps, positions, velocities, quaternions = propagate_imu_trajectory(p0, v0, q0, t0)

    # manually replay the first few steps and check they match exactly
    p, v, q = p0, v0, q0
    for i in range(5):
        dt = (imu_timestamps[start_idx + i + 1] - imu_timestamps[start_idx + i]) / 1e9
        p, v, q = propagate_step(p, v, q, gyro[start_idx + i], accel[start_idx + i], dt)
        assert np.allclose(positions[i + 1], p)
        assert np.allclose(velocities[i + 1], v)
        assert np.allclose(quaternions[i + 1], q)


def test_load_imu_measurements_sorted_and_shaped():
    timestamps, gyro, accel = _load_imu_measurements()
    assert len(timestamps) == len(gyro) == len(accel)
    assert gyro.shape[1] == 3
    assert accel.shape[1] == 3
    assert np.all(np.diff(timestamps) > 0)
