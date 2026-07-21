import os

import numpy as np

from frames import body_to_sensor_position, body_to_sensor_quaternion, load_T_BS, sensor_to_body_quaternion
from quaternion_utils import axis_angle_to_quat, quat_multiply, quaternion_to_rotation_matrix

MAV0_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                         "machine_hall", "MH_01_easy", "MH_01_easy", "mav0")

IDENTITY_T_BS = np.eye(4)


def _random_unit_quaternion(rng):
    v = rng.normal(size=4)
    return v / np.linalg.norm(v)


def test_load_T_BS_reads_cam0_extrinsic():
    T_BS = load_T_BS(os.path.join(MAV0_DIR, "cam0", "sensor.yaml"))
    assert T_BS.shape == (4, 4)
    assert np.allclose(T_BS[3], [0.0, 0.0, 0.0, 1.0])
    R = T_BS[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6)
    # spot-check a couple of known values from cam0/sensor.yaml
    assert np.isclose(T_BS[0, 0], 0.0148655429818)
    assert np.isclose(T_BS[0, 3], -0.0216401454975)


def test_body_to_sensor_quaternion_identity_transform_is_noop():
    rng = np.random.default_rng(0)
    q_body = _random_unit_quaternion(rng)
    q_sensor = body_to_sensor_quaternion(q_body, IDENTITY_T_BS)
    assert np.allclose(q_sensor, q_body, atol=1e-9)


def test_body_to_sensor_quaternion_round_trips_with_inverse():
    rng = np.random.default_rng(1)
    q_body = _random_unit_quaternion(rng)

    T_BS = np.eye(4)
    T_BS[:3, :3] = quaternion_to_rotation_matrix(axis_angle_to_quat(rng.normal(size=3) * 0.5))
    T_BS[:3, 3] = rng.normal(size=3)

    q_sensor = body_to_sensor_quaternion(q_body, T_BS)
    q_body_back = sensor_to_body_quaternion(q_sensor, T_BS)
    if np.dot(q_body, q_body_back) < 0:
        q_body_back = -q_body_back
    assert np.allclose(q_body, q_body_back, atol=1e-9)


def test_body_to_sensor_quaternion_matches_manual_rotation_composition():
    rng = np.random.default_rng(2)
    q_body = _random_unit_quaternion(rng)
    T_BS = np.eye(4)
    T_BS[:3, :3] = quaternion_to_rotation_matrix(axis_angle_to_quat(rng.normal(size=3) * 0.7))

    q_sensor = body_to_sensor_quaternion(q_body, T_BS)
    R_sensor_expected = quaternion_to_rotation_matrix(q_body) @ T_BS[:3, :3]
    R_sensor_actual = quaternion_to_rotation_matrix(q_sensor)
    assert np.allclose(R_sensor_actual, R_sensor_expected, atol=1e-9)


def test_body_to_sensor_position_identity_transform_is_noop():
    rng = np.random.default_rng(3)
    p_body = rng.normal(size=3)
    q_body = _random_unit_quaternion(rng)
    assert np.allclose(body_to_sensor_position(p_body, q_body, IDENTITY_T_BS), p_body)


def test_body_to_sensor_position_pure_translation_along_body_axis():
    # body facing "identity" orientation: sensor offset should apply directly
    # in world coordinates with no rotation involved
    p_body = np.array([1.0, 2.0, 3.0])
    q_body = np.array([1.0, 0.0, 0.0, 0.0])
    T_BS = np.eye(4)
    T_BS[:3, 3] = [0.5, 0.0, 0.0]
    p_sensor = body_to_sensor_position(p_body, q_body, T_BS)
    assert np.allclose(p_sensor, [1.5, 2.0, 3.0])


def test_body_to_sensor_position_rotated_body_applies_lever_arm_in_world_frame():
    p_body = np.zeros(3)
    # body rotated 90 degrees about Z: body's local +X axis now points along world +Y
    q_body = axis_angle_to_quat(np.array([0.0, 0.0, np.pi / 2]))
    T_BS = np.eye(4)
    T_BS[:3, 3] = [1.0, 0.0, 0.0]  # sensor offset along body +X
    p_sensor = body_to_sensor_position(p_body, q_body, T_BS)
    assert np.allclose(p_sensor, [0.0, 1.0, 0.0], atol=1e-9)
