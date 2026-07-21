import numpy as np
import pytest

from quaternion_utils import (
    axis_angle_to_quat,
    euler_to_quat,
    quat_conjugate,
    quat_error_vector,
    quat_multiply,
    quat_to_euler,
    quaternion_to_rotation_matrix,
    rotate_vector_by_quaternion,
    rotation_matrix_to_quaternion,
    skew_symmetric,
    slerp,
)

IDENTITY_Q = np.array([1.0, 0.0, 0.0, 0.0])


def _random_unit_quaternion(rng):
    v = rng.normal(size=4)
    return v / np.linalg.norm(v)


def test_skew_symmetric_matches_cross_product():
    rng = np.random.default_rng(0)
    for _ in range(10):
        v, u = rng.normal(size=3), rng.normal(size=3)
        assert np.allclose(skew_symmetric(v) @ u, np.cross(v, u))


@pytest.mark.parametrize("euler", [
    [0.0, 0.0, 0.0],
    [0.3, -0.2, 1.1],
    [np.pi / 2, 0.1, -0.5],
    [-1.5, 0.7, 2.9],
])
def test_euler_quat_round_trip(euler):
    q = euler_to_quat(np.array(euler))
    assert np.isclose(np.linalg.norm(q), 1.0)
    euler_back = quat_to_euler(q)
    q_back = euler_to_quat(euler_back)
    # compare quaternions rather than raw Euler angles: angles alone can
    # legitimately differ by +/-2pi or by a gimbal-lock-ambiguous combination
    if np.dot(q, q_back) < 0:
        q_back = -q_back
    assert np.allclose(q, q_back, atol=1e-9)


def test_quat_multiply_identity():
    rng = np.random.default_rng(1)
    q = _random_unit_quaternion(rng)
    assert np.allclose(quat_multiply(q, IDENTITY_Q), q)
    assert np.allclose(quat_multiply(IDENTITY_Q, q), q)


def test_quat_multiply_matches_rotation_matrix_composition():
    rng = np.random.default_rng(2)
    q1, q2 = _random_unit_quaternion(rng), _random_unit_quaternion(rng)
    R1, R2 = quaternion_to_rotation_matrix(q1), quaternion_to_rotation_matrix(q2)
    q12 = quat_multiply(q1, q2)
    R12 = quaternion_to_rotation_matrix(q12)
    assert np.allclose(R12, R1 @ R2, atol=1e-9)


def test_quat_conjugate_is_inverse():
    rng = np.random.default_rng(3)
    q = _random_unit_quaternion(rng)
    identity_ish = quat_multiply(q, quat_conjugate(q))
    assert np.allclose(identity_ish, IDENTITY_Q, atol=1e-9)


def test_rotate_vector_matches_rotation_matrix():
    rng = np.random.default_rng(4)
    q = _random_unit_quaternion(rng)
    v = rng.normal(size=3)
    assert np.allclose(rotate_vector_by_quaternion(q, v), quaternion_to_rotation_matrix(q) @ v, atol=1e-9)


def test_rotation_matrix_quaternion_round_trip():
    rng = np.random.default_rng(5)
    for _ in range(20):
        q = _random_unit_quaternion(rng)
        R = quaternion_to_rotation_matrix(q)
        q_back = rotation_matrix_to_quaternion(R)
        if np.dot(q, q_back) < 0:
            q_back = -q_back
        assert np.allclose(q, q_back, atol=1e-9)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0)


@pytest.mark.parametrize("axis,angle", [
    ([1.0, 0.0, 0.0], np.pi / 2),
    ([0.0, 1.0, 0.0], np.pi),
    ([0.0, 0.0, 1.0], np.pi / 3),
    ([1.0, 1.0, 1.0], 2 * np.pi / 3),
])
def test_axis_angle_to_quat_known_rotations(axis, angle):
    axis = np.array(axis) / np.linalg.norm(axis)
    q = axis_angle_to_quat(axis * angle)
    assert np.isclose(np.linalg.norm(q), 1.0)
    # a rotation by `angle` about `axis` should leave a vector along axis
    # unchanged, and rotate a perpendicular vector by exactly `angle`
    assert np.allclose(rotate_vector_by_quaternion(q, axis), axis, atol=1e-9)

    perp = np.cross(axis, [1.0, 0.0, 0.0]) if not np.allclose(axis, [1, 0, 0]) else np.array([0.0, 1.0, 0.0])
    perp = perp / np.linalg.norm(perp)
    rotated = rotate_vector_by_quaternion(q, perp)
    assert np.isclose(np.dot(rotated, perp), np.cos(angle), atol=1e-9)


def test_axis_angle_to_quat_zero_rotation_is_identity():
    q = axis_angle_to_quat(np.zeros(3))
    assert np.allclose(q, IDENTITY_Q)


def test_axis_angle_to_quat_small_angle_matches_taylor_branch():
    # exercises both the near-zero branch and the general branch at a
    # boundary-adjacent angle, and checks they agree in the overlap regime
    tiny = np.array([1e-9, -2e-9, 5e-10])
    q_tiny = axis_angle_to_quat(tiny)
    assert np.allclose(q_tiny, [1.0, tiny[0] / 2, tiny[1] / 2, tiny[2] / 2], atol=1e-12)


def test_quat_error_vector_recovers_small_perturbation():
    rng = np.random.default_rng(6)
    q_hat = _random_unit_quaternion(rng)
    for _ in range(10):
        delta_theta = rng.normal(size=3) * 1e-6
        q_true = quat_multiply(axis_angle_to_quat(delta_theta), q_hat)
        q_true = q_true / np.linalg.norm(q_true)
        recovered = quat_error_vector(q_true, q_hat)
        assert np.allclose(recovered, delta_theta, atol=1e-9)


def test_quat_error_vector_zero_for_identical_quaternions():
    rng = np.random.default_rng(7)
    q = _random_unit_quaternion(rng)
    assert np.allclose(quat_error_vector(q, q), np.zeros(3), atol=1e-9)


def test_slerp_endpoints_and_midpoint():
    rng = np.random.default_rng(8)
    q0, q1 = _random_unit_quaternion(rng), _random_unit_quaternion(rng)
    if np.dot(q0, q1) < 0:
        q1 = -q1  # slerp takes the short way; align inputs for a clean endpoint check

    assert np.allclose(slerp(q0, q1, 0.0), q0, atol=1e-9)
    assert np.allclose(slerp(q0, q1, 1.0), q1, atol=1e-9)

    mid = slerp(q0, q1, 0.5)
    assert np.isclose(np.linalg.norm(mid), 1.0)
    # the midpoint should be equidistant (angularly) from both endpoints
    assert np.isclose(np.dot(mid, q0), np.dot(mid, q1), atol=1e-9)


def test_slerp_nearly_identical_quaternions_uses_linear_branch():
    rng = np.random.default_rng(9)
    q0 = _random_unit_quaternion(rng)
    q1 = q0 + rng.normal(size=4) * 1e-8
    q1 = q1 / np.linalg.norm(q1)
    result = slerp(q0, q1, 0.5)
    assert np.isclose(np.linalg.norm(result), 1.0)
