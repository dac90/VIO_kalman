"""Quaternion math shared across datasets.

Convention throughout: a quaternion q = [w, x, y, z] represents a rotation
from some frame A into frame B such that rotate_vector_by_quaternion(q, v_A)
== v_B (equivalently, quaternion_to_rotation_matrix(q) @ v_A == v_B).
"""
import numpy as np


def skew_symmetric(v):
    """3x3 skew-symmetric cross-product matrix [v]x, such that [v]x @ u == v x u."""
    x, y, z = v
    return np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ])


def quat_to_euler(q):
    """Convert unit quaternion [w, x, y, z] to Euler angles (roll, pitch, yaw) in radians.

    Uses the ZYX (yaw-pitch-roll) convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    """
    w, x, y, z = q

    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)  # guard against numerical drift past +/-1 at the poles
    pitch = np.arcsin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


def euler_to_quat(euler):
    """Convert Euler angles (roll, pitch, yaw) in radians to unit quaternion [w, x, y, z].

    Uses the ZYX (yaw-pitch-roll) convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    """
    roll, pitch, yaw = euler

    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])


def axis_angle_to_quat(rotation_vector):
    """Convert a rotation vector (axis * angle, radians) to a unit quaternion [w, x, y, z].

    Exact for a constant angular velocity integrated over rotation_vector =
    omega * dt: this is the quaternion exponential map, not the small-angle
    Euler-angle composition euler_to_quat performs.
    """
    theta = np.linalg.norm(rotation_vector)
    if theta < 1e-8:
        # first-order Taylor expansion of sin(theta/2)/theta avoids a 0/0 divide
        return np.array([1.0, *(0.5 * np.asarray(rotation_vector))])

    axis = np.asarray(rotation_vector) / theta
    half = theta / 2.0
    return np.array([np.cos(half), *(axis * np.sin(half))])


def quat_error_vector(q_true, q_hat):
    """Small-angle global-frame rotation vector delta_theta with q_true ~= axis_angle_to_quat(delta_theta) (x) q_hat.

    Exact inverse of axis_angle_to_quat in the small-angle limit; used to
    compare a perturbed/estimated orientation against a reference one in the
    error-state (delta_theta) representation used by msckf_state.py.
    """
    q_err = quat_multiply(np.asarray(q_true, dtype=float), quat_conjugate(np.asarray(q_hat, dtype=float)))
    if q_err[0] < 0:
        q_err = -q_err
    return 2.0 * q_err[1:4]


def rotate_vector_by_quaternion(q, v):
    """Rotate 3-vector v from frame A into frame B using unit quaternion q = [w, x, y, z]."""
    w = q[0]
    qv = np.asarray(q[1:4])
    v = np.asarray(v)
    t = 2 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def quat_conjugate(q):
    """Conjugate of unit quaternion q = [w, x, y, z]; equals its inverse since q is unit-norm."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quat_multiply(q1, q2):
    """Hamilton product q1 (x) q2: rotating by the result applies q2's rotation first, then q1's."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def slerp(q0, q1, alpha):
    """Spherical linear interpolation between unit quaternions q0 and q1 at fraction alpha in [0, 1]."""
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)

    dot = np.dot(q0, q1)
    if dot < 0.0:
        # take the shorter path around the hypersphere: q and -q represent
        # the same rotation, so flipping the sign here avoids interpolating
        # "the long way around", which is what produces erratic jumps
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)

    if dot > 0.9995:
        # nearly identical rotations: linear interpolation is numerically
        # safer here since sin(theta_0) below would be close to zero
        result = q0 + alpha * (q1 - q0)
        return result / np.linalg.norm(result)

    theta_0 = np.arccos(dot)
    theta = theta_0 * alpha
    sin_theta_0 = np.sin(theta_0)
    s0 = np.cos(theta) - dot * np.sin(theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    return s0 * q0 + s1 * q1


def quaternion_to_rotation_matrix(q):
    """Convert unit quaternion [w, x, y, z] to its 3x3 rotation matrix R, where R @ v_A == v_B."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rotation_matrix_to_quaternion(R):
    """Convert a 3x3 rotation matrix R (proper, orthonormal) to a unit quaternion [w, x, y, z].

    Numerically stable (Shepperd's method): picks whichever of w/x/y/z has the
    largest magnitude as the "pivot" to divide by, avoiding division by a
    near-zero term.
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    return np.array([qw, qx, qy, qz])
