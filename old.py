import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "AGZ_subset", "Log Files")

# timestamps in this dataset are in microseconds (PX4/Pixhawk hrt_absolute_time
# convention), so 10ms of tolerance is 10,000 units
EXACT_MATCH_TOLERANCE = 10_000

# WGS84 geographic (lat, lon) -> WGS 84 / UTM zone 32N, matching the
# GroundTruthAGL x_gt/y_gt/z_gt coordinate system
_WGS84_TO_UTM32N = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)


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


def rotate_vector_by_quaternion(q, v):
    """Rotate 3-vector v from body frame into world frame using unit quaternion q = [w, x, y, z]."""
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


# Fixed camera-frame -> vehicle-body-frame mounting rotation (a +90 degree
# rotation about the body X axis), taken from main.py's R_cb. omega_gt/phi_gt/
# kappa_gt gives the world-to-camera rotation; this is composed with its
# inverse (camera-to-world) below to express ground-truth orientation in the
# same world-from-body convention as the onboard attitude quaternion.
R_CB = np.array([[1.0, 0.0, 0.0],
                  [0.0, 0.0, -1.0],
                  [0.0, 1.0, 0.0]])
Q_CB = euler_to_quat(np.array([np.pi / 2, 0.0, 0.0]))

# Fixed world-frame yaw reference correction: the UTM-based omega/phi/kappa
# yaw reference and the onboard attitude's own yaw reference are offset by a
# further +90 degrees about the world Z axis (confirmed against the position
# data). Left-multiplying (rather than composing on the body side) shifts
# yaw only, leaving the already-correct roll/pitch untouched.
Q_WORLD_YAW_FIX = euler_to_quat(np.array([0.0, 0.0, np.pi / 2]))


class SequentialInterpolator:
    """Linear interpolator over a monotonically increasing timestamp series.

    Keeps a pointer to the last bracket used so that repeated calls with
    increasing query timestamps resume searching from there instead of
    restarting from the beginning of the series each time.
    """

    def __init__(self, timestamps, positions):
        self.timestamps = np.asarray(timestamps, dtype=float)
        self.positions = np.asarray(positions, dtype=float)
        self._idx = 0

    def interpolate(self, t):
        ts = self.timestamps
        n = len(ts)

        if t <= ts[0]:
            self._idx = 0
            return self.positions[0].copy()
        if t >= ts[-1]:
            self._idx = n - 2
            return self.positions[-1].copy()

        if t < ts[self._idx]:
            self._idx = 0
        while self._idx < n - 2 and ts[self._idx + 1] < t:
            self._idx += 1

        t0, t1 = ts[self._idx], ts[self._idx + 1]

        # Exact Match
        if abs(t - t0) <= EXACT_MATCH_TOLERANCE:
            return self.positions[self._idx].copy()
        if abs(t - t1) <= EXACT_MATCH_TOLERANCE:
            return self.positions[self._idx + 1].copy()

        alpha = (t - t0) / (t1 - t0)
        return self.positions[self._idx] + alpha * (self.positions[self._idx + 1] - self.positions[self._idx])


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


class QuaternionSequentialInterpolator:
    """Like SequentialInterpolator, but spherically interpolates (slerps) unit quaternions.

    Linearly interpolating Euler angles (or raw quaternion components) can
    produce artifacts near angle wraparound / antipodal quaternions -
    slerp avoids both by always taking the shortest rotational path between
    samples and staying on the unit quaternion manifold throughout.
    """

    def __init__(self, timestamps, quaternions):
        self.timestamps = np.asarray(timestamps, dtype=float)
        q = np.asarray(quaternions, dtype=float)
        self.quaternions = q / np.linalg.norm(q, axis=1, keepdims=True)
        self._idx = 0

    def interpolate(self, t):
        ts = self.timestamps
        n = len(ts)

        if t <= ts[0]:
            self._idx = 0
            return self.quaternions[0].copy()
        if t >= ts[-1]:
            self._idx = n - 2
            return self.quaternions[-1].copy()

        if t < ts[self._idx]:
            self._idx = 0
        while self._idx < n - 2 and ts[self._idx + 1] < t:
            self._idx += 1

        t0, t1 = ts[self._idx], ts[self._idx + 1]

        # Exact Match
        if abs(t - t0) <= EXACT_MATCH_TOLERANCE:
            return self.quaternions[self._idx].copy()
        if abs(t - t1) <= EXACT_MATCH_TOLERANCE:
            return self.quaternions[self._idx + 1].copy()

        alpha = (t - t0) / (t1 - t0)
        return slerp(self.quaternions[self._idx], self.quaternions[self._idx + 1], alpha)


_gt_timestamp_lookup = None


def _get_ground_truth_timestamps(log_dir=LOG_DIR):
    """imgid -> timestamp lookup for GroundTruthAGL rows.

    GroundTruthAGL has no timestamp column of its own, so this pulls one
    from GroundTruthAGM: column 0 is the timestamp, column 1 is the imgid
    it corresponds to. Loaded once and cached for subsequent calls.
    """
    global _gt_timestamp_lookup
    if _gt_timestamp_lookup is None:
        agm = pd.read_csv(os.path.join(log_dir, "GroundTruthAGM.csv"), usecols=[0, 1])
        agm.columns = [c.strip() for c in agm.columns]
        _gt_timestamp_lookup = agm.rename(columns={"Timpstemp": "timestamp"})
    return _gt_timestamp_lookup

def _load_ground_truth_interpolator(log_dir=LOG_DIR):
    gt = pd.read_csv(os.path.join(log_dir, "GroundTruthAGL.csv"))
    gt.columns = [c.strip() for c in gt.columns]

    timestamps = _get_ground_truth_timestamps(log_dir)
    merged = gt.merge(timestamps, on="imgid").sort_values("timestamp")
    return SequentialInterpolator(merged["timestamp"].values, merged[["x_gt", "y_gt", "z_gt"]].values)


def _load_gps_interpolator(log_dir=LOG_DIR):
    gps = pd.read_csv(os.path.join(log_dir, "OnboardGPS.csv"))
    gps.columns = [c.strip() for c in gps.columns]
    gps = gps.rename(columns={"Timpstemp": "timestamp"}).sort_values("timestamp")

    # reproject lat/lon (WGS84) into WGS 84 / UTM zone 32N so it lives in the
    # same easting/northing/altitude frame as GroundTruthAGL's x_gt/y_gt/z_gt
    easting, northing = _WGS84_TO_UTM32N.transform(gps["lon"].values, gps["lat"].values)
    positions = np.column_stack([easting, northing, gps["alt"].values])
    return SequentialInterpolator(gps["timestamp"].values, positions)


def _load_ground_truth_orientation_interpolator(log_dir=LOG_DIR):
    gt = pd.read_csv(os.path.join(log_dir, "GroundTruthAGL.csv"))
    gt.columns = [c.strip() for c in gt.columns]

    timestamps = _get_ground_truth_timestamps(log_dir)
    merged = gt.merge(timestamps, on="imgid").sort_values("timestamp")

    # omega_gt/phi_gt/kappa_gt are stored in degrees; convert each row to a
    # quaternion up front so interpolation between rows can slerp instead of
    # linearly blending Euler angles (which is what caused the erratic
    # behavior near angle wraparounds, e.g. kappa_gt swinging through +/-180)
    euler_rad = np.radians(merged[["omega_gt", "phi_gt", "kappa_gt"]].values)

    quats = []
    for e in euler_rad:
        # euler_to_quat(e) gives the world-to-camera rotation; invert it to
        # get camera-to-world, then compose with the fixed camera->body
        # mounting rotation Q_CB so the result is directly comparable to the
        # onboard attitude quaternion (world-from-body), matching how
        # main.py's build_gt_trajectory() derives ground-truth orientation
        q_world_to_camera = euler_to_quat(e)
        q_camera_to_world = quat_conjugate(q_world_to_camera)
        q_body = quat_multiply(q_camera_to_world, Q_CB)
        quats.append(quat_multiply(Q_WORLD_YAW_FIX, q_body))
    quats = np.array(quats)

    return QuaternionSequentialInterpolator(merged["timestamp"].values, quats)


def _load_onboard_orientation_interpolator(log_dir=LOG_DIR):
    pose = pd.read_csv(os.path.join(log_dir, "OnboardPose.csv"))
    pose.columns = [c.strip() for c in pose.columns]
    pose = pose.rename(columns={"Timpstemp": "timestamp"}).sort_values("timestamp")
    quats = pose[["Attitude_w", "Attitude_x", "Attitude_y", "Attitude_z"]].values
    return QuaternionSequentialInterpolator(pose["timestamp"].values, quats)


_gt_interpolator = None
_gps_interpolator = None
_gt_orientation_interpolator = None
_onboard_orientation_interpolator = None


def interpolate_ground_truth_position(t, log_dir=LOG_DIR):
    """Linearly interpolate [x_gt, y_gt, z_gt] at timestamp t. Call with increasing t."""
    global _gt_interpolator
    if _gt_interpolator is None:
        _gt_interpolator = _load_ground_truth_interpolator(log_dir)
    return _gt_interpolator.interpolate(t)


def interpolate_gps_position(t, log_dir=LOG_DIR):
    """Linearly interpolate [easting, northing, alt] (WGS 84 / UTM zone 32N) at timestamp t. Call with increasing t."""
    global _gps_interpolator
    if _gps_interpolator is None:
        _gps_interpolator = _load_gps_interpolator(log_dir)
    return _gps_interpolator.interpolate(t)


def interpolate_ground_truth_orientation(t, log_dir=LOG_DIR):
    """Interpolate ground-truth orientation at timestamp t, returned as a [w, x, y, z] quaternion.

    Each omega_gt/phi_gt/kappa_gt row is first converted to a quaternion,
    then consecutive quaternions are spherically interpolated (slerp) at t.
    Call with increasing t.
    """
    global _gt_orientation_interpolator
    if _gt_orientation_interpolator is None:
        _gt_orientation_interpolator = _load_ground_truth_orientation_interpolator(log_dir)
    return _gt_orientation_interpolator.interpolate(t)


def interpolate_onboard_orientation(t, log_dir=LOG_DIR):
    """Interpolate the onboard/autopilot attitude at timestamp t, returned as a [w, x, y, z] quaternion.

    This is the OnboardPose Attitude_w/x/y/z quaternion, spherically
    interpolated (slerp) between samples. Used only as a benchmark/comparison
    against ground truth, not as part of the estimation pipeline itself.
    Call with increasing t.
    """
    global _onboard_orientation_interpolator
    if _onboard_orientation_interpolator is None:
        _onboard_orientation_interpolator = _load_onboard_orientation_interpolator(log_dir)
    return _onboard_orientation_interpolator.interpolate(t)


def plot_position_comparison(timestamps, gt_positions, gps_positions):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], label="Ground Truth", color="tab:blue")
    ax.plot(gps_positions[:, 0], gps_positions[:, 1], gps_positions[:, 2], label="GPS", color="tab:orange")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_zlabel("Altitude (m)")
    ax.set_title("Drone Position: GPS vs Ground Truth (WGS 84 / UTM zone 32N)")
    ax.legend()
    return fig


def plot_orientation_comparison(timestamps, gt_orientations, onboard_orientations):
    gt_euler_deg = np.degrees(np.array([quat_to_euler(q) for q in gt_orientations]))
    onboard_euler_deg = np.degrees(np.array([quat_to_euler(q) for q in onboard_orientations]))
    labels = ["Omega / Roll (deg)", "Phi / Pitch (deg)", "Kappa / Yaw (deg)"]
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 7))
    for i, ax in enumerate(axes):
        ax.plot(timestamps, gt_euler_deg[:, i], label="Ground Truth", color="tab:blue")
        ax.plot(timestamps, onboard_euler_deg[:, i], label="Onboard (autopilot)", color="tab:orange")
        ax.set_ylabel(labels[i])
        ax.legend()
    axes[-1].set_xlabel("Timestamp")
    fig.suptitle("Drone Orientation: Onboard Autopilot vs Ground Truth")
    return fig


def plot_orientation_vectors_comparison(timestamps, gt_orientations, onboard_orientations):
    # both quaternion series are now in the same world-from-body convention
    # (see the camera->body correction in _load_ground_truth_orientation_interpolator),
    # so a single shared pair of body axes applies to both sources
    body_up = np.array([0.0, 0.0, 1.0])
    body_forward = np.array([1.0, 0.0, 0.0])

    gt_up = np.array([rotate_vector_by_quaternion(q, body_up) for q in gt_orientations])
    gt_forward = np.array([rotate_vector_by_quaternion(q, body_forward) for q in gt_orientations])
    onboard_up = np.array([rotate_vector_by_quaternion(q, body_up) for q in onboard_orientations])
    onboard_forward = np.array([rotate_vector_by_quaternion(q, body_forward) for q in onboard_orientations])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt_up[:, 0], gt_up[:, 1], gt_up[:, 2], linestyle="-", color="tab:blue", label="Ground Truth Up")
    ax.plot(gt_forward[:, 0], gt_forward[:, 1], gt_forward[:, 2], linestyle=":", color="tab:blue",
            label="Ground Truth Forward")
    ax.plot(onboard_up[:, 0], onboard_up[:, 1], onboard_up[:, 2], linestyle="-", color="tab:orange",
            label="Onboard Up")
    ax.plot(onboard_forward[:, 0], onboard_forward[:, 1], onboard_forward[:, 2], linestyle=":", color="tab:orange",
            label="Onboard Forward")

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Drone Orientation: Onboard Autopilot vs Ground Truth (unit vectors)")
    ax.legend()
    return fig


if __name__ == "__main__":
    imgids = np.arange(1, 10002, 30)
    timestamps = _get_ground_truth_timestamps()["timestamp"].values[imgids-1]
    gt_positions = np.array([interpolate_ground_truth_position(t) for t in timestamps])
    gt_orientations = np.array([interpolate_ground_truth_orientation(t) for t in timestamps])
    gps_positions = np.array([interpolate_gps_position(t) for t in timestamps])
    onboard_orientations = np.array([interpolate_onboard_orientation(t) for t in timestamps])
    plot_position_comparison(timestamps, gt_positions, gps_positions)
    plot_orientation_comparison(timestamps, gt_orientations, onboard_orientations)
    plot_orientation_vectors_comparison(timestamps, gt_orientations, onboard_orientations)
    plt.show()
