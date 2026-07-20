"""Ground-truth interpolation and plotting for the EuRoC MH_01_easy dataset.

Mirrors the interpolation/plotting approach in main.py (built for the
AGZ_subset PX4-log dataset), applied to EuRoC's mav0 layout:
state_groundtruth_estimate0/data.csv holds the batch-optimized reference
trajectory (position + orientation, already world-from-body), and
leica0/data.csv holds the raw external position measurements it was derived
from.

All plotting/evaluation here is done at cam0's image timestamps (see
load_cam0_timestamps): the eventual MSCKF only augments its state once per
image, so evaluating ground truth at those same timesteps keeps every
comparison aligned with what the filter will actually process.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from frames import body_to_sensor_quaternion, load_T_BS
from interpolation import QuaternionSequentialInterpolator, SequentialInterpolator
from quaternion_utils import quat_to_euler, rotate_vector_by_quaternion

MAV0_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "machine_hall", "MH_01_easy", "MH_01_easy", "mav0")

# EuRoC's IMU/body frame is NOT aerospace convention (X=forward, Z=down).
# Averaging accelerometer specific force in body frame (dominant reaction to
# gravity) gives ~[9.1, -0.1, -3.4], i.e. body +X is "up"; cam0's T_BS extrinsic
# maps the camera's forward optical axis onto body +Z. Body +Y is the
# remaining (lateral) axis. This fixed remap (expressed as a T_BS-shaped
# transform so it can go through the same frames.py machinery as a real
# sensor.yaml extrinsic) re-expresses ground-truth orientation in a
# conventional aerospace body frame (X=forward, Y=right, Z=down), purely for
# human-readable roll/pitch/yaw plotting.
_AERO_BODY_T_BS = np.array([
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])

# Timestamps are read as exact int64 nanoseconds (see interpolation.py), so no
# fuzzy snapping is needed: an exact-timestamp query lands with alpha=0 or 1
# through the ordinary linear/slerp formula anyway.
EXACT_MATCH_TOLERANCE_NS = 0


def _load_state_ground_truth(mav0_dir=MAV0_DIR):
    path = os.path.join(mav0_dir, "state_groundtruth_estimate0", "data.csv")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.sort_values(df.columns[0])
    timestamps = df[df.columns[0]].values
    positions = df[["p_RS_R_x [m]", "p_RS_R_y [m]", "p_RS_R_z [m]"]].values
    quaternions = df[["q_RS_w []", "q_RS_x []", "q_RS_y []", "q_RS_z []"]].values
    velocities = df[["v_RS_R_x [m s^-1]", "v_RS_R_y [m s^-1]", "v_RS_R_z [m s^-1]"]].values
    return timestamps, positions, quaternions, velocities


def _load_leica_position(mav0_dir=MAV0_DIR):
    path = os.path.join(mav0_dir, "leica0", "data.csv")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.sort_values(df.columns[0])
    timestamps = df[df.columns[0]].values
    positions = df[["p_RS_R_x [m]", "p_RS_R_y [m]", "p_RS_R_z [m]"]].values
    return timestamps, positions


def load_cam0_timestamps(mav0_dir=MAV0_DIR):
    """Sorted int64 cam0 image timestamps (ns) from cam0/data.csv.

    This is the master clock the rest of the pipeline synchronizes to: state
    augmentation happens once per image, so evaluating ground truth at cam0's
    timestamps means every comparison lines up with the exact timesteps the
    filter itself will process.
    """
    path = os.path.join(mav0_dir, "cam0", "data.csv")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return np.sort(df[df.columns[0]].values)


_gt_position_interpolator = None
_gt_orientation_interpolator = None
_gt_velocity_interpolator = None
_leica_position_interpolator = None
_cam0_T_BS = None


def interpolate_ground_truth_position(t, mav0_dir=MAV0_DIR):
    """Linearly interpolate [x, y, z] ground-truth position (m) at timestamp t (ns). Call with increasing t."""
    global _gt_position_interpolator
    if _gt_position_interpolator is None:
        timestamps, positions, _, _ = _load_state_ground_truth(mav0_dir)
        _gt_position_interpolator = SequentialInterpolator(
            timestamps, positions, exact_match_tolerance=EXACT_MATCH_TOLERANCE_NS)
    return _gt_position_interpolator.interpolate(t)


def interpolate_ground_truth_orientation(t, mav0_dir=MAV0_DIR):
    """Slerp-interpolate the ground-truth world-from-body quaternion [w,x,y,z] at timestamp t (ns). Call with increasing t."""
    global _gt_orientation_interpolator
    if _gt_orientation_interpolator is None:
        timestamps, _, quaternions, _ = _load_state_ground_truth(mav0_dir)
        _gt_orientation_interpolator = QuaternionSequentialInterpolator(
            timestamps, quaternions, exact_match_tolerance=EXACT_MATCH_TOLERANCE_NS)
    return _gt_orientation_interpolator.interpolate(t)


def interpolate_ground_truth_velocity(t, mav0_dir=MAV0_DIR):
    """Linearly interpolate [vx, vy, vz] ground-truth world-frame velocity (m/s) at timestamp t (ns). Call with increasing t."""
    global _gt_velocity_interpolator
    if _gt_velocity_interpolator is None:
        timestamps, _, _, velocities = _load_state_ground_truth(mav0_dir)
        _gt_velocity_interpolator = SequentialInterpolator(
            timestamps, velocities, exact_match_tolerance=EXACT_MATCH_TOLERANCE_NS)
    return _gt_velocity_interpolator.interpolate(t)


def interpolate_leica_position(t, mav0_dir=MAV0_DIR):
    """Linearly interpolate [x, y, z] raw Leica position (m) at timestamp t (ns). Call with increasing t."""
    global _leica_position_interpolator
    if _leica_position_interpolator is None:
        timestamps, positions = _load_leica_position(mav0_dir)
        _leica_position_interpolator = SequentialInterpolator(
            timestamps, positions, exact_match_tolerance=EXACT_MATCH_TOLERANCE_NS)
    return _leica_position_interpolator.interpolate(t)


def interpolate_cam0_orientation(t, mav0_dir=MAV0_DIR):
    """World-from-cam0 orientation quaternion [w,x,y,z] at timestamp t (ns).

    Derived from the interpolated ground-truth body orientation and cam0's
    own T_BS extrinsic (loaded from its sensor.yaml), via frames.py's
    body-to-sensor conversion. Call with increasing t.
    """
    global _cam0_T_BS
    if _cam0_T_BS is None:
        _cam0_T_BS = load_T_BS(os.path.join(mav0_dir, "cam0", "sensor.yaml"))
    q_world_body = interpolate_ground_truth_orientation(t, mav0_dir)
    return body_to_sensor_quaternion(q_world_body, _cam0_T_BS)


def plot_position_comparison(gt_positions, leica_positions, imu_positions):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], label="Ground Truth", color="tab:blue")
    ax.plot(leica_positions[:, 0], leica_positions[:, 1], leica_positions[:, 2], label="Leica (interpolated)", color="tab:orange")
    ax.plot(imu_positions[:, 0], imu_positions[:, 1], imu_positions[:, 2], label="IMU (dead reckoning)", color="tab:green")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("MH_01_easy Position: Leica vs Ground Truth vs IMU (at cam0 timestamps)")
    ax.legend()
    return fig


def plot_orientation_comparison(timestamps, gt_orientations, imu_orientations):
    gt_aero_orientations = np.array([body_to_sensor_quaternion(q, _AERO_BODY_T_BS) for q in gt_orientations])
    imu_aero_orientations = np.array([body_to_sensor_quaternion(q, _AERO_BODY_T_BS) for q in imu_orientations])
    gt_euler_rad = np.array([quat_to_euler(q) for q in gt_aero_orientations])
    imu_euler_rad = np.array([quat_to_euler(q) for q in imu_aero_orientations])
    # roll/yaw sit close to the world frame's (arbitrary) heading-zero reference
    # for long stretches of this sequence, so unwrap each angle's +/-180 wraps
    # into a continuous curve instead of a discontinuity-driven flicker
    gt_euler_deg = np.degrees(np.unwrap(gt_euler_rad, axis=0))
    imu_euler_deg = np.degrees(np.unwrap(imu_euler_rad, axis=0))
    labels = ["Roll (deg)", "Pitch (deg)", "Yaw (deg)"]
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 7))
    for i, ax in enumerate(axes):
        ax.plot(timestamps, gt_euler_deg[:, i], label="Ground Truth", color="tab:blue")
        ax.plot(timestamps, imu_euler_deg[:, i], label="IMU (dead reckoning)", color="tab:green")
        ax.set_ylabel(labels[i])
        ax.legend()
    axes[-1].set_xlabel("Timestamp (ns)")
    fig.suptitle("MH_01_easy Orientation: Ground Truth vs IMU")
    return fig


def plot_orientation_vectors_comparison(gt_orientations, imu_orientations):
    # native IMU/body axes: +X is up, +Z is forward (see _AERO_BODY_T_BS comment above)
    body_up = np.array([1.0, 0.0, 0.0])
    body_forward = np.array([0.0, 0.0, 1.0])

    gt_up = np.array([rotate_vector_by_quaternion(q, body_up) for q in gt_orientations])
    gt_forward = np.array([rotate_vector_by_quaternion(q, body_forward) for q in gt_orientations])
    imu_up = np.array([rotate_vector_by_quaternion(q, body_up) for q in imu_orientations])
    imu_forward = np.array([rotate_vector_by_quaternion(q, body_forward) for q in imu_orientations])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt_up[:, 0], gt_up[:, 1], gt_up[:, 2], linestyle="-", color="tab:blue", label="Ground Truth Up")
    ax.plot(gt_forward[:, 0], gt_forward[:, 1], gt_forward[:, 2], linestyle=":", color="tab:blue",
            label="Ground Truth Forward")
    ax.plot(imu_up[:, 0], imu_up[:, 1], imu_up[:, 2], linestyle="-", color="tab:green", label="IMU Up")
    ax.plot(imu_forward[:, 0], imu_forward[:, 1], imu_forward[:, 2], linestyle=":", color="tab:green",
            label="IMU Forward")

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("MH_01_easy Orientation (unit vectors): Ground Truth vs IMU")
    ax.legend()
    return fig


if __name__ == "__main__":
    from imu_propagation import propagate_imu_trajectory

    gt_timestamps, gt_raw_positions, gt_raw_quaternions, gt_raw_velocities = _load_state_ground_truth()

    # ground truth doesn't cover cam0's first/last ~1s (its own logging starts
    # late and ends early relative to imu0/cam0), so restrict to cam0
    # timestamps actually within ground truth's range to avoid comparing
    # against a clamped/extrapolated ground-truth value at either end
    timestamps = load_cam0_timestamps()
    timestamps = timestamps[(timestamps >= gt_timestamps[0]) & (timestamps <= gt_timestamps[-1])]

    gt_positions = np.array([interpolate_ground_truth_position(t) for t in timestamps])
    gt_orientations = np.array([interpolate_ground_truth_orientation(t) for t in timestamps])
    leica_positions = np.array([interpolate_leica_position(t) for t in timestamps])

    q_world_cam0 = interpolate_cam0_orientation(int(timestamps[0]))
    print("cam0 world-frame orientation at first timestamp [w,x,y,z]:", q_world_cam0)

    # dead-reckon the IMU from ground truth's exact first state (zero bias
    # assumed), then resample the propagated trajectory onto the same cam0
    # timestamps as everything else, for a fair comparison
    p0, q0, v0 = gt_raw_positions[0], gt_raw_quaternions[0], gt_raw_velocities[0]
    imu_timestamps, imu_positions_raw, _, imu_quaternions_raw = propagate_imu_trajectory(
        p0, v0, q0, int(gt_timestamps[0]))
    imu_position_interp = SequentialInterpolator(imu_timestamps, imu_positions_raw,
                                                  exact_match_tolerance=EXACT_MATCH_TOLERANCE_NS)
    imu_orientation_interp = QuaternionSequentialInterpolator(imu_timestamps, imu_quaternions_raw,
                                                               exact_match_tolerance=EXACT_MATCH_TOLERANCE_NS)
    imu_positions = np.array([imu_position_interp.interpolate(t) for t in timestamps])
    imu_orientations = np.array([imu_orientation_interp.interpolate(t) for t in timestamps])

    plot_position_comparison(gt_positions, leica_positions, imu_positions)
    plot_orientation_comparison(timestamps, gt_orientations, imu_orientations)
    plot_orientation_vectors_comparison(gt_orientations, imu_orientations)
    plt.show()
