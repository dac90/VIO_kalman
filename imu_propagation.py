"""Nominal-state IMU strapdown propagation (dead reckoning) for MH_01_easy.

Assumes zero gyro/accel bias; corrects for gravity. This is the raw,
uncorrected trajectory an IMU-only integrator produces -- it has no way to
fuse camera measurements yet, so it is expected to drift away from ground
truth steadily (that drift is exactly what the MSCKF update will later keep
in check).
"""
import os

import numpy as np
import pandas as pd

from quaternion_utils import axis_angle_to_quat, quat_multiply, rotate_vector_by_quaternion

MAV0_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "machine_hall", "MH_01_easy", "MH_01_easy", "mav0")

GYRO_COLS = ["w_RS_S_x [rad s^-1]", "w_RS_S_y [rad s^-1]", "w_RS_S_z [rad s^-1]"]
ACCEL_COLS = ["a_RS_S_x [m s^-2]", "a_RS_S_y [m s^-2]", "a_RS_S_z [m s^-2]"]

# World frame is Z-up (see ground_truth.py's gravity-direction analysis: the
# accelerometer's specific force, rotated into world frame by the ground-truth
# orientation, averages to ~[0, 0, +9.8]).
GRAVITY_WORLD = np.array([0.0, 0.0, -9.81])


def _load_imu_measurements(mav0_dir=MAV0_DIR):
    path = os.path.join(mav0_dir, "imu0", "data.csv")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.sort_values(df.columns[0]).reset_index(drop=True)
    timestamps = df[df.columns[0]].values
    gyro = df[GYRO_COLS].to_numpy()
    accel = df[ACCEL_COLS].to_numpy()
    return timestamps, gyro, accel


def propagate_step(p, v, q, gyro, accel, dt, bias_gyro=np.zeros(3), bias_accel=np.zeros(3)):
    """One discrete strapdown INS step over dt (s). Returns (p_new, v_new, q_new).

    Orientation is integrated with the exact constant-angular-velocity
    quaternion exponential map (axis_angle_to_quat); velocity/position assume
    the world-frame specific force is constant (its value at the start of the
    interval) over dt.
    """
    omega = gyro - bias_gyro
    f = accel - bias_accel

    q_new = quat_multiply(q, axis_angle_to_quat(omega * dt))
    q_new = q_new / np.linalg.norm(q_new)

    a_world = rotate_vector_by_quaternion(q, f) + GRAVITY_WORLD
    v_new = v + a_world * dt
    p_new = p + v * dt + 0.5 * a_world * dt ** 2

    return p_new, v_new, q_new


def propagate_imu_trajectory(p0, v0, q0, t0, mav0_dir=MAV0_DIR, bias_gyro=np.zeros(3), bias_accel=np.zeros(3)):
    """Dead-reckon imu0 measurements from timestamp t0 (ns) onward.

    (p0, v0, q0) is the known state at t0; t0 need not be an exact imu0
    sample timestamp (the first interval is integrated over the partial dt up
    to the next real sample). bias_gyro/bias_accel are held fixed (constant)
    over the whole run, applied at every step exactly as propagate_step does.

    Returns (timestamps, positions, velocities, quaternions), one entry per
    imu0 sample at/after t0, with the first entry equal to (t0, p0, v0, q0).
    """
    imu_timestamps, gyro, accel = _load_imu_measurements(mav0_dir)
    start_idx = np.searchsorted(imu_timestamps, t0, side="right")

    timestamps = np.concatenate(([t0], imu_timestamps[start_idx:]))
    n = len(timestamps)

    positions = np.empty((n, 3))
    velocities = np.empty((n, 3))
    quaternions = np.empty((n, 4))
    positions[0], velocities[0], quaternions[0] = p0, v0, q0

    p, v, q = p0, v0, q0
    for i in range(1, n):
        dt = (timestamps[i] - timestamps[i - 1]) / 1e9
        # zero-order-hold: use the measurement sampled at (or just before) the
        # start of this interval, i.e. index start_idx + i - 2 (clamped to 0
        # for the case t0 precedes every imu0 sample)
        meas_idx = max(start_idx + i - 2, 0)
        p, v, q = propagate_step(p, v, q, gyro[meas_idx], accel[meas_idx], dt, bias_gyro, bias_accel)
        positions[i], velocities[i], quaternions[i] = p, v, q

    return timestamps, positions, velocities, quaternions
