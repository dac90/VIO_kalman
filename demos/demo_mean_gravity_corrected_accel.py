"""Estimate a rough constant accelerometer bias, for use as a "dummy" bias
correction in msckf_state.py.

Recipe: at each imu0 sample, rotate the raw accelerometer reading into the
world frame (using ground truth orientation) and add back gravity, exactly
as imu_propagation.propagate_step does with bias assumed zero:
    a_world(t) = R(q_gt(t)) @ accel(t) + GRAVITY_WORLD
If accel had zero bias, a_world(t) would be the vehicle's true dynamic
acceleration, which should average out close to zero over the whole
~3-minute sequence (it starts and ends near the same speed, doing a loop
through the hall). So the mean of a_world(t) is mostly capturing whatever a
constant accelerometer bias contributes, NOT real motion. That mean is
computed in world frame (since gravity only cancels cleanly there), so the
last step rotates it back into the IMU's own frame (using the first in-range
sample's orientation as the reference "IMU frame") to get something directly
usable as bias_accel.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for the top-level modules below

import numpy as np

import ground_truth as gt
from imu_propagation import GRAVITY_WORLD, _load_imu_measurements
from quaternion_utils import quat_conjugate, rotate_vector_by_quaternion


def main():
    gt_timestamps, _, _, _ = gt._load_state_ground_truth()
    imu_timestamps, gyro, accel = _load_imu_measurements()

    # ground truth doesn't cover imu0's first/last ~1s (see ground_truth.py),
    # so only use imu samples where we actually have a ground-truth orientation
    in_range = (imu_timestamps >= gt_timestamps[0]) & (imu_timestamps <= gt_timestamps[-1])
    imu_timestamps, accel = imu_timestamps[in_range], accel[in_range]

    a_world = np.array([
        rotate_vector_by_quaternion(gt.interpolate_ground_truth_orientation(int(t)), a) + GRAVITY_WORLD
        for t, a in zip(imu_timestamps, accel)
    ])
    mean_a_world = a_world.mean(axis=0)

    q_ref = gt.interpolate_ground_truth_orientation(int(imu_timestamps[0]))
    bias_estimate_body = rotate_vector_by_quaternion(quat_conjugate(q_ref), mean_a_world)

    print(f"Samples used: {len(imu_timestamps)}")
    print(f"Raw accel mean (body frame, includes gravity reaction): {accel.mean(axis=0)}")
    print(f"Mean gravity-eliminated acceleration (world frame):     {mean_a_world}")
    print(f"Same, rotated back into the IMU's frame (dummy bias):   {bias_estimate_body}")


if __name__ == "__main__":
    main()
