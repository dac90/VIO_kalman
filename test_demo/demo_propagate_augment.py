"""End-to-end demo: run the MSCKF's propagate+augment loop (stages 1+2) over
real MH_01_easy data.

No measurement update or marginalization yet (those are later stages), so
this only demonstrates nominal IMU dead reckoning, error-covariance growth,
and camera-clone augmentation with its covariance coupling. Kept in its own
file, separate from ground_truth.py/msckf_state.py, so it can be run and
tweaked freely without touching the modules it exercises.

Since clones currently accumulate without bound (marginalization is a later
stage), this only runs a short window (a few seconds) instead of the full
~3-minute sequence -- otherwise the covariance matrix would grow to tens of
thousands of dimensions and every propagate/augment call would become
extremely slow.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for the top-level modules below

import matplotlib.pyplot as plt
import numpy as np

import ground_truth as gt
from frames import body_to_sensor_position, load_T_BS
from imu_propagation import _load_imu_measurements
from msckf_state import MSCKFState, augment, load_imu_noise_params, propagate

DEMO_DURATION_S = 5.0

# "Dummy" bias correction: the mean gyro reading over the whole sequence (see
# plot.py), used here as a fixed initial gyro-bias estimate. Not a real
# calibration/estimation procedure -- just something to compare against
# assuming zero bias, until the filter can estimate bias from measurements.
DUMMY_BIAS_GYRO = np.array([-0.00233173, 0.02172386, 0.07821335])
DUMMY_BIAS_ACCEL = np.array([-0.04066623, 0.1155297, 0.05121861])

# Initial uncertainty: tight on position/orientation (we initialize from
# ground truth), looser on velocity, and bias uncertainty in line with
# typical MEMS IMU bias magnitudes. Placeholders -- not yet tuned against an
# actual measurement update, since there isn't one yet.
INITIAL_SIGMA = dict(theta=1e-3, b_g=1e-2, v=1e-1, b_a=1e-1, p=1e-3)


def initial_covariance():
    diag = (
        [INITIAL_SIGMA["theta"] ** 2] * 3
        + [INITIAL_SIGMA["b_g"] ** 2] * 3
        + [INITIAL_SIGMA["v"] ** 2] * 3
        + [INITIAL_SIGMA["b_a"] ** 2] * 3
        + [INITIAL_SIGMA["p"] ** 2] * 3
    )
    return np.diag(diag)


def run(bias_gyro, bias_accel, label):
    gt_timestamps, gt_positions, gt_quaternions, gt_velocities = gt._load_state_ground_truth()
    t0 = int(gt_timestamps[0])
    t_end = t0 + int(DEMO_DURATION_S * 1e9)

    cam0_timestamps = gt.load_cam0_timestamps()
    cam0_timestamps = cam0_timestamps[(cam0_timestamps >= t0) & (cam0_timestamps <= t_end)]

    imu_timestamps, gyro, accel = _load_imu_measurements()
    start_idx = np.searchsorted(imu_timestamps, t0, side="right")
    end_idx = np.searchsorted(imu_timestamps, t_end, side="right")

    p0, q0, v0 = gt_positions[0], gt_quaternions[0], gt_velocities[0]
    state = MSCKFState.initialize(p0, v0, q0, bias_gyro, bias_accel, initial_covariance())
    noise_params = load_imu_noise_params()
    T_BS_cam0 = load_T_BS(f"{gt.MAV0_DIR}/cam0/sensor.yaml")

    clone_timestamps = []
    position_sigma_history = []

    cam0_idx = 0
    t_prev = t0
    for i in range(start_idx, end_idx):
        t_curr = int(imu_timestamps[i])
        dt = (t_curr - t_prev) / 1e9
        state = propagate(state, gyro[i], accel[i], dt, noise_params)
        t_prev = t_curr

        while cam0_idx < len(cam0_timestamps) and cam0_timestamps[cam0_idx] <= t_curr:
            state = augment(state, T_BS_cam0)
            clone_timestamps.append(int(cam0_timestamps[cam0_idx]))
            cam0_idx += 1

        position_sigma_history.append(np.sqrt(np.diag(state.P)[12:15]))

    position_sigma_history = np.array(position_sigma_history)

    print(f"[{label}] Ran {end_idx - start_idx} IMU steps over {DEMO_DURATION_S:.1f}s, "
          f"produced {state.n_clones} camera clones.")
    print(f"[{label}] Final IMU position std dev (m): {position_sigma_history[-1]}")

    # ground-truth camera position at each clone's timestamp, for comparison
    # (interpolators require increasing t -- clone_timestamps is already sorted)
    gt_clone_positions = np.array([
        body_to_sensor_position(gt.interpolate_ground_truth_position(t),
                                 gt.interpolate_ground_truth_orientation(t),
                                 T_BS_cam0)
        for t in clone_timestamps
    ])
    clone_positions = np.array(state.clone_positions)

    position_error = np.linalg.norm(clone_positions - gt_clone_positions, axis=1)
    print(f"[{label}] Clone position error vs ground truth: first={position_error[0]:.4f}m, "
          f"last={position_error[-1]:.4f}m")

    return gt_clone_positions, clone_positions, position_sigma_history


def main():
    gt_clone_positions, zero_bias_positions, zero_bias_sigma = run(
        np.zeros(3), np.zeros(3), label="zero bias")
    _, dummy_bias_positions, dummy_bias_sigma = run(
        DUMMY_BIAS_GYRO, DUMMY_BIAS_ACCEL, label="dummy bias")

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt_clone_positions[:, 0], gt_clone_positions[:, 1], gt_clone_positions[:, 2],
            label="Ground truth cam0", color="tab:blue")
    ax.plot(zero_bias_positions[:, 0], zero_bias_positions[:, 1], zero_bias_positions[:, 2],
            label="Zero bias (dead reckoning)", color="tab:green")
    ax.plot(dummy_bias_positions[:, 0], dummy_bias_positions[:, 1], dummy_bias_positions[:, 2],
            label="Dummy bias (dead reckoning)", color="tab:red")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Stage 1+2 demo: propagate+augment over {DEMO_DURATION_S:.0f}s")
    ax.legend()

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    axes2[0].plot(zero_bias_sigma)
    axes2[0].set_title("Zero bias")
    axes2[1].plot(dummy_bias_sigma)
    axes2[1].set_title("Dummy bias")
    for ax2 in axes2:
        ax2.set_xlabel("IMU step")
        ax2.legend(["x", "y", "z"])
    axes2[0].set_ylabel("Position std dev (m)")
    fig2.suptitle("IMU position uncertainty growth (1-sigma)")

    plt.show()


if __name__ == "__main__":
    main()
