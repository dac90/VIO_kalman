"""Visual spot-check of stage 7 (marginalization): run propagate + augment +
enforce_sliding_window over a real window long enough that unbounded clone
growth would previously have been impractical (that's exactly why the old
demo_propagate_augment.py capped itself at 5s), and show the window size
and state dimension staying bounded throughout instead of growing forever.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for the top-level modules below

import matplotlib.pyplot as plt
import numpy as np

import ground_truth as gt
from frames import load_T_BS
from imu_propagation import _load_imu_measurements
from msckf_state import MSCKFState, N_CLONE_ERROR, N_IMU_ERROR, augment, enforce_sliding_window, \
    load_imu_noise_params, propagate

DURATION_S = 30.0
MAX_CLONES = 20


def main():
    T_BS_cam0 = load_T_BS(f"{gt.MAV0_DIR}/cam0/sensor.yaml")

    gt_timestamps, gt_positions, gt_quaternions, gt_velocities = gt._load_state_ground_truth()
    cam0_all_timestamps = gt.load_cam0_timestamps()
    start_frame = int(np.searchsorted(cam0_all_timestamps, gt_timestamps[0])) + 30
    window_timestamps = cam0_all_timestamps[start_frame:start_frame + int(DURATION_S * 20)]

    t0 = int(window_timestamps[0])
    p0 = gt.interpolate_ground_truth_position(t0)
    v0 = gt.interpolate_ground_truth_velocity(t0)
    q0 = gt.interpolate_ground_truth_orientation(t0)
    state = MSCKFState.initialize(p0, v0, q0, np.zeros(3), np.zeros(3), np.eye(N_IMU_ERROR) * 1e-4)
    noise_params = load_imu_noise_params()

    imu_timestamps, gyro, accel = _load_imu_measurements()
    imu_idx = np.searchsorted(imu_timestamps, t0, side="right")
    t_prev = t0

    n_clones_history, state_dim_history = [], []
    for frame_idx, frame_t in enumerate(window_timestamps):
        while imu_idx < len(imu_timestamps) and imu_timestamps[imu_idx] <= frame_t:
            t_curr = int(imu_timestamps[imu_idx])
            state = propagate(state, gyro[imu_idx], accel[imu_idx], (t_curr - t_prev) / 1e9, noise_params)
            t_prev = t_curr
            imu_idx += 1

        state = augment(state, T_BS_cam0)
        state = enforce_sliding_window(state, MAX_CLONES)

        assert state.n_clones <= MAX_CLONES
        assert np.allclose(state.P, state.P.T, atol=1e-9)
        n_clones_history.append(state.n_clones)
        state_dim_history.append(state.P.shape[0])

    print(f"Ran {len(window_timestamps)} frames ({DURATION_S:.0f}s) with a {MAX_CLONES}-clone cap.")
    print(f"Final clone count: {state.n_clones}, final state dimension: {state.P.shape[0]}")
    print(f"Peak state dimension: {max(state_dim_history)} "
          f"(would have been {N_IMU_ERROR + N_CLONE_ERROR * len(window_timestamps)} uncapped)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(n_clones_history)
    axes[0].axhline(MAX_CLONES, color="black", linestyle="--", label="cap")
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Active clone count")
    axes[0].set_title("Sliding window size over time")
    axes[0].legend()

    axes[1].plot(state_dim_history, label="With marginalization (this run)")
    uncapped = N_IMU_ERROR + N_CLONE_ERROR * np.arange(1, len(window_timestamps) + 1)
    axes[1].plot(uncapped, "--", color="gray", label="Without marginalization (hypothetical)")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Full state dimension (P size)")
    axes[1].set_title("State dimension: capped vs unbounded growth")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
