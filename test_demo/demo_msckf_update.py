"""Visual spot-check of stage 6 on a real mini end-to-end pipeline: dead-reckon
a short real window (with drift), track and triangulate real features using
the filter's own (drifted) pose estimates, chi-square gate them, run one EKF
update, and compare camera-pose error against ground truth before vs after.

Also includes one deliberately corrupted feature to show the gate rejecting
it, alongside a histogram of every feature's Mahalanobis distance against the
chi-square threshold.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for top-level imports

import matplotlib.pyplot as plt
import numpy as np

import ground_truth as gt
from feature_tracker import FeatureTracker, iter_cam0_frames
from frames import body_to_sensor_position, load_T_BS
from measurement_model import null_space_project, stack_feature_observations
from msckf_state import MSCKFState, augment, load_imu_noise_params, propagate
from msckf_update import chi_square_threshold, ekf_update, mahalanobis_distance, passes_chi_square_gate
from imu_propagation import _load_imu_measurements
from triangulation import load_cam0_intrinsics, triangulate_feature, undistort_normalized

N_FRAMES = 40  # 2s at 20Hz
MIN_TRACK_LENGTH = 15
OBSERVATION_NOISE_STD = 1.0 / 458.0  # ~1 pixel, in normalized (bearing) units at cam0's focal length

# Same "dummy bias" from demo_propagate_augment.py: without it, zero-bias
# dead reckoning's *real* error badly outpaces what the filter's own P
# reports (P underestimates the truth, as that demo showed directly), so
# nearly every honest measurement would look like a chi-square outlier
# relative to an overconfident covariance. With it, error and P stay
# reasonably matched, which is what the gate actually needs to be meaningful.
DUMMY_BIAS_GYRO = np.array([-0.00233173, 0.02172386, 0.07821335])
DUMMY_BIAS_ACCEL = np.array([-0.04066623, 0.1155297, 0.05121861])

# tight initial uncertainty (we initialize from ground truth)
INITIAL_SIGMA = dict(theta=1e-3, b_g=1e-2, v=1e-1, b_a=1e-1, p=1e-3)


def initial_covariance():
    diag = ([INITIAL_SIGMA["theta"] ** 2] * 3 + [INITIAL_SIGMA["b_g"] ** 2] * 3 + [INITIAL_SIGMA["v"] ** 2] * 3
            + [INITIAL_SIGMA["b_a"] ** 2] * 3 + [INITIAL_SIGMA["p"] ** 2] * 3)
    return np.diag(diag)


def main():
    K, dist_coeffs = load_cam0_intrinsics()
    T_BS_cam0 = load_T_BS(f"{gt.MAV0_DIR}/cam0/sensor.yaml")

    gt_timestamps, gt_positions, gt_quaternions, gt_velocities = gt._load_state_ground_truth()
    cam0_all_timestamps = gt.load_cam0_timestamps()
    start_frame = int(np.searchsorted(cam0_all_timestamps, gt_timestamps[0])) + 30

    frames = list(iter_cam0_frames(max_frames=N_FRAMES, start_frame=start_frame))
    window_timestamps = [t for t, _ in frames]
    clone_index_of = {t: i for i, t in enumerate(window_timestamps)}

    # --- dead-reckon the IMU through this window, augmenting a clone at every frame ---
    t0 = window_timestamps[0]
    p0 = gt.interpolate_ground_truth_position(t0)
    v0 = gt.interpolate_ground_truth_velocity(t0)
    q0 = gt.interpolate_ground_truth_orientation(t0)
    state = MSCKFState.initialize(p0, v0, q0, DUMMY_BIAS_GYRO, DUMMY_BIAS_ACCEL, initial_covariance())
    noise_params = load_imu_noise_params()

    imu_timestamps, gyro, accel = _load_imu_measurements()
    imu_idx = np.searchsorted(imu_timestamps, t0, side="right")
    t_prev = t0
    frame_idx = 0
    while frame_idx < len(window_timestamps):
        if imu_idx < len(imu_timestamps) and imu_timestamps[imu_idx] <= window_timestamps[frame_idx]:
            t_curr = int(imu_timestamps[imu_idx])
            dt = (t_curr - t_prev) / 1e9
            state = propagate(state, gyro[imu_idx], accel[imu_idx], dt, noise_params)
            t_prev = t_curr
            imu_idx += 1
        else:
            state = augment(state, T_BS_cam0)
            frame_idx += 1

    # --- track real features over the same window ---
    tracker = FeatureTracker(max_features=150)
    for timestamp, image in frames:
        tracker.process_frame(image, timestamp)

    def clone_pose_error(i):
        gt_p = body_to_sensor_position(gt.interpolate_ground_truth_position(window_timestamps[i]),
                                        gt.interpolate_ground_truth_orientation(window_timestamps[i]), T_BS_cam0)
        return np.linalg.norm(state.clone_positions[i] - gt_p)

    print(f"Dead-reckoned {N_FRAMES} frames ({N_FRAMES / 20:.1f}s). "
          f"Camera position error before update: first={clone_pose_error(0):.4f}m, "
          f"last={clone_pose_error(state.n_clones - 1):.4f}m")

    # --- build (r_o, H_o) for every sufficiently-long track, using the filter's own drifted poses ---
    camera_poses_by_clone = list(zip(state.clone_orientations, state.clone_positions))
    long_tracks = [tid for tid, obs in tracker.tracks.items() if len(obs) >= MIN_TRACK_LENGTH]
    print(f"{len(long_tracks)} tracks with >= {MIN_TRACK_LENGTH} observations in this window.")

    gammas, thresholds, accepted = [], [], []
    r_o_accepted, H_o_accepted = [], []
    for k, track_id in enumerate(long_tracks):
        observations = tracker.tracks[track_id]
        clone_indices = [clone_index_of[t] for t, _ in observations]
        camera_poses = [camera_poses_by_clone[i] for i in clone_indices]
        bearings = [undistort_normalized([[u, v]], K, dist_coeffs)[0] for _, (u, v) in observations]

        if k == 0:
            bearings[len(bearings) // 2] = bearings[len(bearings) // 2] + np.array([0.3, -0.3])  # corrupt one feature

        X_est = triangulate_feature(camera_poses, bearings)
        r, H_x, H_f = stack_feature_observations(X_est, camera_poses, bearings, clone_indices,
                                                  n_clones=state.n_clones)
        r_o, H_o = null_space_project(r, H_x, H_f)

        gamma = mahalanobis_distance(r_o, H_o, state.P, OBSERVATION_NOISE_STD)
        threshold = chi_square_threshold(len(r_o))
        gammas.append(gamma)
        thresholds.append(threshold)
        ok = gamma <= threshold
        accepted.append(ok)
        if ok:
            r_o_accepted.append(r_o)
            H_o_accepted.append(H_o)

    print(f"Chi-square gate: accepted {sum(accepted)}/{len(long_tracks)} tracks "
          f"(feature 0 was deliberately corrupted -> accepted={accepted[0]})")

    positions_before = [p.copy() for p in state.clone_positions]
    if r_o_accepted:
        state = ekf_update(state, np.concatenate(r_o_accepted), np.concatenate(H_o_accepted, axis=0),
                            OBSERVATION_NOISE_STD)

    errors_before = [np.linalg.norm(positions_before[i] - body_to_sensor_position(
        gt.interpolate_ground_truth_position(window_timestamps[i]),
        gt.interpolate_ground_truth_orientation(window_timestamps[i]), T_BS_cam0)) for i in range(state.n_clones)]
    errors_after = [clone_pose_error(i) for i in range(state.n_clones)]

    print(f"Mean camera position error: before={np.mean(errors_before):.4f}m, "
          f"after={np.mean(errors_after):.4f}m")

    gt_positions_window = np.array([body_to_sensor_position(gt.interpolate_ground_truth_position(t),
                                                             gt.interpolate_ground_truth_orientation(t), T_BS_cam0)
                                     for t in window_timestamps])

    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(121, projection="3d")
    ax.plot(*gt_positions_window.T, color="tab:blue", label="Ground truth")
    ax.plot(*np.array(positions_before).T, color="tab:red", label="Before update (dead reckoning)")
    ax.plot(*np.array(state.clone_positions).T, color="tab:green", label="After update")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title("Camera trajectory: before vs after EKF update")
    ax.legend(fontsize=8)

    ax2 = fig.add_subplot(122)
    ax2.bar(range(len(gammas)), gammas, color=["tab:green" if a else "tab:red" for a in accepted])
    ax2.axhline(thresholds[0], color="black", linestyle="--", label="chi-square threshold")
    ax2.set_yscale("log")  # gross outliers can be many orders of magnitude past the threshold
    ax2.set_xlabel("Track index (index 0 deliberately corrupted)")
    ax2.set_ylabel("Mahalanobis distance")
    ax2.set_title("Chi-square gate: accepted (green) vs rejected (red)")
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
