"""Visual check that the filter actually improves its own IMU bias estimate
from real feature observations, as discussed: starting from a poor (zero)
bias guess with loose bias covariance, does the online EKF pull b_g/b_a
toward a sensible value using nothing but propagate/augment/track/
triangulate/gate/update/marginalize -- no bias value is ever injected here.

Runs a longer real window than the single-update stage 6 demo, periodically
batching newly-accumulated observations (from both ended and still-ongoing
tracks) into EKF updates so bias actually gets enough information to move,
then marginalizes down to a bounded window (stage 7) so this stays cheap.

The comparison line isn't ground truth (there is none for a real IMU's
bias) -- it's demo_mean_gravity_corrected_accel.py's estimate, which is
derived by an entirely different method (averaging gravity-corrected
acceleration over the whole ~3 minute sequence). Agreement between the two
independent estimates is a meaningful sanity check either way.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for top-level imports

import matplotlib.pyplot as plt
import numpy as np

import ground_truth as gt
from feature_tracker import FeatureTracker, iter_cam0_frames
from frames import load_T_BS
from imu_propagation import _load_imu_measurements
from measurement_model import null_space_project, stack_feature_observations
from msckf_state import MSCKFState, N_IMU_ERROR, augment, enforce_sliding_window, load_imu_noise_params, propagate
from msckf_update import ekf_update, passes_chi_square_gate
from triangulation import load_cam0_intrinsics, triangulate_feature, undistort_normalized

DURATION_S = 20.0
MAX_CLONES = 20
PROCESS_INTERVAL_FRAMES = 10  # batch newly-accumulated observations every 0.5s
MIN_NEW_OBSERVATIONS = 6
OBSERVATION_NOISE_STD = 1.0 / 458.0  # ~1 pixel, in normalized (bearing) units at cam0's focal length

# independent reference from demo_mean_gravity_corrected_accel.py -- NOT ground truth, just a sanity check
REFERENCE_BIAS_GYRO = np.array([-0.00233173, 0.02172386, 0.07821335])
REFERENCE_BIAS_ACCEL = np.array([-0.04066623, 0.1155297, 0.05121861])

# poor initial guess (zero) but loose covariance, so the filter is actually
# free to move away from it -- this is the whole point of the demo
INITIAL_SIGMA = dict(theta=1e-3, b_g=0.05, v=1e-1, b_a=0.2, p=1e-3)


def initial_covariance():
    diag = ([INITIAL_SIGMA["theta"] ** 2] * 3 + [INITIAL_SIGMA["b_g"] ** 2] * 3 + [INITIAL_SIGMA["v"] ** 2] * 3
            + [INITIAL_SIGMA["b_a"] ** 2] * 3 + [INITIAL_SIGMA["p"] ** 2] * 3)
    return np.diag(diag)


def _process_new_observations(state, tracker, clone_frame_ids, n_used_observations):
    """Batch every track's newly-accumulated (not-yet-used) observations into one EKF update."""
    clone_index_of = {t: i for i, t in enumerate(clone_frame_ids)}
    K, dist_coeffs = load_cam0_intrinsics()

    r_o_batch, H_o_batch = [], []
    for track_id, observations in tracker.tracks.items():
        already_used = n_used_observations.get(track_id, 0)
        new_obs = observations[already_used:]
        n_used_observations[track_id] = len(observations)  # mark consumed either way, to never reuse/double-count

        usable = [(t, uv) for t, uv in new_obs if t in clone_index_of]
        if len(usable) < MIN_NEW_OBSERVATIONS:
            continue

        clone_indices = [clone_index_of[t] for t, _ in usable]
        camera_poses = [(state.clone_orientations[i], state.clone_positions[i]) for i in clone_indices]
        bearings = [undistort_normalized([[u, v]], K, dist_coeffs)[0] for _, (u, v) in usable]

        try:
            X_est = triangulate_feature(camera_poses, bearings)
            r, H_x, H_f = stack_feature_observations(X_est, camera_poses, bearings, clone_indices,
                                                      n_clones=state.n_clones)
            r_o, H_o = null_space_project(r, H_x, H_f)
        except np.linalg.LinAlgError:
            continue  # degenerate geometry (e.g. near-zero parallax): skip, no parallax check yet (deferred)

        if passes_chi_square_gate(r_o, H_o, state.P, OBSERVATION_NOISE_STD):
            r_o_batch.append(r_o)
            H_o_batch.append(H_o)

    if r_o_batch:
        state = ekf_update(state, np.concatenate(r_o_batch), np.concatenate(H_o_batch, axis=0),
                            OBSERVATION_NOISE_STD)
    return state, len(r_o_batch)


def main():
    T_BS_cam0 = load_T_BS(f"{gt.MAV0_DIR}/cam0/sensor.yaml")

    gt_timestamps, _, _, _ = gt._load_state_ground_truth()
    cam0_all_timestamps = gt.load_cam0_timestamps()
    start_frame = int(np.searchsorted(cam0_all_timestamps, gt_timestamps[0])) + 30
    n_frames = int(DURATION_S * 20)

    frames = list(iter_cam0_frames(max_frames=n_frames, start_frame=start_frame))
    window_timestamps = [t for t, _ in frames]

    t0 = window_timestamps[0]
    p0 = gt.interpolate_ground_truth_position(t0)
    v0 = gt.interpolate_ground_truth_velocity(t0)
    q0 = gt.interpolate_ground_truth_orientation(t0)
    state = MSCKFState.initialize(p0, v0, q0, np.zeros(3), np.zeros(3), initial_covariance())
    noise_params = load_imu_noise_params()

    imu_timestamps, gyro, accel = _load_imu_measurements()
    imu_idx = np.searchsorted(imu_timestamps, t0, side="right")
    t_prev = t0

    tracker = FeatureTracker(max_features=150)
    clone_frame_ids = []      # clone_frame_ids[i] = timestamp of state's i-th clone
    n_used_observations = {}  # track_id -> how many of its observations have already gone into an update

    history = {"b_g": [], "b_a": [], "sigma_b_g": [], "sigma_b_a": [], "n_updates_used": []}

    for frame_idx, (frame_t, image) in enumerate(frames):
        while imu_idx < len(imu_timestamps) and imu_timestamps[imu_idx] <= frame_t:
            t_curr = int(imu_timestamps[imu_idx])
            state = propagate(state, gyro[imu_idx], accel[imu_idx], (t_curr - t_prev) / 1e9, noise_params)
            t_prev = t_curr
            imu_idx += 1

        tracker.process_frame(image, frame_t)
        state = augment(state, T_BS_cam0)
        clone_frame_ids.append(frame_t)

        n_updates_used = 0
        if (frame_idx + 1) % PROCESS_INTERVAL_FRAMES == 0:
            state, n_updates_used = _process_new_observations(state, tracker, clone_frame_ids, n_used_observations)

            n_to_remove = state.n_clones - MAX_CLONES
            if n_to_remove > 0:
                state = enforce_sliding_window(state, MAX_CLONES)
                clone_frame_ids = clone_frame_ids[n_to_remove:]

        history["b_g"].append(state.b_g.copy())
        history["b_a"].append(state.b_a.copy())
        history["sigma_b_g"].append(np.sqrt(np.diag(state.P)[3:6]))
        history["sigma_b_a"].append(np.sqrt(np.diag(state.P)[9:12]))
        history["n_updates_used"].append(n_updates_used)

    b_g_hist = np.array(history["b_g"])
    b_a_hist = np.array(history["b_a"])
    sigma_bg_hist = np.array(history["sigma_b_g"])
    sigma_ba_hist = np.array(history["sigma_b_a"])

    print(f"Ran {n_frames} frames ({DURATION_S:.0f}s). Total feature updates applied: "
          f"{sum(1 for n in history['n_updates_used'] if n > 0)} batches, "
          f"{sum(history['n_updates_used'])} feature tracks total.")
    print(f"Final gyro bias estimate:  {state.b_g}  (reference: {REFERENCE_BIAS_GYRO})")
    print(f"Final accel bias estimate: {state.b_a}  (reference: {REFERENCE_BIAS_ACCEL})")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    labels = ["x", "y", "z"]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    t_axis = np.arange(n_frames) / 20.0

    for i in range(3):
        axes[0, 0].plot(t_axis, b_g_hist[:, i], color=colors[i], label=f"b_g {labels[i]}")
        axes[0, 0].axhline(REFERENCE_BIAS_GYRO[i], color=colors[i], linestyle="--", alpha=0.5)
        axes[0, 1].plot(t_axis, b_a_hist[:, i], color=colors[i], label=f"b_a {labels[i]}")
        axes[0, 1].axhline(REFERENCE_BIAS_ACCEL[i], color=colors[i], linestyle="--", alpha=0.5)
        axes[1, 0].plot(t_axis, sigma_bg_hist[:, i], color=colors[i], label=labels[i])
        axes[1, 1].plot(t_axis, sigma_ba_hist[:, i], color=colors[i], label=labels[i])

    axes[0, 0].set_title("Gyro bias estimate (dashed = independent reference)")
    axes[0, 0].set_ylabel("rad/s")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_title("Accel bias estimate (dashed = independent reference)")
    axes[0, 1].set_ylabel("m/s^2")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].set_title("Gyro bias 1-sigma uncertainty")
    axes[1, 0].set_ylabel("rad/s")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 1].set_title("Accel bias 1-sigma uncertainty")
    axes[1, 1].set_ylabel("m/s^2")
    axes[1, 1].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
