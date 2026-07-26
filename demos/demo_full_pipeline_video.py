"""Stage 8 capstone video: renders a live 2x2 dashboard as the full pipeline
runs -- the trajectory and position-error plots growing frame by frame,
alongside the raw cam0 feed and a feature-tracking visualization -- and
writes it out as an MP4.

Reuses demo_full_pipeline.py's exact setup (same duration, same dummy bias,
same MAX_CLONES) so the numbers in the video match that demo's printed ATE/RPE.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for the top-level modules below

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import ground_truth as gt
from feature_tracker import iter_cam0_frames
from frames import body_to_sensor_position, load_T_BS
from imu_propagation import _load_imu_measurements
from msckf_pipeline import MSCKFPipeline
from msckf_state import load_imu_noise_params
from quaternion_utils import quat_error_vector
from triangulation import load_cam0_intrinsics

DURATION_S = 60.0
MAX_CLONES = 20
VIDEO_FPS = 20
TRAIL_LENGTH = 20  # how many past observations to draw per tracked feature
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_pipeline.mp4")

# same independent "dummy bias" used throughout the demos -- see
# demos/demo_full_pipeline.py for the full explanation
DUMMY_BIAS_GYRO = np.array([-0.00233173, 0.02172386, 0.07821335])
DUMMY_BIAS_ACCEL = np.array([-0.04066623, 0.1155297, 0.05121861])

INITIAL_SIGMA = dict(theta=1e-3, b_g=1e-2, v=1e-1, b_a=1e-1, p=1e-3)


def initial_covariance():
    diag = ([INITIAL_SIGMA["theta"] ** 2] * 3 + [INITIAL_SIGMA["b_g"] ** 2] * 3 + [INITIAL_SIGMA["v"] ** 2] * 3
            + [INITIAL_SIGMA["b_a"] ** 2] * 3 + [INITIAL_SIGMA["p"] ** 2] * 3)
    return np.diag(diag)


def draw_tracked_features(image, tracker):
    """Grayscale frame -> BGR frame with each active track's recent trail drawn on it."""
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for track_id in tracker.active_ids:
        observations = tracker.tracks[int(track_id)][-TRAIL_LENGTH:]
        points = [(int(u), int(v)) for _, (u, v) in observations]
        for p0, p1 in zip(points[:-1], points[1:]):
            cv2.line(overlay, p0, p1, (0, 255, 0), 1)
        cv2.circle(overlay, points[-1], 3, (0, 0, 255), -1)
    return overlay


def main():
    K, dist_coeffs = load_cam0_intrinsics()
    T_BS_cam0 = load_T_BS(f"{gt.MAV0_DIR}/cam0/sensor.yaml")
    noise_params = load_imu_noise_params()

    gt_timestamps, _, _, _ = gt._load_state_ground_truth()
    cam0_all_timestamps = gt.load_cam0_timestamps()
    start_frame = int(np.searchsorted(cam0_all_timestamps, gt_timestamps[0])) + 30
    n_frames = int(DURATION_S * 20)

    frames = list(iter_cam0_frames(max_frames=n_frames, start_frame=start_frame))
    t0 = frames[0][0]
    p0 = gt.interpolate_ground_truth_position(t0)
    v0 = gt.interpolate_ground_truth_velocity(t0)
    q0 = gt.interpolate_ground_truth_orientation(t0)

    pipeline = MSCKFPipeline(p0, v0, q0, DUMMY_BIAS_GYRO, DUMMY_BIAS_ACCEL, initial_covariance(),
                              T_BS_cam0, K, dist_coeffs, noise_params, max_clones=MAX_CLONES)

    imu_timestamps, gyro, accel = _load_imu_measurements()
    imu_idx = np.searchsorted(imu_timestamps, t0, side="right")
    t_prev = t0

    # ground truth for every frame's timestamp, computed up front for the
    # trajectory plot's static reference line and the growing error curves
    gt_positions_full = np.array([body_to_sensor_position(gt.interpolate_ground_truth_position(t),
                                                            gt.interpolate_ground_truth_orientation(t), T_BS_cam0)
                                   for t, _ in frames])
    gt_orientations_full = [gt.interpolate_cam0_orientation(t) for t, _ in frames]

    # ---- figure: 2x2 grid (trajectory / error-over-time / camera / tracked features) ----
    fig = plt.figure(figsize=(12, 9))
    ax_traj = fig.add_subplot(2, 2, 1, projection="3d")
    ax_err = fig.add_subplot(2, 2, 2)
    ax_cam = fig.add_subplot(2, 2, 3)
    ax_feat = fig.add_subplot(2, 2, 4)

    ax_traj.plot(*gt_positions_full.T, color="tab:blue", label="Ground truth", linewidth=1)
    (est_line,) = ax_traj.plot([], [], [], color="tab:green", label="MSCKF estimate", linewidth=1.5)
    ax_traj.set_xlabel("X (m)")
    ax_traj.set_ylabel("Y (m)")
    ax_traj.set_zlabel("Z (m)")
    ax_traj.set_title("Trajectory: estimate vs ground truth")
    ax_traj.legend(fontsize=8, loc="upper left")
    margin = 0.5
    mins = gt_positions_full.min(axis=0) - margin
    maxs = gt_positions_full.max(axis=0) + margin
    ax_traj.set_xlim(mins[0], maxs[0])
    ax_traj.set_ylim(mins[1], maxs[1])
    ax_traj.set_zlim(mins[2], maxs[2])

    (err_line,) = ax_err.plot([], [], color="tab:red", label="Position error")
    ax_err.set_xlim(0, DURATION_S)
    ax_err.set_ylim(0, 0.1)
    ax_err.set_xlabel("Time (s)")
    ax_err.set_ylabel("Position error (m)", color="tab:red")
    ax_err.tick_params(axis="y", labelcolor="tab:red")
    ax_err.set_title("Absolute position + orientation error over time")

    # orientation error shares the same panel via a second y-axis (different units --
    # radians, not meters -- so it can't share the first axis directly)
    ax_err_theta = ax_err.twinx()
    (err_theta_line,) = ax_err_theta.plot([], [], color="tab:purple", label="Orientation error")
    ax_err_theta.set_ylim(0, 0.1)
    ax_err_theta.set_ylabel("Orientation error (rad)", color="tab:purple")
    ax_err_theta.tick_params(axis="y", labelcolor="tab:purple")
    ax_err.legend([err_line, err_theta_line], ["Position error", "Orientation error"], fontsize=8, loc="upper left")

    blank = np.zeros((480, 752, 3), dtype=np.uint8)
    cam_im = ax_cam.imshow(blank)
    ax_cam.set_title("cam0")
    ax_cam.axis("off")

    feat_im = ax_feat.imshow(blank)
    ax_feat.set_title("Tracked features")
    ax_feat.axis("off")

    fig.tight_layout()
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (width, height))

    est_positions = []
    error_history = []
    theta_error_history = []
    time_history = []
    max_error_seen = 0.1
    max_theta_error_seen = 0.1

    for frame_idx, (frame_t, image) in enumerate(frames):
        while imu_idx < len(imu_timestamps) and imu_timestamps[imu_idx] <= frame_t:
            t_curr = int(imu_timestamps[imu_idx])
            pipeline.process_imu(gyro[imu_idx], accel[imu_idx], (t_curr - t_prev) / 1e9)
            t_prev = t_curr
            imu_idx += 1

        pipeline.process_image(frame_t, image)

        est_p = pipeline.state.clone_positions[-1]
        est_q = pipeline.state.clone_orientations[-1]
        est_positions.append(est_p)
        error_history.append(np.linalg.norm(est_p - gt_positions_full[frame_idx]))
        theta_error_history.append(np.linalg.norm(quat_error_vector(gt_orientations_full[frame_idx], est_q)))
        time_history.append(frame_idx / 20.0)

        est_arr = np.array(est_positions)
        est_line.set_data(est_arr[:, 0], est_arr[:, 1])
        est_line.set_3d_properties(est_arr[:, 2])

        err_line.set_data(time_history, error_history)
        if error_history[-1] > max_error_seen:
            max_error_seen = error_history[-1] * 1.1
            ax_err.set_ylim(0, max_error_seen)

        err_theta_line.set_data(time_history, theta_error_history)
        if theta_error_history[-1] > max_theta_error_seen:
            max_theta_error_seen = theta_error_history[-1] * 1.1
            ax_err_theta.set_ylim(0, max_theta_error_seen)

        cam_im.set_data(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
        feat_im.set_data(cv2.cvtColor(draw_tracked_features(image, pipeline.tracker), cv2.COLOR_BGR2RGB))

        fig.canvas.draw()
        frame_rgba = np.asarray(fig.canvas.buffer_rgba())
        writer.write(cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR))

        if frame_idx % 50 == 0:
            print(f"frame {frame_idx}/{len(frames)}  pos_err={error_history[-1]:.4f}m  "
                  f"theta_err={theta_error_history[-1]:.4f}rad")

    writer.release()
    print(f"Saved video ({len(frames)} frames, {DURATION_S:.0f}s @ {VIDEO_FPS}fps) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
