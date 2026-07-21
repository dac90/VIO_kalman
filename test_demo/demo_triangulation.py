"""Visual + statistical spot-check of triangulation (stage 4): tracks real
cam0 features, triangulates each surviving track using ground-truth camera
poses, and plots the resulting 3D point cloud alongside the camera trajectory.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for the top-level modules below

import matplotlib.pyplot as plt
import numpy as np

import ground_truth as gt
from feature_tracker import FeatureTracker, iter_cam0_frames
from frames import body_to_sensor_position, load_T_BS
from quaternion_utils import quat_conjugate, rotate_vector_by_quaternion
from triangulation import load_cam0_intrinsics, reprojection_residuals, triangulate_feature, undistort_normalized

N_FRAMES = 60
MIN_TRACK_LENGTH = 10


def main():
    K, dist_coeffs = load_cam0_intrinsics()
    T_BS_cam0 = load_T_BS(f"{gt.MAV0_DIR}/cam0/sensor.yaml")

    # ground truth doesn't cover cam0's first ~1s, so start well past that
    gt_timestamps, _, _, _ = gt._load_state_ground_truth()
    cam0_frames = gt.load_cam0_timestamps()
    start_frame = int(np.searchsorted(cam0_frames, gt_timestamps[0])) + 30

    tracker = FeatureTracker(max_features=150)
    camera_positions = []
    for timestamp, image in iter_cam0_frames(max_frames=N_FRAMES, start_frame=start_frame):
        tracker.process_frame(image, timestamp)
        q_body = gt.interpolate_ground_truth_orientation(timestamp)
        p_body = gt.interpolate_ground_truth_position(timestamp)
        camera_positions.append(body_to_sensor_position(p_body, q_body, T_BS_cam0))
    camera_positions = np.array(camera_positions)

    points_world = []
    reprojection_errors = []
    depths = []
    for track_id, observations in tracker.tracks.items():
        if len(observations) < MIN_TRACK_LENGTH:
            continue

        cameras, bearings = [], []
        for timestamp, (u, v) in observations:
            q_cam = gt.interpolate_cam0_orientation(timestamp)
            p_cam = body_to_sensor_position(gt.interpolate_ground_truth_position(timestamp),
                                             gt.interpolate_ground_truth_orientation(timestamp), T_BS_cam0)
            cameras.append((q_cam, p_cam))
            bearings.append(undistort_normalized([[u, v]], K, dist_coeffs)[0])

        X_world = triangulate_feature(cameras, bearings)

        cam_depths = [rotate_vector_by_quaternion(quat_conjugate(q), X_world - p)[2] for q, p in cameras]
        if min(cam_depths) <= 0:
            continue  # behind some camera -- a bad/degenerate track, skip for this spot-check

        points_world.append(X_world)
        reprojection_errors.append(np.linalg.norm(reprojection_residuals(X_world, cameras, bearings)) / len(cameras))
        depths.append(np.mean(cam_depths))

    points_world = np.array(points_world)
    print(f"Ran {N_FRAMES} frames; {len(tracker.tracks)} tracks total, "
          f"{len(points_world)} triangulated with length >= {MIN_TRACK_LENGTH} and positive depth.")
    print(f"Mean depth: {np.mean(depths):.2f}m, range [{min(depths):.2f}, {max(depths):.2f}]m")
    print(f"Mean per-observation reprojection error (normalized units): {np.mean(reprojection_errors):.5f}")

    # a handful of poorly-constrained tracks (little parallax over this short
    # window) triangulate far away and would otherwise blow out the axis
    # scale -- no outlier rejection yet (that's stage 6), so just clip the
    # view around the camera trajectory for a readable plot; nothing is
    # dropped from the printed stats above
    margin = 3.0
    center = camera_positions.mean(axis=0)
    half_range = np.maximum(np.ptp(camera_positions, axis=0) / 2 + margin, margin)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2],
             color="tab:blue", label="Camera trajectory (ground truth)")
    ax.scatter(points_world[:, 0], points_world[:, 1], points_world[:, 2],
               color="tab:orange", s=8, label="Triangulated features")
    ax.set_xlim(center[0] - half_range[0], center[0] + half_range[0])
    ax.set_ylim(center[1] - half_range[1], center[1] + half_range[1])
    ax.set_zlim(center[2] - half_range[2], center[2] + half_range[2])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Stage 4 demo: {len(points_world)} triangulated points over {N_FRAMES} frames")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
