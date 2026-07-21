"""Visual spot-check of stage 5 (measurement_model.py): reprojection error on
a real triangulated track, and a direct demonstration of the null-space
projection's core property -- perturbing the feature's position changes the
raw residual linearly, but leaves the *projected* residual changing only
quadratically (i.e. its first-order feature-position dependence is exactly
eliminated).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for the top-level modules below

import matplotlib.pyplot as plt
import numpy as np

import ground_truth as gt
from feature_tracker import FeatureTracker, iter_cam0_frames
from frames import body_to_sensor_position, load_T_BS
from measurement_model import null_space_project, reprojection_residual_and_jacobians, stack_feature_observations
from triangulation import load_cam0_intrinsics, triangulate_feature, undistort_normalized

N_FRAMES = 20
MIN_TRACK_LENGTH = 15


def pick_real_track():
    K, dist_coeffs = load_cam0_intrinsics()
    T_BS_cam0 = load_T_BS(f"{gt.MAV0_DIR}/cam0/sensor.yaml")

    gt_timestamps, _, _, _ = gt._load_state_ground_truth()
    cam0_frames = gt.load_cam0_timestamps()
    start_frame = int(np.searchsorted(cam0_frames, gt_timestamps[0])) + 30

    tracker = FeatureTracker(max_features=150)
    for timestamp, image in iter_cam0_frames(max_frames=N_FRAMES, start_frame=start_frame):
        tracker.process_frame(image, timestamp)

    track_id = max(tracker.tracks, key=lambda tid: len(tracker.tracks[tid]))
    observations = tracker.tracks[track_id]
    assert len(observations) >= MIN_TRACK_LENGTH, "didn't find a long enough real track in this window"

    camera_poses, bearings = [], []
    for timestamp, (u, v) in observations:
        q_cam = gt.interpolate_cam0_orientation(timestamp)
        p_cam = body_to_sensor_position(gt.interpolate_ground_truth_position(timestamp),
                                         gt.interpolate_ground_truth_orientation(timestamp), T_BS_cam0)
        camera_poses.append((q_cam, p_cam))
        bearings.append(undistort_normalized([[u, v]], K, dist_coeffs)[0])

    return camera_poses, bearings, K


def main():
    camera_poses, bearings, K = pick_real_track()
    focal_length = K[0, 0]
    X_world = triangulate_feature(camera_poses, bearings)

    # --- per-observation reprojection error, converted to ~pixels for intuition ---
    pixel_errors = []
    for (q_cam, p_cam), bearing in zip(camera_poses, bearings):
        r, _, _ = reprojection_residual_and_jacobians(X_world, q_cam, p_cam, bearing)
        pixel_errors.append(np.linalg.norm(r) * focal_length)

    print(f"Track length: {len(bearings)} observations")
    print(f"Reprojection error (pixels): mean={np.mean(pixel_errors):.3f}, max={np.max(pixel_errors):.3f}")

    # --- null-space invariance: sweep a feature-position perturbation and
    # compare how the RAW vs PROJECTED residual changes scale with it ---
    clone_indices = list(range(len(camera_poses)))
    r0, H_x0, H_f0 = stack_feature_observations(X_world, camera_poses, bearings, clone_indices,
                                                 n_clones=len(camera_poses))
    _, H_o0 = null_space_project(r0, H_x0, H_f0)
    Q, _ = np.linalg.qr(H_f0, mode="complete")
    null_basis = Q[:, 3:]  # same basis null_space_project computes internally

    rng = np.random.default_rng(0)
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)

    epsilons = np.logspace(-5, -2, 15)
    raw_diffs, projected_diffs = [], []
    for eps in epsilons:
        r1, _, _ = stack_feature_observations(X_world + direction * eps, camera_poses, bearings, clone_indices,
                                               n_clones=len(camera_poses))
        raw_diffs.append(np.linalg.norm(r1 - r0))
        projected_diffs.append(np.linalg.norm(null_basis.T @ (r1 - r0)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(range(len(pixel_errors)), pixel_errors, color="tab:blue")
    axes[0].set_xlabel("Observation index (frame order)")
    axes[0].set_ylabel("Reprojection error (pixels)")
    axes[0].set_title("Per-observation reprojection error")

    axes[1].loglog(epsilons, raw_diffs, "o-", label="Raw residual change (H_x, H_f)")
    axes[1].loglog(epsilons, projected_diffs, "o-", label="Null-space projected residual change")
    axes[1].loglog(epsilons, epsilons, "k--", alpha=0.5, label="slope 1 (linear) reference")
    axes[1].loglog(epsilons, epsilons ** 2, "k:", alpha=0.5, label="slope 2 (quadratic) reference")
    axes[1].set_xlabel("Feature position perturbation (m)")
    axes[1].set_ylabel("Residual change (normalized units)")
    axes[1].set_title("Null-space projection eliminates first-order\nfeature-position dependence")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
