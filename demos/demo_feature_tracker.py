"""Visual + statistical spot-check of the feature tracker (stage 3) on real
cam0 frames: draws each currently-active track's trail over the last frame,
and plots active-track-count and track-length statistics over the run.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for the top-level modules below

import cv2
import matplotlib.pyplot as plt
import numpy as np

from feature_tracker import FeatureTracker, iter_cam0_frames

N_FRAMES = 100
MAX_FEATURES = 150
TRAIL_LENGTH = 20  # how many past observations to draw per active track


def main():
    tracker = FeatureTracker(max_features=MAX_FEATURES)
    active_count_history = []
    last_image = None

    for timestamp, image in iter_cam0_frames(max_frames=N_FRAMES):
        tracker.process_frame(image, timestamp)
        active_count_history.append(tracker.n_active)
        last_image = image

    track_lengths = [len(obs) for obs in tracker.tracks.values()]
    print(f"Ran {N_FRAMES} frames, target {MAX_FEATURES} features.")
    print(f"Total tracks ever created: {len(tracker.tracks)}")
    print(f"Currently active: {tracker.n_active}")
    print(f"Track length: mean={np.mean(track_lengths):.1f}, median={np.median(track_lengths):.0f}, "
          f"max={max(track_lengths)}")
    print(f"Active count over time: min={min(active_count_history)}, max={max(active_count_history)}")

    overlay = cv2.cvtColor(last_image, cv2.COLOR_GRAY2BGR)
    for track_id in tracker.active_ids:
        obs = tracker.tracks[int(track_id)][-TRAIL_LENGTH:]
        points = [(int(u), int(v)) for _, (u, v) in obs]
        for p0, p1 in zip(points[:-1], points[1:]):
            cv2.line(overlay, p0, p1, (0, 255, 0), 1)
        cv2.circle(overlay, points[-1], 3, (0, 0, 255), -1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Active tracks over last {TRAIL_LENGTH} frames (frame {N_FRAMES})")
    axes[0].axis("off")

    axes[1].plot(active_count_history)
    axes[1].axhline(MAX_FEATURES, color="gray", linestyle="--", label="target")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Active track count")
    axes[1].set_title("Active track count over time")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
