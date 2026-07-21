"""Monocular feature tracking frontend for cam0 (MH_01_easy): Shi-Tomasi
corner detection + KLT (Lucas-Kanade pyramidal) optical flow.

Tracks stay in raw (distorted) pixel coordinates -- KLT operates on image
intensities directly, so lens distortion doesn't matter for tracking itself.
Undistorting into bearing vectors is deferred to the triangulation/measurement
stage, which is the only place it's actually needed.
"""
import os

import cv2
import numpy as np
import pandas as pd

MAV0_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "machine_hall", "MH_01_easy", "MH_01_easy", "mav0")

DEFAULT_MAX_FEATURES = 150
DEFAULT_REPLENISH_RATIO = 0.7  # detect new features once active count drops below this fraction of max
DEFAULT_MIN_DISTANCE = 15  # pixels, both for goodFeaturesToTrack spacing and the new-feature exclusion mask
DEFAULT_QUALITY_LEVEL = 0.01
DEFAULT_KLT_WIN_SIZE = (21, 21)
DEFAULT_KLT_MAX_LEVEL = 3
DEFAULT_KLT_MAX_ERROR = 12.0  # matches OpenCV's typical LK error scale; drop tracks worse than this


def load_cam0_frame_list(mav0_dir=MAV0_DIR):
    """Sorted list of (timestamp_ns, absolute_image_path) for every cam0 frame."""
    path = os.path.join(mav0_dir, "cam0", "data.csv")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.sort_values(df.columns[0])
    data_dir = os.path.join(mav0_dir, "cam0", "data")
    return [(int(row[0]), os.path.join(data_dir, row[1])) for row in df.itertuples(index=False)]


def iter_cam0_frames(mav0_dir=MAV0_DIR, max_frames=None, start_frame=0):
    """Yield (timestamp_ns, grayscale_image) for cam0 frames in order, loaded lazily."""
    frame_list = load_cam0_frame_list(mav0_dir)
    frame_list = frame_list[start_frame:]
    if max_frames is not None:
        frame_list = frame_list[:max_frames]
    for timestamp, path in frame_list:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"could not read image: {path}")
        yield timestamp, image


class FeatureTracker:
    """Maintains a set of KLT-tracked features across frames, with per-track pixel-observation history."""

    def __init__(self, max_features=DEFAULT_MAX_FEATURES, replenish_ratio=DEFAULT_REPLENISH_RATIO,
                 min_distance=DEFAULT_MIN_DISTANCE, quality_level=DEFAULT_QUALITY_LEVEL,
                 klt_win_size=DEFAULT_KLT_WIN_SIZE, klt_max_level=DEFAULT_KLT_MAX_LEVEL,
                 klt_max_error=DEFAULT_KLT_MAX_ERROR):
        self.max_features = max_features
        self.replenish_ratio = replenish_ratio
        self.min_distance = min_distance
        self.quality_level = quality_level
        self.klt_win_size = klt_win_size
        self.klt_max_level = klt_max_level
        self.klt_max_error = klt_max_error

        self._next_id = 0
        self.active_ids = np.empty((0,), dtype=np.int64)
        self._active_points = np.empty((0, 1, 2), dtype=np.float32)
        self._prev_gray = None

        # every track ever created, active or not: track_id -> [(timestamp, (u, v)), ...]
        self.tracks = {}

    @property
    def n_active(self):
        return len(self.active_ids)

    def process_frame(self, image, timestamp):
        """Advance the tracker by one frame. image is a single-channel (grayscale) array."""
        if self._prev_gray is None:
            self._detect_new_features(image, timestamp)
            self._prev_gray = image
            return

        if self.n_active > 0:
            self._track_existing_features(image, timestamp)

        if self.n_active < self.max_features * self.replenish_ratio:
            self._detect_new_features(image, timestamp)

        self._prev_gray = image

    def _track_existing_features(self, image, timestamp):
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, image, self._active_points, None,
            winSize=self.klt_win_size, maxLevel=self.klt_max_level)

        h, w = image.shape[:2]
        in_bounds = (
            (new_points[:, 0, 0] >= 0) & (new_points[:, 0, 0] < w)
            & (new_points[:, 0, 1] >= 0) & (new_points[:, 0, 1] < h)
        )
        keep = (status.reshape(-1) == 1) & (error.reshape(-1) <= self.klt_max_error) & in_bounds

        self._active_points = new_points[keep]
        self.active_ids = self.active_ids[keep]
        for track_id, point in zip(self.active_ids, self._active_points):
            self.tracks[int(track_id)].append((timestamp, (float(point[0, 0]), float(point[0, 1]))))

    def _detect_new_features(self, image, timestamp):
        mask = np.full(image.shape[:2], 255, dtype=np.uint8)
        for point in self._active_points:
            cv2.circle(mask, (int(point[0, 0]), int(point[0, 1])), self.min_distance, 0, -1)

        n_needed = self.max_features - self.n_active
        if n_needed <= 0:
            return
        corners = cv2.goodFeaturesToTrack(image, maxCorners=n_needed, qualityLevel=self.quality_level,
                                           minDistance=self.min_distance, mask=mask)
        if corners is None:
            return

        new_ids = np.arange(self._next_id, self._next_id + len(corners))
        self._next_id += len(corners)
        for track_id, point in zip(new_ids, corners):
            self.tracks[int(track_id)] = [(timestamp, (float(point[0, 0]), float(point[0, 1])))]

        self._active_points = np.concatenate([self._active_points, corners], axis=0)
        self.active_ids = np.concatenate([self.active_ids, new_ids])
