"""Stage 8: the full MSCKF pipeline, wiring IMU propagation, feature
tracking, triangulation, the measurement update, and marginalization
together into one continuously-runnable system.

A feature's observation history is used in an EKF update when its track
naturally ends (KLT stops tracking it) -- that's the common case, and it
gets to use its whole accumulated history at once.

Eviction defers to this: max_clones is a *soft* cap. A clone still
referenced by a still-active track's unprocessed observations is not
evicted; the window is simply allowed to grow a little until that track
ends on its own. Without this, a long-lived track sitting right at the
window's edge would get force-processed on every single subsequent frame
with only ~1-2 new (low-parallax, largely useless) observations each time --
repeatedly failing the chi-square gate and never contributing anything,
while its early observations (its best baseline) are wasted piecemeal. A
hard_max_clones safety valve still forces this in the rare case a track
outlives it, so the window can't grow unboundedly.
"""
import numpy as np

from feature_tracker import FeatureTracker
from measurement_model import null_space_project, stack_feature_observations
from msckf_state import MSCKFState, augment, marginalize_clones, propagate
from msckf_update import ekf_update, passes_chi_square_gate
from triangulation import triangulate_feature, undistort_normalized


class MSCKFPipeline:
    """Owns the filter state, feature tracker, and per-track bookkeeping.

    Call process_imu for every IMU sample (in timestamp order) and
    process_image for every camera frame, interleaved in timestamp order.
    """

    def __init__(self, p0, v0, q0, b_g0, b_a0, P0, T_BS_cam0, K, dist_coeffs, noise_params,
                 max_clones=20, hard_max_clones=None, max_features=150,
                 observation_noise_std=5.0 / 458.0,  # ~5px: 1px was overconfident once triangulation
                                                      # uses the filter's own (imperfect) poses instead
                                                      # of ground truth -- see the divergence this caused
                                                      # when tuning this pipeline over long runs
                 min_track_length=6, gate_confidence=0.95):
        self.state = MSCKFState.initialize(p0, v0, q0, b_g0, b_a0, P0)
        self.T_BS_cam0 = T_BS_cam0
        self.K = K
        self.dist_coeffs = dist_coeffs
        self.noise_params = noise_params
        self.max_clones = max_clones
        self.hard_max_clones = hard_max_clones if hard_max_clones is not None else max_clones * 3
        self.observation_noise_std = observation_noise_std
        self.min_track_length = min_track_length
        self.gate_confidence = gate_confidence

        self.tracker = FeatureTracker(max_features=max_features)
        self.clone_frame_ids = []       # clone_frame_ids[i] = timestamp of state.clone_positions[i]
        self._n_used_observations = {}  # track_id -> how many observations already used in an update

        # (timestamp, position, orientation) for every clone once it's
        # marginalized -- otherwise its final (fully-updated) pose would be
        # lost the moment it leaves the sliding window, making it impossible
        # to evaluate anything but the current tail of the trajectory
        self.trajectory_history = []

        self.n_updates_applied = 0
        self.n_tracks_rejected = 0

    def process_imu(self, gyro, accel, dt):
        self.state = propagate(self.state, gyro, accel, dt, self.noise_params)

    def process_image(self, timestamp, image):
        """Track features, clone the current pose, use any now-ready tracks, then marginalize.

        Returns (n_updates_this_step, n_rejected_this_step).
        """
        previously_active = set(self.tracker.active_ids.tolist())
        self.tracker.process_frame(image, timestamp)
        self.state = augment(self.state, self.T_BS_cam0)
        self.clone_frame_ids.append(timestamp)

        just_ended = previously_active - set(self.tracker.active_ids.tolist())

        n_to_evict = self._n_safe_to_evict()
        doomed_timestamps = set(self.clone_frame_ids[:n_to_evict]) if n_to_evict > 0 else set()

        n_updates, n_rejected = self._process_ready_tracks(just_ended, doomed_timestamps)

        if n_to_evict > 0:
            for i in range(n_to_evict):
                self.trajectory_history.append((self.clone_frame_ids[i], self.state.clone_positions[i].copy(),
                                                 self.state.clone_orientations[i].copy()))
            self.state = marginalize_clones(self.state, range(n_to_evict))
            self.clone_frame_ids = self.clone_frame_ids[n_to_evict:]

        return n_updates, n_rejected

    def _n_safe_to_evict(self):
        """How many of the oldest clones can be dropped right now.

        Normally, that's every clone past max_clones except ones a
        still-active track has an unprocessed observation at -- those are
        left alone (growing the window) until that track ends naturally.
        If a track's longevity would grow the window past hard_max_clones,
        this forces eviction anyway; _process_ready_tracks then has to use
        whatever that track has accumulated so far, however little.
        """
        target = self.state.n_clones - self.max_clones
        if target <= 0:
            return 0

        referenced = set()
        for track_id in self.tracker.active_ids.tolist():
            already_used = self._n_used_observations.get(track_id, 0)
            for t, _ in self.tracker.tracks[track_id][already_used:]:
                referenced.add(t)

        n_safe = 0
        for t in self.clone_frame_ids[:target]:
            if t in referenced:
                break
            n_safe += 1

        n_forced = self.state.n_clones - self.hard_max_clones
        return max(n_safe, n_forced)

    def full_trajectory(self):
        """(timestamps, positions, orientations) for every clone ever created, marginalized or still active."""
        timestamps = [t for t, _, _ in self.trajectory_history] + list(self.clone_frame_ids)
        positions = [p for _, p, _ in self.trajectory_history] + list(self.state.clone_positions)
        orientations = [q for _, _, q in self.trajectory_history] + list(self.state.clone_orientations)
        return timestamps, positions, orientations

    def _process_ready_tracks(self, just_ended, doomed_timestamps):
        clone_index_of = {t: i for i, t in enumerate(self.clone_frame_ids)}

        r_o_batch, H_o_batch = [], []
        n_rejected = 0
        for track_id, observations in self.tracker.tracks.items():
            already_used = self._n_used_observations.get(track_id, 0)
            new_obs = observations[already_used:]
            if not new_obs:
                continue

            touches_doomed_clone = any(t in doomed_timestamps for t, _ in new_obs)
            if track_id not in just_ended and not touches_doomed_clone:
                continue  # not ready yet: still being tracked and in no danger of losing data

            self._n_used_observations[track_id] = len(observations)
            usable = [(t, uv) for t, uv in new_obs if t in clone_index_of]

            # a track that ended naturally has no time pressure, so it can
            # afford to require a nicer minimum before bothering at all. But
            # a still-active track forced out early by eviction pressure
            # (touches_doomed_clone, not just_ended) has no such luxury: its
            # observations are gone the moment we skip them, so use whatever
            # it's got -- down to the bare minimum of 2 views needed to
            # triangulate at all -- rather than lose them for nothing. This
            # matters most for long-lived tracks, which otherwise get
            # re-triggered every single frame with ~1 new observation each
            # time (never enough to clear min_track_length) and have their
            # entire, most-valuable-baseline history silently discarded one
            # frame at a time.
            required_length = self.min_track_length if track_id in just_ended else 2
            if len(usable) < required_length:
                continue

            r_o, H_o = self._build_measurement(usable, clone_index_of)
            if r_o is None:
                continue

            if passes_chi_square_gate(r_o, H_o, self.state.P, self.observation_noise_std, self.gate_confidence):
                r_o_batch.append(r_o)
                H_o_batch.append(H_o)
            else:
                n_rejected += 1

        n_updates = len(r_o_batch)
        if r_o_batch:
            self.state = ekf_update(self.state, np.concatenate(r_o_batch), np.concatenate(H_o_batch, axis=0),
                                     self.observation_noise_std)
        self.n_updates_applied += n_updates
        self.n_tracks_rejected += n_rejected
        return n_updates, n_rejected

    def _build_measurement(self, usable, clone_index_of):
        clone_indices = [clone_index_of[t] for t, _ in usable]
        camera_poses = [(self.state.clone_orientations[i], self.state.clone_positions[i]) for i in clone_indices]
        bearings = [undistort_normalized([[u, v]], self.K, self.dist_coeffs)[0] for _, (u, v) in usable]
        try:
            X_est = triangulate_feature(camera_poses, bearings)
            r, H_x, H_f = stack_feature_observations(X_est, camera_poses, bearings, clone_indices,
                                                      n_clones=self.state.n_clones)
            return null_space_project(r, H_x, H_f)
        except np.linalg.LinAlgError:
            # degenerate geometry (e.g. near-zero parallax); no parallax check yet (deferred earlier)
            return None, None
