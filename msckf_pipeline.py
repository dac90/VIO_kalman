"""The full MSCKF pipeline: wires IMU propagation, feature tracking,
triangulation, the EKF measurement update, and marginalization into one
continuously-runnable filter.

A feature's observation history is used in an EKF update when its track
naturally ends (KLT stops tracking it), consuming its whole accumulated
history at once. max_clones is a strict, always-enforced cap -- the
sliding window is marginalized back down to it every frame. A still-active
track about to lose its oldest observation to eviction is force-processed
with whatever it has (down to a bare minimum of 2 views) rather than
losing that data outright.

A *soft* cap (letting the window grow past max_clones for still-active
tracks) was tried and reverted: a longer window lets features get
triangulated against increasingly stale camera poses, and Gauss-Newton
fits the feature to those poses by construction, so the reprojection
residual looks small and "consistent" even when the poses themselves are
wrong. The EKF update then reads as highly informative and shrinks P
accordingly, reinforcing the existing drift instead of correcting it -- a
feedback loop that gets worse the longer the window grows. A small,
strictly bounded window (matching canonical MSCKF-VIO practice) starves
that loop of the staleness it needs; triangulation.py's parallax and
cheirality checks handle the thin, force-processed tracks this policy
produces. See demos/demo_full_pipeline.py's docstring for validated numbers.

A trio of stationary-only pseudo-measurements -- ZUPT (zero-velocity),
ZARU (zero angular-rate), and gravity/tilt alignment, all in
msckf_update.py -- fires whenever _update_stationary_state() detects the
platform is at rest from raw IMU statistics. A stationary camera has zero
parallax on everything it sees, so vision updates go from unhelpful to
actively harmful (the same self-referential-triangulation problem above,
now with no corrective signal to push back against it), while unaided
dead-reckoning drifts on residual bias, unconstrained. See
_update_stationary_state's docstring for the detector itself, and
zero_velocity_update's docstring for why its mean correction is restricted
to the velocity block only -- ZARU and gravity-alignment don't need that
restriction, since they measure bias/tilt directly rather than through a
correlation, which is also why they exist: a bare ZUPT has no way to
correct the residual accel bias actually driving the drift it fights.
Detection and the corrections both run at IMU rate (200Hz) rather than
camera rate (20Hz), since neither depends on camera frames at all.
"""
from collections import deque

import numpy as np

from feature_tracker import FeatureTracker
from imu_propagation import GRAVITY_WORLD
from measurement_model import null_space_project, stack_feature_observations
from msckf_state import MSCKFState, augment, marginalize_clones, propagate
from msckf_update import chi_square_threshold, ekf_update, gravity_alignment_update, passes_chi_square_gate, \
    zero_angular_rate_update, zero_velocity_update
from triangulation import triangulate_and_validate, undistort_normalized

GRAVITY_MAGNITUDE = np.linalg.norm(GRAVITY_WORLD)

# Covariance floor (std, per IMU error-state block) -- see ekf_update's docstring in
# msckf_update.py. Empirically swept (demos/demo_divergence_analysis.py), not picked from
# first principles: a floor an order of magnitude looser than this makes long-run divergence
# *worse*, since keeping P artificially open raises the Kalman gain for every subsequent
# measurement, including bad self-referential ones. These values roughly halve position
# error at t=30s and cut it ~5x at t=50s versus no floor, without eliminating eventual
# divergence past t~60s.
DEFAULT_MIN_SIGMA = dict(theta=4e-4, b_g=4e-4, v=4e-3, b_a=4e-3, p=4e-4)


class MSCKFPipeline:
    """Owns the filter state, feature tracker, and per-track bookkeeping.

    Call process_imu for every IMU sample (in timestamp order) and
    process_image for every camera frame, interleaved in timestamp order.
    """

    def __init__(self, p0, v0, q0, b_g0, b_a0, P0, T_BS_cam0, K, dist_coeffs, noise_params,
                 max_clones=20, max_features=150,
                 observation_noise_std=5.0 / 458.0,  # ~5px: 1px is overconfident once triangulation
                                                      # uses the filter's own (imperfect) poses
                                                      # rather than ground truth
                 min_track_length=6, gate_confidence=0.95,
                 min_parallax=0.2,  # meters of camera translation orthogonal to a feature's bearing
                                    # ray (MSCKF-VIO's default) -- below this, depth along that ray
                                    # is unobservable regardless of what the chi-square gate thinks
                 max_reprojection_error=None,
                 min_sigma=DEFAULT_MIN_SIGMA,  # covariance floor (std); pass None/{} to disable
                 enable_zupt=True,
                 zupt_window_samples=100,  # ~0.5s at the EuRoC IMU's 200Hz
                 zupt_confidence=0.95,  # GLRT confidence -- see stationary-detector docstring below
                 zupt_noise_inflation=20.0,  # empirically tuned -- see docstring below
                 zupt_hold_seconds=2.0,  # continuous non-stationary time needed to exit ZUPT
                 zupt_noise_std=1e-3,  # m/s; tight, since a detected-stationary period is high-confidence
                 zaru_noise_std=1e-3,  # rad/s; ZUPT's rotational companion (see zero_angular_rate_update)
                 tilt_noise_std=1e-2,  # m/s^2; see gravity_alignment_update -- looser than
                                       # ZUPT/ZARU since "a=0" is only approximate under real vibration
                 zupt_update_interval=0.05):  # s; corrections apply at most this often -- see
                                               # process_imu's docstring for why detection and
                                               # application run at different rates
        self.state = MSCKFState.initialize(p0, v0, q0, b_g0, b_a0, P0)
        self.T_BS_cam0 = T_BS_cam0
        self.K = K
        self.dist_coeffs = dist_coeffs
        self.noise_params = noise_params
        self.max_clones = max_clones
        self.observation_noise_std = observation_noise_std
        self.min_track_length = min_track_length
        self.gate_confidence = gate_confidence
        self.min_parallax = min_parallax
        # a fixed (P-independent) sanity bound on post-refinement reprojection error, in
        # normalized bearing units -- default is a generous multiple of the assumed
        # observation noise, just to catch triangulations that clearly didn't converge
        self.max_reprojection_error = (max_reprojection_error if max_reprojection_error is not None
                                        else 5.0 * observation_noise_std)
        self.min_variance = (np.array([min_sigma["theta"] ** 2] * 3 + [min_sigma["b_g"] ** 2] * 3
                                       + [min_sigma["v"] ** 2] * 3 + [min_sigma["b_a"] ** 2] * 3
                                       + [min_sigma["p"] ** 2] * 3)
                              if min_sigma else None)

        self.enable_zupt = enable_zupt
        self.zupt_confidence = zupt_confidence
        self.zupt_noise_inflation = zupt_noise_inflation
        self.zupt_hold_seconds = zupt_hold_seconds
        self.zupt_noise_std = zupt_noise_std
        self.zaru_noise_std = zaru_noise_std
        self.tilt_noise_std = tilt_noise_std
        self.zupt_update_interval = zupt_update_interval
        self._imu_history = deque(maxlen=zupt_window_samples)
        self._zupt_active = False
        self._zupt_hold_time = 0.0
        self._time_since_last_stationary_update = 0.0

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
        """Propagate, then (if enabled) update stationarity detection and, if warranted,
        apply the stationary-period corrections.

        Detection (_update_stationary_state) runs every IMU sample (200Hz) so a stop gets
        noticed as fast as possible, but applying the corrections at that same rate is
        actively harmful: consecutive accelerometer/gyro samples 5ms apart aren't independent
        draws (real sensor noise and vibration are correlated over such short gaps), so
        treating ~200 corrections a second as that many independent measurements manufactures
        far more confidence than the data supports -- enough to let a small tilt/accel-bias
        ambiguity inherent to gravity_alignment_update get "locked in" wrong. Corrections
        apply at a bounded rate instead (zupt_update_interval, default 20Hz), keeping the
        fast reaction time from detection without the overcounting.
        """
        self.state = propagate(self.state, gyro, accel, dt, self.noise_params)
        if not self.enable_zupt:
            return
        self._imu_history.append((np.asarray(gyro), np.asarray(accel)))
        self._update_stationary_state(dt)

        self._time_since_last_stationary_update += dt
        if self._zupt_active and self._time_since_last_stationary_update >= self.zupt_update_interval:
            self._time_since_last_stationary_update = 0.0
            self.state = zero_velocity_update(self.state, self.zupt_noise_std, min_variance=self.min_variance)
            self.state = zero_angular_rate_update(self.state, gyro, self.zaru_noise_std,
                                                   min_variance=self.min_variance)
            self.state = gravity_alignment_update(self.state, accel, self.tilt_noise_std,
                                                   min_variance=self.min_variance)

    def _update_stationary_state(self, dt):
        """Update self._zupt_active using a GLRT (generalized likelihood-ratio test) on raw
        IMU statistics -- deliberately independent of the filter's own pose/velocity estimate
        (unlike triangulation.py's parallax check, which uses the filter's own camera-clone
        poses and so can be fooled: once the filter has drifted, it can believe a stationary
        scene has real parallax, since its own poses appear to have moved). Runs once per IMU
        sample (200Hz) rather than once per camera frame, since ZUPT/ZARU/gravity-alignment
        are pure IMU-side corrections with no dependency on camera frames at all.

        This is the SHOE ("stance hypothesis optimal estimation") detector from Skog et al.,
        "Zero-Velocity Detection -- An Algorithm Evaluation" (IEEE Trans. Biomedical
        Engineering, 2010), standard in ZUPT-aided INS. It compares two hypotheses per
        window: H0 (stationary: gyro reads pure noise, accel reads gravity from a fixed but
        unknown direction) vs. H1 (moving). Under H0, each raw residual, normalized by the
        IMU's own characterized noise density (self.noise_params, discretized as
        density^2/dt), is approximately a unit-variance Gaussian, so the summed squared
        residuals over the window are approximately chi-square distributed -- reusing
        chi_square_threshold (already used for vision-outlier gating) to supply one
        principled zupt_confidence parameter instead of hand-tuned thresholds.

        Two adaptations from the textbook derivation, both because this IMU's biases are too
        large to ignore: gyro and accel readings are corrected by the filter's own b_g/b_a
        estimates before testing (otherwise a real but steady rotation/acceleration reads as
        falsely "stationary", mistaken for bias). Gravity direction is estimated fresh from
        each window's own mean rather than the filter's orientation estimate, to avoid the
        same self-referential trap as the parallax check.

        zupt_noise_inflation compensates for the manufacturer's noise-density spec being an
        idealized white-noise number real sensor data doesn't live up to (residual
        bias-correction error, real vibration, and non-white noise all inflate variance
        beyond the datasheet number -- a standard, documented gap in ZUPT/SHOE deployments
        generally). The default of 20.0 was empirically swept against ground-truth-labeled
        stationary/moving data: below ~5 the detector never fires; 5-45 never misfires during
        genuine motion, with detection latency improving only marginally past 20; at 50 it
        starts misfiring during real motion. 20.0 sits at the elbow of that latency-vs-margin
        tradeoff. This is evidence from one dataset with one stationary event, not a
        guarantee it generalizes -- recalibrate the same way if deploying elsewhere. (A
        follow-up push to 30.0, to react faster through touchdown bouncing, was tried and
        reverted: detection timing barely changed, but the downstream self-referential-
        triangulation feedback loop turned out to be sensitive to exactly when the binary
        flip happens, and the 60s diagnostic's final error roughly doubled. Lesson: always
        re-run the full diagnostic, not just this detector's own sweep, before keeping a
        tuning change here.)

        Hysteresis: a real stationary period can still have brief vibration blips that push
        the instantaneous test above threshold. Once active, ZUPT requires zupt_hold_seconds
        of *continuous* (time-based, so this stays correct regardless of call frequency)
        non-stationary readings before deactivating, so a blip doesn't reopen the
        self-referential vision divergence this detector exists to prevent.
        """
        if len(self._imu_history) < self._imu_history.maxlen:
            self._zupt_active = False
            return

        window_size = len(self._imu_history)
        gyro_hist = np.array([g for g, _ in self._imu_history]) - self.state.b_g
        accel_hist = np.array([a for _, a in self._imu_history]) - self.state.b_a

        accel_mean = accel_hist.mean(axis=0)
        gravity_direction = accel_mean / np.linalg.norm(accel_mean)
        accel_reference = GRAVITY_MAGNITUDE * gravity_direction

        # continuous-time noise density -> discrete per-sample variance: sigma^2 = density^2/dt
        sigma_gyro_sq = self.noise_params["gyro_noise"] ** 2 / dt * self.zupt_noise_inflation
        sigma_accel_sq = self.noise_params["accel_noise"] ** 2 / dt * self.zupt_noise_inflation

        test_statistic = (np.sum(gyro_hist ** 2) / sigma_gyro_sq
                           + np.sum((accel_hist - accel_reference) ** 2) / sigma_accel_sq)
        # dof: 6 raw scalars (3 gyro + 3 accel) per sample, minus 2 for the gravity
        # direction's 2 free parameters (a unit vector) estimated via ML from this same window
        dof = 6 * window_size - 2
        instantaneous = test_statistic < chi_square_threshold(dof, self.zupt_confidence)

        if instantaneous:
            self._zupt_hold_time = 0.0
            self._zupt_active = True
        elif self._zupt_active:
            self._zupt_hold_time += dt
            if self._zupt_hold_time > self.zupt_hold_seconds:
                self._zupt_active = False

    def process_image(self, timestamp, image):
        """Track features, clone the current pose, use any now-ready tracks, then marginalize.

        Returns (n_updates_this_step, n_rejected_this_step).
        """
        previously_active = set(self.tracker.active_ids.tolist())
        self.tracker.process_frame(image, timestamp)
        self.state = augment(self.state, self.T_BS_cam0)
        self.clone_frame_ids.append(timestamp)

        just_ended = previously_active - set(self.tracker.active_ids.tolist())

        n_to_evict = max(0, self.state.n_clones - self.max_clones)
        doomed_timestamps = set(self.clone_frame_ids[:n_to_evict]) if n_to_evict > 0 else set()

        n_updates, n_rejected = self._process_ready_tracks(just_ended, doomed_timestamps)

        if n_to_evict > 0:
            for i in range(n_to_evict):
                self.trajectory_history.append((self.clone_frame_ids[i], self.state.clone_positions[i].copy(),
                                                 self.state.clone_orientations[i].copy()))
            self.state = marginalize_clones(self.state, range(n_to_evict))
            self.clone_frame_ids = self.clone_frame_ids[n_to_evict:]

        return n_updates, n_rejected

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

            # a track that ended naturally can afford to require a nicer minimum. A still-
            # active track forced out early by eviction (touches_doomed_clone, not just_ended)
            # has no such luxury -- its observations are gone the moment we skip them -- so it
            # uses whatever it's got, down to the bare minimum of 2 views needed to
            # triangulate. Otherwise long-lived tracks get re-triggered every frame with ~1
            # new observation each time (never enough to clear min_track_length), silently
            # discarding their most valuable asset -- long baseline -- one frame at a time.
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
                                     self.observation_noise_std, min_variance=self.min_variance)
        self.n_updates_applied += n_updates
        self.n_tracks_rejected += n_rejected
        return n_updates, n_rejected

    def _build_measurement(self, usable, clone_index_of):
        clone_indices = [clone_index_of[t] for t, _ in usable]
        camera_poses = [(self.state.clone_orientations[i], self.state.clone_positions[i]) for i in clone_indices]
        bearings = [undistort_normalized([[u, v]], self.K, self.dist_coeffs)[0] for _, (u, v) in usable]
        try:
            X_est = triangulate_and_validate(camera_poses, bearings, self.min_parallax,
                                              self.max_reprojection_error)
            if X_est is None:
                return None, None
            r, H_x, H_f = stack_feature_observations(X_est, camera_poses, bearings, clone_indices,
                                                      n_clones=self.state.n_clones)
            return null_space_project(r, H_x, H_f)
        except np.linalg.LinAlgError:
            # degenerate geometry the SVD/QR itself couldn't handle
            return None, None
