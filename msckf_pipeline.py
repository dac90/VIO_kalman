"""Stage 8: the full MSCKF pipeline, wiring IMU propagation, feature
tracking, triangulation, the measurement update, and marginalization
together into one continuously-runnable system.

A feature's observation history is used in an EKF update when its track
naturally ends (KLT stops tracking it) -- that's the common case, and it
gets to use its whole accumulated history at once.

max_clones is a strict, always-enforced cap: the sliding window is
marginalized back down to it every single frame, no exceptions. A still-
active track whose oldest unprocessed observation is about to be evicted is
force-processed with whatever it's accumulated so far (down to a bare
minimum of 2 views), rather than losing that data outright.

A trio of stationary-only pseudo-measurements -- ZUPT (zero-velocity),
ZARU (zero angular-rate), and gravity/tilt alignment, all in msckf_update.py
-- is applied whenever _update_stationary_state() detects the platform is at
rest from raw IMU statistics. Found necessary after the user noticed (from
the video demo) that this pipeline's residual long-run divergence began right
when the drone in the test window lands and holds still: a stationary camera
gets zero parallax on everything it sees, so vision updates go from merely
unhelpful to actively harmful (the same self-referential triangulation
problem below, but with no real corrective signal available at all to
counteract it), while unaided dead-reckoning drifts on whatever residual bias
remains. See _update_stationary_state's docstring for why this needs its own
detector (fooled once by the filter's own drifting bias estimate already) and
zero_velocity_update's docstring for why *that specific* update's mean
correction is restricted to the velocity block only (fooled once by its own
cross-correlation with gyro bias) -- ZARU and gravity-alignment don't need
that restriction, since they measure bias/tilt directly rather than through a
correlation, which is also why they were added: a bare ZUPT has no way to
correct the residual accel bias actually driving the drift it's trying to
stop, so position kept creeping even while ZUPT was firing correctly the
whole time. Both this and the stationary detector itself now run once per
IMU sample (200Hz) rather than once per camera frame (20Hz) as an earlier
version did, since none of this depends on camera frames at all -- reacting
to a stop up to 10x faster.

_update_stationary_state also drives a second, continuous mechanism (1c):
vision's observation noise (used in _process_ready_tracks) is smoothly
inflated in proportion to how stationary recent IMU data looks, tracking the
same GLRT statistic that drives the binary ZUPT/ZARU/tilt gate but without
that gate's hysteresis. Added because touchdown bouncing/settling is an
ambiguous middle ground the binary gate structurally can't react to quickly:
real settling vibration keeps the instantaneous test from passing for a
while after net translation is already ~0, and pushing the gate to trigger
earlier by further tuning zupt_noise_inflation was tried and found to make
things *worse* overall (see _update_stationary_state's docstring) via the
self-referential-triangulation loop's sensitivity to exactly when that
binary flip happens. Vision's trust doesn't need a firm yes/no the way
asserting v=0 does, so it can fade continuously instead.

The same GLRT statistic also softens ZUPT/ZARU/gravity-alignment themselves
(the mirror image of 1c): each one's noise std is scaled up when the
*current* instantaneous reading looks less confidently stationary than the
hysteresis-latched _zupt_active flag implies (e.g. mid-blip, when hysteresis
is bridging a brief burst of real-looking motion) -- so a correction applied
while _zupt_active is only true because of hysteresis carries proportionally
less weight than one applied while the signal actually looks quiet right now.

A second trigger for _zupt_active comes from vision itself
(_check_vision_zero_parallax, called from process_image): if every currently
active feature track shows essentially no parallax (the same check_parallax
test that gates triangulation quality, read here as "the camera hasn't
translated meaningfully"), that's supporting evidence of stationarity from a
completely different sensor. This can only ever force _zupt_active on, never
off, and even then only when the IMU test isn't confidently reading "moving"
(self._stationarity_ratio below zero_parallax_max_imu_ratio) -- it corroborates
the IMU signal rather than overriding it. That gate is load-bearing, not
cosmetic: checked directly against this dataset's real forward flight
(~0.4-0.7 m/s), the *majority* of active tracks routinely show zero parallax
simultaneously, because tracks near the camera's direction of travel (the
focus of expansion) have bearing rays nearly parallel to the translation
vector -- check_parallax's own documented blind spot (translation *along* a
bearing ray is invisible to it) -- regardless of how fast the platform is
actually moving. "All active tracks lack parallax" is therefore not, on its
own, reliable evidence of stationarity for a forward-facing camera; it only
becomes reliable once the IMU has already ruled out the "fast, low-rotation
translation" explanation for the same symptom.

An earlier version of this pipeline made max_clones a *soft* cap instead --
letting the window grow (up to 3x) rather than force-process a still-active
track early. That was found (via long-run divergence debugging) to be a
significant net harm, not a fix: a longer window means features get
triangulated against increasingly stale camera-pose estimates that have
already drifted from truth, and Gauss-Newton refinement fits the feature to
those poses by construction, giving a small, "consistent-looking" residual
even when the poses themselves are wrong. The EKF update then reads as
highly informative and shrinks P accordingly, reinforcing the existing
drift instead of correcting it -- a self-fulfilling feedback loop that gets
worse the longer the window is left to grow. A small, strictly bounded
window (matching canonical MSCKF-VIO practice) starves that feedback loop of
the staleness it needs; combined with triangulation.py's parallax and
cheirality checks (which now handle the force-processed, thin-observation
tracks that motivated the soft cap in the first place), this measured
~25x lower position error at t=20s and pushed honest stable operation well
past that mark in testing. See demos/demo_full_pipeline.py's docstring for
current validated numbers.
"""
from collections import deque

import numpy as np

from feature_tracker import FeatureTracker
from imu_propagation import GRAVITY_WORLD
from measurement_model import null_space_project, stack_feature_observations
from msckf_state import MSCKFState, augment, marginalize_clones, propagate
from msckf_update import chi_square_threshold, ekf_update, gravity_alignment_update, passes_chi_square_gate, \
    zero_angular_rate_update, zero_velocity_update
from triangulation import check_parallax, triangulate_and_validate, undistort_normalized

GRAVITY_MAGNITUDE = np.linalg.norm(GRAVITY_WORLD)

# Covariance floor (std, per IMU error-state block) -- see ekf_update's docstring
# in msckf_update.py. Empirically swept (demos/demo_divergence_analysis.py) rather
# than picked from first principles: values an order of magnitude looser than
# these (closer to INITIAL_SIGMA/10) made the long-run divergence *worse*, not
# better -- keeping P artificially open raises the Kalman gain for *every*
# subsequent measurement, including the bad self-referential ones, so a floor
# that's too generous makes the state more susceptible to corruption, not less.
# These values (empirically ~0.4x that naive first guess) were the sweet spot
# found by sweeping: they roughly halve position error at t=30s and cut it
# ~5x at t=50s versus no floor at all, at the cost of a small (~0.5m) regression
# at exactly t=20s. They don't eliminate eventual divergence past t~60s.
DEFAULT_MIN_SIGMA = dict(theta=4e-4, b_g=4e-4, v=4e-3, b_a=4e-3, p=4e-4)


class MSCKFPipeline:
    """Owns the filter state, feature tracker, and per-track bookkeeping.

    Call process_imu for every IMU sample (in timestamp order) and
    process_image for every camera frame, interleaved in timestamp order.
    """

    def __init__(self, p0, v0, q0, b_g0, b_a0, P0, T_BS_cam0, K, dist_coeffs, noise_params,
                 max_clones=20, max_features=150,
                 observation_noise_std=5.0 / 458.0,  # ~5px: 1px was overconfident once triangulation
                                                      # uses the filter's own (imperfect) poses instead
                                                      # of ground truth -- see the divergence this caused
                                                      # when tuning this pipeline over long runs
                 min_track_length=6, gate_confidence=0.95,
                 min_parallax=0.2,  # meters of camera translation orthogonal to a feature's
                                    # bearing ray -- MSCKF-VIO's default; below this, depth
                                    # along that ray is unobservable and triangulation is
                                    # degenerate regardless of what the chi-square gate thinks
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
                 zupt_update_interval=0.05,  # s; corrections apply at most this often -- see
                                              # process_imu's docstring for why detection and
                                              # application run at different rates
                 vision_trust_max_inflation=50.0,  # cap on how much _update_stationary_state can
                                                    # inflate vision's observation noise std when
                                                    # recent IMU excitation looks low -- see its
                                                    # docstring. UNTUNED: a reasonable starting
                                                    # order of magnitude (matches zupt_noise_inflation's
                                                    # scale), not yet empirically swept the way that
                                                    # parameter was.
                 stationary_uncertainty_max_inflation=10.0,  # cap on how much ZUPT/ZARU/tilt's own
                                                              # noise stds loosen when _zupt_active is
                                                              # only true via hysteresis, not the current
                                                              # instantaneous reading -- see docstring
                                                              # below. Also UNTUNED.
                 zero_parallax_min_tracks=5,  # minimum evaluable active tracks required before
                                               # "all active tracks lack parallax" counts as
                                               # evidence of stationarity -- see
                                               # _check_vision_zero_parallax's docstring
                 zero_parallax_max_imu_ratio=20.0):  # the vision zero-parallax signal can only
                                                      # trigger _zupt_active while the IMU-side
                                                      # stationarity_ratio is already below this --
                                                      # see _check_vision_zero_parallax's docstring
                                                      # for why this gate is load-bearing, not
                                                      # optional. >4x margin below the ~70-150
                                                      # ratios seen during real forward flight on
                                                      # this dataset; still generous room above the
                                                      # ~1-10 range seen approaching a real stop.
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
        self.vision_trust_max_inflation = vision_trust_max_inflation
        self.stationary_uncertainty_max_inflation = stationary_uncertainty_max_inflation
        self.zero_parallax_min_tracks = zero_parallax_min_tracks
        self.zero_parallax_max_imu_ratio = zero_parallax_max_imu_ratio
        self._imu_history = deque(maxlen=zupt_window_samples)
        self._zupt_active = False
        self._zupt_hold_time = 0.0
        self._time_since_last_stationary_update = 0.0
        self._vision_noise_scale = 1.0
        self._stationary_uncertainty_scale = 1.0
        self._stationarity_ratio = None

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

        Detection (_update_stationary_state) runs every IMU sample (200Hz on this
        dataset) so a stop gets *noticed* as fast as possible. Applying the corrections
        themselves at that same rate turned out to be actively harmful, though: consecutive
        accelerometer/gyro samples 5ms apart are not independent draws (real sensor noise
        and real vibration are correlated over such short gaps), so treating ~200
        corrections a second as that many independent fresh measurements manufactures far
        more confidence than the data actually supports -- a real incident on this dataset
        had this collapse P so hard that a small theta/b_a estimation ambiguity inherent to
        gravity_alignment_update (a small tilt error and a small accel-bias error affect the
        predicted reading almost identically) got "locked in" wrong and then compounded for
        the ~24s of the stationary hold, leaving accel bias error 10x worse than before this
        fix and the subsequent resumed-motion phase catastrophically worse than with no
        stationary corrections at all. Applying the corrections themselves at a bounded rate
        (zupt_update_interval, default 20Hz -- the same cadence validated before detection
        was moved to IMU rate) avoids that overcounting while keeping the fast reaction time.
        """
        self.state = propagate(self.state, gyro, accel, dt, self.noise_params)
        if not self.enable_zupt:
            return
        self._imu_history.append((np.asarray(gyro), np.asarray(accel)))
        self._update_stationary_state(dt)

        self._time_since_last_stationary_update += dt
        if self._zupt_active and self._time_since_last_stationary_update >= self.zupt_update_interval:
            self._time_since_last_stationary_update = 0.0
            # softened by _stationary_uncertainty_scale (>1 whenever the *current* instantaneous
            # reading looks less stationary than the hysteresis-latched flag implies) -- see
            # _update_stationary_state's docstring
            scale = self._stationary_uncertainty_scale
            self.state = zero_velocity_update(self.state, self.zupt_noise_std * scale,
                                               min_variance=self.min_variance)
            self.state = zero_angular_rate_update(self.state, gyro, self.zaru_noise_std * scale,
                                                   min_variance=self.min_variance)
            self.state = gravity_alignment_update(self.state, accel, self.tilt_noise_std * scale,
                                                   min_variance=self.min_variance)

    def _update_stationary_state(self, dt):
        """Update self._zupt_active using a GLRT (generalized likelihood-ratio test) on raw
        IMU statistics -- deliberately independent of the filter's own pose/velocity estimate
        (unlike triangulation.py's parallax check, which uses the filter's *own* camera-clone
        poses and so can be fooled: once the filter has accumulated some drift, it can
        believe a stationary scene has real parallax, since its own poses appear to have
        moved even though the camera never did). Called once per IMU sample (200Hz on this
        dataset), not once per camera frame (20Hz) as an earlier version did -- detecting and
        reacting to a stop up to 10x faster, since ZUPT/ZARU/gravity-alignment are pure
        IMU-side corrections with no dependency on camera frames at all.

        This is the SHOE ("stance hypothesis optimal estimation") detector from Skog et al.,
        "Zero-Velocity Detection -- An Algorithm Evaluation" (IEEE Trans. Biomedical
        Engineering, 2010) -- standard in ZUPT-aided INS -- rather than a hand-tuned
        threshold on the raw signal's mean/variance (an earlier version of this method used
        exactly that, and it worked, but every threshold was picked by eyeballing this one
        dataset's numbers with no principled way to know if they'd generalize). The GLRT
        compares two hypotheses per window: H0 (stationary: gyro reads pure noise, accel
        reads gravity from a fixed but unknown direction) vs H1 (moving: no such structure).
        Under H0, each raw residual, normalized by the IMU's own characterized noise density
        (self.noise_params, discretized as density^2/dt), is approximately a unit-variance
        Gaussian, so the summed squared residuals over the window are approximately
        chi-square distributed -- letting the existing chi_square_threshold utility (already
        used for vision-outlier gating) supply a single principled zupt_confidence parameter
        in place of three separately hand-tuned magic numbers.

        Two adaptations from the textbook SHOE detector, both because this IMU's biases are
        too large to ignore (unlike the noise-only model the original derivation assumes):
        gyro readings are corrected by the filter's own b_g estimate before testing (raw
        gyro is dominated by a ~0.08 rad/s constant bias here -- without this correction, a
        brief window of smooth, low-variance flight reads as falsely "stationary", since a
        real but steady rotation rate is easily mistaken for bias); accel readings are
        similarly corrected by b_a. The gravity *direction* itself is still estimated fresh
        from each window's own mean (the textbook approach) rather than from the filter's
        orientation estimate, specifically to avoid the same self-referential trap the
        parallax check has -- this detector shouldn't need to trust the filter's own
        attitude belief to work.

        zupt_noise_inflation exists because the manufacturer's noise-density spec
        (self.noise_params, from imu0/sensor.yaml) is an idealized white-noise number that
        real sensor data doesn't actually live up to -- checked directly on this dataset: the
        raw test statistic during a period ground truth confirms is genuinely stationary came
        out 1x-17x *larger* than chi_square_threshold at zupt_confidence=0.95 using the
        noise density as-is (residual bias-correction error, real mounting vibration even at
        rest, and non-white noise all inflate real variance beyond the datasheet number, and
        none of that is specific to this sensor -- it's a standard, documented gap in
        ZUPT/SHOE deployments generally, not a derivation error here).

        The default of 20.0 was empirically tuned against ground truth (the only labeled
        stationary/moving data available here): swept 1-50, using ground-truth velocity to
        label every IMU sample as truly stationary (|v|<0.02 m/s) or clearly moving
        (|v|>0.1 m/s), and running the full pipeline once per candidate value. Below ~5, the
        detector never fires at all (too strict to ever accept real IMU data as "noise
        only"). From 5 up through 45, it never once fires during genuine motion anywhere in
        a 60s window spanning real flight, landing, the ~24s hold, and the resumed takeoff,
        while detection latency at the real stationary onset improves only marginally with
        further inflation (20.4s at 5, down to 18.7s at 45 -- most of that gain is already
        captured by 20). At 50, it starts firing during the early takeoff ramp (t=42.2s,
        true |v|=0.10 m/s) -- a real false positive, actively harmful, since it would apply
        ZUPT/ZARU/gravity-alignment against a platform that's actually accelerating. 20.0
        sits at the "elbow" of the latency-vs-margin tradeoff: essentially all the reachable
        latency improvement, with better than 2x margin below the observed failure point.
        This is one dataset with one real stationary event, though -- it's evidence, not a
        guarantee this generalizes to a different platform or IMU; recalibrate the same way
        if deploying elsewhere.

        Tried pushing this to 30.0 to react faster through touchdown bouncing (still >1.5x
        margin below the failure point) -- reverted. The detected active windows barely
        moved (within ~0.1-0.2s of the 20.0 windows), but the 60s diagnostic's final position
        error nearly doubled (22m vs 10.6m) and its 3-sigma escape point moved from t=51.6s
        to t=17.6s, *before* the real landing even starts. Since the stationary-window timing
        itself was nearly identical, the regression isn't really about this detector at all --
        it's the downstream self-referential-triangulation feedback loop's known sensitivity
        to small timing perturbations (see demo_full_pipeline.py's docstring) reacting badly
        to a slightly different bias trajectory. A cautionary data point for tuning this
        value by feel: a change that looks like a strict improvement on the metric it directly
        targets (detection latency) can still make the overall filter worse through a
        different, indirect mechanism -- always re-run the full diagnostic, not just the
        detector's own sweep, before keeping a new value.

        Hysteresis: a real stationary period (e.g. landed, motors idling) can still have
        brief vibration blips that push the instantaneous test above threshold for a
        second or two. A first version without hysteresis flagged "moving" during one such
        blip and never went back to "stationary" until real motion actually resumed tens of
        seconds later -- reopening exactly the self-referential vision divergence this
        detector exists to prevent, for that whole gap. Once active, ZUPT now requires
        zupt_hold_seconds of *continuous* (time-based, not a raw check count -- so this
        stays correct regardless of how often this method gets called) non-stationary
        readings before deactivating.

        1c: this method also sets self._vision_noise_scale, a *continuous* multiplier on
        vision's observation noise std (applied in _process_ready_tracks), separate from and
        not hysteresis-gated like self._zupt_active. Motivation: ZUPT/ZARU/gravity-alignment
        assert specific pseudo-measurements (v=0, omega=b_g, a=g), so they need a firm,
        sticky yes/no decision -- there's no such thing as "50% asserting v=0". Vision's
        trust, on the other hand, is already just a noise parameter, so it can track the
        *current* excitation level immediately and smoothly instead of waiting for a
        hysteresis-protected verdict. This targets the touchdown-bounce case specifically:
        real settling vibration keeps the binary GLRT from firing for a while after net
        translation is already ~0 (and pushing zupt_noise_inflation higher to force an
        earlier binary trigger was tried and reverted -- see above -- since the downstream
        self-referential-triangulation feedback loop turned out to be surprisingly sensitive
        to exactly when that discrete flip happens). A continuous vision-trust scale
        sidesteps needing that flip to happen at the "right" instant at all: during the
        bounce, vision gets partially discounted in proportion to how quiet the IMU
        currently looks, and by the time the binary detector does fire, vision has typically
        already been mostly (not suddenly) sidelined.

        Reuses the same GLRT statistic as the binary test rather than a separate heuristic:
        stationarity_ratio = test_statistic / chi_square_threshold(...) is <1 exactly when
        the instantaneous check would pass. self._vision_noise_scale is a Lorentzian-shaped
        falloff in that ratio: 1 + (vision_trust_max_inflation - 1) / (1 + ratio^2) -- equal
        to the max cap at ratio=0 (perfectly still) and decaying smoothly to 1.0 (no extra
        distrust) as ratio grows, already bounded to [1, vision_trust_max_inflation] without
        needing an explicit clip.

        Deliberately *not* 1/ratio clipped to [1, max] (the first version of this, before it
        was checked against real data): that formula only starts discounting once ratio is
        already below 1 -- i.e. only once the instantaneous check would already pass --
        which is exactly the same instant the binary gate fires, adding nothing beforehand.
        Checked directly against this dataset's real touchdown: ratio descends through a real
        gradient (~55 -> ~25 -> ~2.9 -> ~2.6 -> ~2.2 -> ~1.1) over about a second *before*
        crossing 1.0, and during clearly-moving flight it typically sits at 10-800+ (median
        ~110). Squaring ratio in the denominator keeps that typical-flight regime essentially
        undiscounted (ratio=110 -> scale~1.004) while already meaningfully discounting by
        ratio~3-6, roughly half a second to a second ahead of the binary trigger -- letting
        vision fade out ahead of, not simultaneously with, that discrete flip.

        This method also sets self._stationary_uncertainty_scale, the mirror image of
        _vision_noise_scale applied to ZUPT/ZARU/gravity-alignment's own noise stds
        (process_imu) instead of vision's. Motivation: self._zupt_active can be true purely
        because hysteresis is bridging a brief blip (see above) while the *current*
        instantaneous ratio actually looks like real motion -- in that moment, asserting
        v=0/omega=b_g/a=g with full, undiscounted confidence is exactly backwards; those
        pseudo-measurements should carry proportionally less weight the less the current
        instant actually looks stationary, even while the sticky flag stays on. Same
        underlying ratio, opposite shape from vision's: 1 + (stationary_uncertainty_max_inflation
        - 1) * ratio^2/(1+ratio^2) -- 1.0 (no loosening) at ratio=0 (currently looks perfectly
        still), growing toward the cap as ratio grows past 1 (currently looks like real
        motion, hysteresis is doing all the work of keeping this active). Also naturally
        bounded to [1, stationary_uncertainty_max_inflation] with no explicit clip needed.
        UNTUNED -- added on the same reasoning as 1c but not yet checked against ground
        truth the way zupt_noise_inflation and vision_trust_max_inflation's shape were.
        """
        if len(self._imu_history) < self._imu_history.maxlen:
            self._zupt_active = False
            self._vision_noise_scale = 1.0
            self._stationary_uncertainty_scale = 1.0
            self._stationarity_ratio = None
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
        threshold = chi_square_threshold(dof, self.zupt_confidence)
        stationarity_ratio = test_statistic / threshold
        self._stationarity_ratio = stationarity_ratio
        instantaneous = stationarity_ratio < 1.0

        if instantaneous:
            self._zupt_hold_time = 0.0
            self._zupt_active = True
        elif self._zupt_active:
            self._zupt_hold_time += dt
            if self._zupt_hold_time > self.zupt_hold_seconds:
                self._zupt_active = False

        self._vision_noise_scale = 1.0 + (self.vision_trust_max_inflation - 1.0) / (1.0 + stationarity_ratio ** 2)
        excitation = stationarity_ratio ** 2 / (1.0 + stationarity_ratio ** 2)
        self._stationary_uncertainty_scale = 1.0 + (self.stationary_uncertainty_max_inflation - 1.0) * excitation

    def _check_vision_zero_parallax(self):
        """Secondary, vision-native trigger for self._zupt_active, corroborating (not
        replacing) the IMU-only GLRT above: if every currently active feature track shows
        essentially no parallax -- the same check_parallax test that gates triangulation
        quality (triangulation.py), read here as "the camera hasn't translated meaningfully"
        rather than "this feature isn't triangulable" -- that's supporting evidence of
        stationarity from a completely different sensor. Runs once per camera frame (this is
        inherently vision-rate; there's no camera data to check at IMU rate), right after
        process_image tracks/augments, before it processes tracks for updates.

        Two guards, both load-bearing (found necessary by checking this against real data,
        not assumed up front):

        1. Requires at least zero_parallax_min_tracks evaluable active tracks (an active
           track with fewer than 2 observations in the known clone window can't be checked
           at all). Below that, this stays silent rather than treating a near-empty active
           set as evidence either way -- a tracker that's temporarily lost most of its
           features is far more likely a tracking hiccup than a scene that stopped moving.

        2. Requires self._stationarity_ratio (the IMU test's own statistic, from
           _update_stationary_state) to already be below zero_parallax_max_imu_ratio --
           i.e. the IMU isn't confidently reading "moving". Checked directly against this
           dataset's real forward flight (~0.4-0.7 m/s) and found essential, not optional: a
           *majority* of active tracks routinely read "no parallax" simultaneously during
           ordinary forward motion, because tracks near the camera's direction of travel
           (the focus of expansion) have bearing rays nearly parallel to the translation
           vector -- check_parallax's own documented blind spot (translation *along* a
           bearing ray is invisible to it), regardless of real speed. Without this guard,
           the very first fast-forward-flight segment of a real run falsely triggered ZUPT.
           The IMU ratio stayed reliably >60 throughout that same segment (vs. ~1-10
           approaching a real stop), so requiring it below 20 filters out exactly this false-
           positive class while leaving real, IMU-quiet approaches free to benefit.

        Can only ever force self._zupt_active *on* (mirroring exactly what the IMU test's own
        "instantaneous=True" branch does -- same _zupt_hold_time reset, so hysteresis behaves
        identically regardless of which signal triggered it), never off: even with guard 2, a
        parallax-only signal says nothing about rotation (a genuinely rotating-but-not-
        translating platform could still pass this check while ZARU/gravity-alignment's "not
        rotating" assumption is false), so it stays a one-directional corroboration, never an
        override of the IMU test's "moving" verdict.
        """
        if self._stationarity_ratio is None or self._stationarity_ratio >= self.zero_parallax_max_imu_ratio:
            return

        clone_index_of = {t: i for i, t in enumerate(self.clone_frame_ids)}
        evaluable = 0
        for track_id in self.tracker.active_ids:
            observations = [(t, uv) for t, uv in self.tracker.tracks[int(track_id)] if t in clone_index_of]
            if len(observations) < 2:
                continue
            clone_indices = [clone_index_of[t] for t, _ in observations]
            camera_poses = [(self.state.clone_orientations[i], self.state.clone_positions[i])
                             for i in clone_indices]
            bearings = [undistort_normalized([[u, v]], self.K, self.dist_coeffs)[0] for _, (u, v) in observations]
            evaluable += 1
            if check_parallax(camera_poses, bearings, self.min_parallax):
                return  # at least one active track has real parallax -- not a parallax-frozen scene

        if evaluable >= self.zero_parallax_min_tracks:
            self._zupt_active = True
            self._zupt_hold_time = 0.0

    def process_image(self, timestamp, image):
        """Track features, clone the current pose, use any now-ready tracks, then marginalize.

        Returns (n_updates_this_step, n_rejected_this_step).
        """
        previously_active = set(self.tracker.active_ids.tolist())
        self.tracker.process_frame(image, timestamp)
        self.state = augment(self.state, self.T_BS_cam0)
        self.clone_frame_ids.append(timestamp)

        if self.enable_zupt:
            self._check_vision_zero_parallax()

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

        # 1c: inflate vision's assumed noise (hence shrink its Kalman gain) in proportion to
        # how stationary recent IMU data looks -- see _update_stationary_state's docstring.
        # 1.0 when enable_zupt is off or recent motion looks clearly real; grows smoothly as
        # the platform looks quieter, well before (and independent of) the binary ZUPT gate.
        effective_noise_std = self.observation_noise_std * self._vision_noise_scale

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

            if passes_chi_square_gate(r_o, H_o, self.state.P, effective_noise_std, self.gate_confidence):
                r_o_batch.append(r_o)
                H_o_batch.append(H_o)
            else:
                n_rejected += 1

        n_updates = len(r_o_batch)
        if r_o_batch:
            self.state = ekf_update(self.state, np.concatenate(r_o_batch), np.concatenate(H_o_batch, axis=0),
                                     effective_noise_std, min_variance=self.min_variance)
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
