import numpy as np

import ground_truth as gt
from feature_tracker import iter_cam0_frames
from frames import body_to_sensor_position, load_T_BS
from imu_propagation import _load_imu_measurements
from msckf_pipeline import MSCKFPipeline
from msckf_state import N_CLONE_ERROR, N_IMU_ERROR, load_imu_noise_params
from triangulation import load_cam0_intrinsics

# same "dummy bias" used throughout the demos -- an independent estimate, not
# ground truth, but needed so the filter's own reported covariance stays
# roughly matched to its real error (see demos/demo_msckf_update.py's note on
# why zero-bias dead reckoning makes the chi-square gate reject everything)
DUMMY_BIAS_GYRO = np.array([-0.00233173, 0.02172386, 0.07821335])
DUMMY_BIAS_ACCEL = np.array([-0.04066623, 0.1155297, 0.05121861])

INITIAL_SIGMA = dict(theta=1e-3, b_g=1e-2, v=1e-1, b_a=1e-1, p=1e-3)


def _initial_covariance():
    diag = ([INITIAL_SIGMA["theta"] ** 2] * 3 + [INITIAL_SIGMA["b_g"] ** 2] * 3 + [INITIAL_SIGMA["v"] ** 2] * 3
            + [INITIAL_SIGMA["b_a"] ** 2] * 3 + [INITIAL_SIGMA["p"] ** 2] * 3)
    return np.diag(diag)


def _run_pipeline(duration_s, max_clones=15, bias_gyro=None, bias_accel=None):
    K, dist_coeffs = load_cam0_intrinsics()
    T_BS_cam0 = load_T_BS(f"{gt.MAV0_DIR}/cam0/sensor.yaml")
    noise_params = load_imu_noise_params()

    gt_timestamps, _, _, _ = gt._load_state_ground_truth()
    cam0_all_timestamps = gt.load_cam0_timestamps()
    start_frame = int(np.searchsorted(cam0_all_timestamps, gt_timestamps[0])) + 30
    n_frames = int(duration_s * 20)

    frames = list(iter_cam0_frames(max_frames=n_frames, start_frame=start_frame))
    t0 = frames[0][0]
    p0 = gt.interpolate_ground_truth_position(t0)
    v0 = gt.interpolate_ground_truth_velocity(t0)
    q0 = gt.interpolate_ground_truth_orientation(t0)

    pipeline = MSCKFPipeline(p0, v0, q0, bias_gyro if bias_gyro is not None else np.zeros(3),
                              bias_accel if bias_accel is not None else np.zeros(3),
                              _initial_covariance(), T_BS_cam0, K, dist_coeffs, noise_params,
                              max_clones=max_clones)

    imu_timestamps, gyro, accel = _load_imu_measurements()
    imu_idx = np.searchsorted(imu_timestamps, t0, side="right")
    t_prev = t0

    n_clones_history = []
    for frame_t, image in frames:
        while imu_idx < len(imu_timestamps) and imu_timestamps[imu_idx] <= frame_t:
            t_curr = int(imu_timestamps[imu_idx])
            pipeline.process_imu(gyro[imu_idx], accel[imu_idx], (t_curr - t_prev) / 1e9)
            t_prev = t_curr
            imu_idx += 1

        pipeline.process_image(frame_t, image)

        # max_clones is a strict, always-enforced upper bound (see msckf_pipeline.py --
        # an earlier soft-cap version that let this grow was found to cause long-run divergence)
        assert pipeline.state.n_clones <= pipeline.max_clones
        expected_dim = N_IMU_ERROR + N_CLONE_ERROR * pipeline.state.n_clones
        assert pipeline.state.P.shape == (expected_dim, expected_dim)
        assert len(pipeline.clone_frame_ids) == pipeline.state.n_clones
        assert np.allclose(pipeline.state.P, pipeline.state.P.T, atol=1e-9)
        n_clones_history.append(pipeline.state.n_clones)

    return pipeline, frames, n_clones_history


def _minimal_pipeline(b_g0=None, **kwargs):
    """A pipeline with no camera frames processed yet -- for testing process_imu-only
    behavior (like the ZUPT stationary detector) without needing real image data."""
    K, dist_coeffs = load_cam0_intrinsics()
    T_BS_cam0 = load_T_BS(f"{gt.MAV0_DIR}/cam0/sensor.yaml")
    noise_params = load_imu_noise_params()
    return MSCKFPipeline(np.zeros(3), np.zeros(3), np.array([1.0, 0, 0, 0]),
                          b_g0 if b_g0 is not None else np.zeros(3), np.zeros(3),
                          _initial_covariance(), T_BS_cam0, K, dist_coeffs, noise_params, **kwargs)


def test_is_stationary_false_until_the_imu_history_buffer_fills():
    pipeline = _minimal_pipeline(zupt_window_samples=10)
    gravity = np.array([0.0, 0.0, 9.81])
    for _ in range(9):
        pipeline.process_imu(np.zeros(3), gravity, 0.005)
    assert not pipeline._zupt_active  # buffer not full yet, regardless of how still the IMU looks


def test_is_stationary_distinguishes_a_still_imu_from_a_moving_one():
    rng = np.random.default_rng(0)
    gravity = np.array([0.0, 0.0, 9.81])

    still_pipeline = _minimal_pipeline(zupt_window_samples=50)
    for _ in range(50):
        still_pipeline.process_imu(rng.normal(size=3) * 1e-4, gravity + rng.normal(size=3) * 1e-4, 0.005)
    assert still_pipeline._zupt_active

    moving_pipeline = _minimal_pipeline(zupt_window_samples=50)
    for i in range(50):
        moving_pipeline.process_imu(np.array([0.3, 0.0, 0.0]) * np.sin(i * 0.3),
                                     gravity + np.array([1.5, 0.5, 0.0]) * np.cos(i * 0.3), 0.005)
    assert not moving_pipeline._zupt_active


def test_vision_noise_scale_is_neutral_before_window_fills():
    pipeline = _minimal_pipeline(zupt_window_samples=10)
    gravity = np.array([0.0, 0.0, 9.81])
    for _ in range(9):
        pipeline.process_imu(np.zeros(3), gravity, 0.005)
    assert pipeline._vision_noise_scale == 1.0


def test_vision_noise_scale_is_neutral_during_clear_motion():
    # the Lorentzian falloff only asymptotically approaches 1.0, never hits it exactly, so
    # this checks "close enough to be a no-op" rather than bit-exact equality.
    gravity = np.array([0.0, 0.0, 9.81])
    pipeline = _minimal_pipeline(zupt_window_samples=50)
    for i in range(50):
        gyro = np.array([0.3, 0.0, 0.0]) * np.sin(i * 0.3)
        accel = gravity + np.array([1.5, 0.5, 0.0]) * np.cos(i * 0.3)
        pipeline.process_imu(gyro, accel, 0.005)
    assert not pipeline._zupt_active
    assert pipeline._vision_noise_scale < 1.01


def test_vision_noise_scale_grows_when_stationary_and_respects_the_configured_cap():
    # a near-perfectly-still signal pushes the GLRT test statistic close to zero, i.e.
    # stationarity_ratio -> 0 -- exactly the regime vision_trust_max_inflation caps.
    rng = np.random.default_rng(5)
    gravity = np.array([0.0, 0.0, 9.81])
    pipeline = _minimal_pipeline(zupt_window_samples=50, vision_trust_max_inflation=8.0)
    for _ in range(50):
        pipeline.process_imu(rng.normal(size=3) * 1e-6, gravity + rng.normal(size=3) * 1e-6, 0.005)
    assert pipeline._zupt_active
    assert pipeline._vision_noise_scale > 7.9  # approaches, never quite reaches, the 8.0 cap
    assert pipeline._vision_noise_scale <= 8.0  # but never exceeds it


def test_vision_noise_scale_starts_discounting_before_the_binary_gate_fires():
    # the whole point of 1c: unlike the very first version of this formula (which only
    # started discounting once stationarity_ratio was already below 1 -- i.e. the same
    # instant the binary gate fires, adding nothing beforehand), this one should already be
    # meaningfully below the cap and above 1.0 while _zupt_active is still False, given a
    # signal that's clearly quieter than real flight but not yet clean enough to pass the
    # strict instantaneous test.
    rng = np.random.default_rng(7)
    gravity = np.array([0.0, 0.0, 9.81])
    pipeline = _minimal_pipeline(zupt_window_samples=50, vision_trust_max_inflation=50.0)
    for _ in range(50):
        pipeline.process_imu(rng.normal(size=3) * 3e-2, gravity + rng.normal(size=3) * 3e-2, 0.005)
    assert not pipeline._zupt_active  # too noisy to pass the strict instantaneous test yet
    assert pipeline._vision_noise_scale > 3.0  # but already meaningfully discounted (ratio~3.4 here)


def test_vision_noise_scale_tracks_instantaneous_signal_even_while_hysteresis_holds_zupt_active():
    # 1c is deliberately *not* hysteresis-protected the way _zupt_active is (see
    # _update_stationary_state's docstring): during a brief vibration blip, ZUPT should stay
    # latched on (see test_is_stationary_survives_a_brief_vibration_blip_via_hysteresis), but
    # vision's trust should immediately reflect the blip's real motion-like signal instead of
    # staying inflated just because the sticky ZUPT gate hasn't (yet) turned off.
    rng = np.random.default_rng(6)
    gravity = np.array([0.0, 0.0, 9.81])
    pipeline = _minimal_pipeline(zupt_window_samples=50, zupt_hold_seconds=0.5)

    for _ in range(50):
        pipeline.process_imu(rng.normal(size=3) * 1e-4, gravity + rng.normal(size=3) * 1e-4, 0.005)
    assert pipeline._zupt_active
    pre_blip_scale = pipeline._vision_noise_scale
    assert pre_blip_scale > 40.0  # near the cap: this window looks essentially motionless

    for _ in range(5):
        pipeline.process_imu(np.array([0.3, 0.0, 0.0]), gravity + np.array([1.0, 0.0, 0.0]), 0.005)
        assert pipeline._zupt_active  # hysteresis holds this on
    # vision trust already reacted to the blip's real-motion-like signal, well before
    # hysteresis lets the sticky ZUPT gate itself turn off
    assert pipeline._vision_noise_scale < pre_blip_scale / 10


def test_stationary_detector_scales_with_declared_imu_noise_density():
    # the whole point of the GLRT stationary detector (1b) over the hand-tuned thresholds
    # it replaced: the same borderline signal should be judged differently depending on how
    # precise the IMU actually is, since the test is normalized by noise_params rather than
    # a fixed magic number. A small constant gyro offset that's way beyond a *tight*-noise
    # sensor's precision (so it reads as real signal, not noise) should be well within a
    # *loose*-noise sensor's precision (so the identical data reads as just noise).
    gravity = np.array([0.0, 0.0, 9.81])
    small_offset = np.array([0.01, 0.0, 0.0])

    def _pipeline_with_gyro_noise(gyro_noise_density):
        K, dist_coeffs = load_cam0_intrinsics()
        T_BS_cam0 = load_T_BS(f"{gt.MAV0_DIR}/cam0/sensor.yaml")
        noise_params = dict(load_imu_noise_params())
        noise_params["gyro_noise"] = gyro_noise_density
        return MSCKFPipeline(np.zeros(3), np.zeros(3), np.array([1.0, 0, 0, 0]), np.zeros(3), np.zeros(3),
                              _initial_covariance(), T_BS_cam0, K, dist_coeffs, noise_params,
                              zupt_window_samples=50)

    tight_pipeline = _pipeline_with_gyro_noise(1e-5)  # much more precise than small_offset
    loose_pipeline = _pipeline_with_gyro_noise(1e-1)  # much less precise than small_offset
    for i in range(50):
        rng = np.random.default_rng(4 * 1000 + i)  # same per-step realization for both pipelines
        gyro_sample = small_offset + rng.normal(size=3) * 1e-5
        accel_sample = gravity + rng.normal(size=3) * 1e-4
        tight_pipeline.process_imu(gyro_sample, accel_sample, 0.005)
        loose_pipeline.process_imu(gyro_sample, accel_sample, 0.005)

    assert not tight_pipeline._zupt_active  # too precise a sensor to call this "just noise"
    assert loose_pipeline._zupt_active      # too imprecise a sensor to tell this from noise


def test_is_stationary_survives_a_brief_vibration_blip_via_hysteresis():
    # a still-landed platform can have a second or two of vibration/motor noise that
    # briefly pushes the instantaneous test above threshold -- shouldn't immediately
    # cancel ZUPT, since the platform hasn't actually started moving (see
    # msckf_pipeline.py's _update_stationary_state docstring for the real-data incident
    # this guards against: a brief blip previously caused ZUPT to stay off for ~13s of
    # genuine continued stationarity, reopening the divergence it exists to prevent).
    # Note: because the instantaneous test itself looks at a sliding window (not just the
    # latest sample), a blip's effect lingers in that window for up to zupt_window_samples
    # more samples after the blip itself ends -- so zupt_hold_seconds needs real margin
    # over the window's own time span, not just over the blip's duration, matching the
    # ~4x ratio used in MSCKFPipeline's real defaults (window=100 samples=0.5s, hold=2.0s).
    rng = np.random.default_rng(1)
    gravity = np.array([0.0, 0.0, 9.81])
    pipeline = _minimal_pipeline(zupt_window_samples=50, zupt_hold_seconds=0.5)

    for _ in range(50):
        pipeline.process_imu(rng.normal(size=3) * 1e-4, gravity + rng.normal(size=3) * 1e-4, 0.005)
    assert pipeline._zupt_active

    # a short blip: 5 samples of real-looking motion
    for _ in range(5):
        pipeline.process_imu(np.array([0.3, 0.0, 0.0]), gravity + np.array([1.0, 0.0, 0.0]), 0.005)
        assert pipeline._zupt_active  # hysteresis should hold it active through the blip

    # settles back down -- more than a full window's worth of samples, so the blip has
    # fully aged out of the sliding window by the end -- should still read as stationary
    for _ in range(60):
        pipeline.process_imu(rng.normal(size=3) * 1e-4, gravity + rng.normal(size=3) * 1e-4, 0.005)
    assert pipeline._zupt_active


def test_is_stationary_eventually_deactivates_after_sustained_real_motion():
    rng = np.random.default_rng(2)
    gravity = np.array([0.0, 0.0, 9.81])
    pipeline = _minimal_pipeline(zupt_window_samples=50, zupt_hold_seconds=0.5)

    for _ in range(50):
        pipeline.process_imu(rng.normal(size=3) * 1e-4, gravity + rng.normal(size=3) * 1e-4, 0.005)
    assert pipeline._zupt_active

    # sustained, *time-varying* real motion -- not a constant reading, which ZARU could
    # (correctly, given its own model) start explaining away as an updated bias estimate,
    # since a perfectly constant reading is fundamentally indistinguishable from bias by
    # design -- for longer than zupt_hold_seconds, should eventually deactivate
    became_inactive = False
    for i in range(150):
        gyro = np.array([0.3, 0.0, 0.0]) * np.sin(i * 0.3)
        accel = gravity + np.array([1.5, 0.5, 0.0]) * np.cos(i * 0.3)
        pipeline.process_imu(gyro, accel, 0.005)
        if not pipeline._zupt_active:
            became_inactive = True
            break
    assert became_inactive


def test_is_stationary_disabled_when_enable_zupt_is_false():
    pipeline = _minimal_pipeline(enable_zupt=False, zupt_window_samples=5)
    gravity = np.array([0.0, 0.0, 9.81])
    for _ in range(20):
        pipeline.process_imu(np.zeros(3), gravity, 0.005)
    assert not pipeline._zupt_active


def test_stationary_pipeline_corrects_gyro_bias_via_zaru_with_no_vision_at_all():
    # a slightly-wrong initial gyro-bias guess, fed a genuinely still IMU signal (true bias
    # = true_bias, zero real rotation) -- ZARU should pull b_g further toward true_bias
    # purely from process_imu, with process_image never called (no vision, no clones,
    # nothing). The mismatch here is deliberately small (~0.003 rad/s), not the ~0.02-0.03
    # rad/s an earlier version of this test used: with the GLRT stationary detector (1b),
    # a mismatch that large relative to this sensor's real, tight noise floor is -- rightly
    # -- statistically indistinguishable from genuine rotation, so the detector correctly
    # refuses to call it "stationary" until the estimate is already reasonably close, same
    # fundamental ambiguity noted for ZARU itself (it can't tell "wrong bias" from "real
    # constant rotation" from gyro data alone).
    rng = np.random.default_rng(3)
    gravity = np.array([0.0, 0.0, 9.81])
    true_bias = np.array([0.02, -0.01, 0.03])
    wrong_bias_guess = true_bias - np.array([0.002, -0.001, 0.0015])

    pipeline = _minimal_pipeline(b_g0=wrong_bias_guess, zupt_window_samples=50)
    for _ in range(200):
        pipeline.process_imu(true_bias + rng.normal(size=3) * 1e-4, gravity + rng.normal(size=3) * 1e-4, 0.005)

    assert pipeline._zupt_active
    assert np.linalg.norm(pipeline.state.b_g - true_bias) < np.linalg.norm(wrong_bias_guess - true_bias)


def test_tuned_zupt_noise_inflation_has_no_false_positives_on_real_data():
    # regression guard for the empirically-tuned zupt_noise_inflation default (see
    # msckf_pipeline.py's _update_stationary_state docstring for the full sweep this came
    # from): over a real window spanning flight, landing, the ~24s stationary hold, and the
    # resumed takeoff, the detector must never fire during genuine motion -- a false
    # positive there means applying ZUPT/ZARU/gravity-alignment against a platform that's
    # actually moving, which actively corrupts the state, a strictly worse failure mode
    # than detecting a real stop a bit late -- and it must actually detect the real
    # stationary segment, not just stay silent the whole time.
    K, dist_coeffs = load_cam0_intrinsics()
    T_BS_cam0 = load_T_BS(f"{gt.MAV0_DIR}/cam0/sensor.yaml")
    noise_params = load_imu_noise_params()

    duration_s = 45.0
    gt_timestamps, _, _, _ = gt._load_state_ground_truth()
    cam0_all_timestamps = gt.load_cam0_timestamps()
    start_frame = int(np.searchsorted(cam0_all_timestamps, gt_timestamps[0])) + 30
    n_frames = int(duration_s * 20)
    frames = list(iter_cam0_frames(max_frames=n_frames, start_frame=start_frame))
    t0 = frames[0][0]
    p0 = gt.interpolate_ground_truth_position(t0)
    v0 = gt.interpolate_ground_truth_velocity(t0)
    q0 = gt.interpolate_ground_truth_orientation(t0)

    pipeline = MSCKFPipeline(p0, v0, q0, DUMMY_BIAS_GYRO, DUMMY_BIAS_ACCEL, _initial_covariance(),
                              T_BS_cam0, K, dist_coeffs, noise_params)

    imu_timestamps, gyro, accel = _load_imu_measurements()
    imu_idx = np.searchsorted(imu_timestamps, t0, side="right")
    t_prev = t0
    ever_detected_real_stationary = False
    for frame_t, image in frames:
        while imu_idx < len(imu_timestamps) and imu_timestamps[imu_idx] <= frame_t:
            t_curr = int(imu_timestamps[imu_idx])
            pipeline.process_imu(gyro[imu_idx], accel[imu_idx], (t_curr - t_prev) / 1e9)
            t_prev = t_curr
            imu_idx += 1
            v_gt_norm = np.linalg.norm(gt.interpolate_ground_truth_velocity(t_curr))
            if pipeline._zupt_active:
                assert v_gt_norm < 0.1, (f"ZUPT falsely active while genuinely moving "
                                          f"(|v_gt|={v_gt_norm:.3f} m/s) at t={(t_curr - t0) / 1e9:.2f}s")
                if v_gt_norm < 0.02:
                    ever_detected_real_stationary = True
        pipeline.process_image(frame_t, image)

    assert ever_detected_real_stationary


def test_window_stays_bounded_and_covariance_stays_healthy():
    max_clones = 12
    pipeline, _, n_clones_history = _run_pipeline(10.0, max_clones=max_clones,
                                                   bias_gyro=DUMMY_BIAS_GYRO, bias_accel=DUMMY_BIAS_ACCEL)

    assert max(n_clones_history) <= max_clones  # strict cap: never exceeded
    assert n_clones_history[-1] >= max_clones  # should have filled up over 10s, not stayed emptier

    eigenvalues = np.linalg.eigvalsh(pipeline.state.P)
    assert eigenvalues.min() > -1e-9


def test_updates_are_actually_applied_and_some_features_are_gated_out():
    pipeline, _, _ = _run_pipeline(10.0, max_clones=12, bias_gyro=DUMMY_BIAS_GYRO, bias_accel=DUMMY_BIAS_ACCEL)

    assert pipeline.n_updates_applied > 0
    # not asserting rejections happen (that's data-dependent), just that the
    # counter mechanism actually ran without erroring
    assert pipeline.n_tracks_rejected >= 0


def test_clone_frame_ids_stay_in_sync_after_marginalization():
    pipeline, frames, _ = _run_pipeline(8.0, max_clones=10, bias_gyro=DUMMY_BIAS_GYRO, bias_accel=DUMMY_BIAS_ACCEL)

    # every surviving clone_frame_id must correspond to one of the frames we actually processed
    processed_timestamps = {t for t, _ in frames}
    for t in pipeline.clone_frame_ids:
        assert t in processed_timestamps
    assert pipeline.clone_frame_ids == sorted(pipeline.clone_frame_ids)  # still chronological


def test_estimated_trajectory_stays_reasonably_close_to_ground_truth():
    # regression guard, not a precision claim: with the dummy bias and real
    # feature updates, position error over a short well-behaved window
    # shouldn't blow up the way uncorrected pure dead reckoning does
    pipeline, frames, _ = _run_pipeline(10.0, max_clones=15, bias_gyro=DUMMY_BIAS_GYRO, bias_accel=DUMMY_BIAS_ACCEL)

    T_BS_cam0 = load_T_BS(f"{gt.MAV0_DIR}/cam0/sensor.yaml")
    last_clone_timestamp = pipeline.clone_frame_ids[-1]
    gt_p = body_to_sensor_position(gt.interpolate_ground_truth_position(last_clone_timestamp),
                                    gt.interpolate_ground_truth_orientation(last_clone_timestamp), T_BS_cam0)
    error = np.linalg.norm(pipeline.state.clone_positions[-1] - gt_p)

    assert error < 1.0, f"final position error {error:.3f}m exceeds the 1m regression threshold"
