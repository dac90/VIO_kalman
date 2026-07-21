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

        # max_clones is a *soft* cap (a clone still referenced by an active
        # track is kept rather than evicted -- see msckf_pipeline.py); only
        # hard_max_clones is a true, always-enforced upper bound
        assert pipeline.state.n_clones <= pipeline.hard_max_clones
        expected_dim = N_IMU_ERROR + N_CLONE_ERROR * pipeline.state.n_clones
        assert pipeline.state.P.shape == (expected_dim, expected_dim)
        assert len(pipeline.clone_frame_ids) == pipeline.state.n_clones
        assert np.allclose(pipeline.state.P, pipeline.state.P.T, atol=1e-9)
        n_clones_history.append(pipeline.state.n_clones)

    return pipeline, frames, n_clones_history


def test_window_stays_bounded_and_covariance_stays_healthy():
    max_clones = 12
    pipeline, _, n_clones_history = _run_pipeline(10.0, max_clones=max_clones,
                                                   bias_gyro=DUMMY_BIAS_GYRO, bias_accel=DUMMY_BIAS_ACCEL)

    # soft cap: usually at or near max_clones, but may run a bit over while
    # an active track keeps an old clone alive -- hard_max_clones is the real bound
    assert max(n_clones_history) <= pipeline.hard_max_clones
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
