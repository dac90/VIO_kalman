"""Stage 8 capstone: run the full MSCKFPipeline over an extended real window
and evaluate the estimated trajectory against ground truth with the standard
VIO metrics -- ATE (absolute trajectory error) and RPE (relative pose error
at a fixed time delta).

No SE3/Umeyama alignment step is needed before computing ATE here: the
filter is initialized directly from ground truth's own starting pose, so
estimated and ground-truth trajectories already live in the same frame
(unlike a real deployed VIO system, whose own coordinate frame would need
aligning to ground truth first).

DURATION_S is deliberately conservative (validated to stay under ~0.1m ATE
at 20s during tuning). Longer runs (found while tuning this pipeline: 60s)
eventually diverge via the same slow feedback loop dead reckoning always
has -- a run of small individual errors nudges triangulation slightly,
which increases rejection rates, which starves the filter of corrections,
which lets error grow further. Two real issues in that loop were found and
fixed here (see msckf_pipeline.py: the eviction policy that used to nibble
long-lived tracks to death one frame at a time, and an overly-tight default
observation_noise_std), which pushed the honest "stable" duration from ~5s
to ~20s -- but didn't eliminate the failure mode outright. The parallax
check discussed and deferred earlier (see triangulation.py) is the most
likely next lever for extending this further.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for the top-level modules below

import matplotlib.pyplot as plt
import numpy as np

import ground_truth as gt
from feature_tracker import iter_cam0_frames
from frames import body_to_sensor_position, load_T_BS
from imu_propagation import _load_imu_measurements
from msckf_pipeline import MSCKFPipeline
from msckf_state import load_imu_noise_params
from triangulation import load_cam0_intrinsics

DURATION_S = 20.0
MAX_CLONES = 20

# same independent "dummy bias" used throughout the demos -- see
# demos/demo_mean_gravity_corrected_accel.py for how it was derived, and
# demos/demo_msckf_update.py for why zero bias makes the chi-square gate
# reject nearly everything (the filter's own P badly underestimates zero-bias
# dead-reckoning's real error)
DUMMY_BIAS_GYRO = np.array([-0.00233173, 0.02172386, 0.07821335])
DUMMY_BIAS_ACCEL = np.array([-0.04066623, 0.1155297, 0.05121861])

INITIAL_SIGMA = dict(theta=1e-3, b_g=1e-2, v=1e-1, b_a=1e-1, p=1e-3)

RPE_DELTA_S = 1.0


def initial_covariance():
    diag = ([INITIAL_SIGMA["theta"] ** 2] * 3 + [INITIAL_SIGMA["b_g"] ** 2] * 3 + [INITIAL_SIGMA["v"] ** 2] * 3
            + [INITIAL_SIGMA["b_a"] ** 2] * 3 + [INITIAL_SIGMA["p"] ** 2] * 3)
    return np.diag(diag)


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

    for frame_t, image in frames:
        while imu_idx < len(imu_timestamps) and imu_timestamps[imu_idx] <= frame_t:
            t_curr = int(imu_timestamps[imu_idx])
            pipeline.process_imu(gyro[imu_idx], accel[imu_idx], (t_curr - t_prev) / 1e9)
            t_prev = t_curr
            imu_idx += 1
        pipeline.process_image(frame_t, image)

    timestamps, positions, _ = pipeline.full_trajectory()
    positions = np.array(positions)
    gt_positions = np.array([body_to_sensor_position(gt.interpolate_ground_truth_position(t),
                                                       gt.interpolate_ground_truth_orientation(t), T_BS_cam0)
                              for t in timestamps])

    position_errors = np.linalg.norm(positions - gt_positions, axis=1)
    ate = np.sqrt(np.mean(position_errors ** 2))

    rpe_errors = []
    delta_ns = int(RPE_DELTA_S * 1e9)
    j = 0
    for i, t in enumerate(timestamps):
        while j < len(timestamps) and timestamps[j] - t < delta_ns:
            j += 1
        if j >= len(timestamps):
            break
        est_delta = positions[j] - positions[i]
        gt_delta = gt_positions[j] - gt_positions[i]
        rpe_errors.append(np.linalg.norm(est_delta - gt_delta))
    rpe = np.sqrt(np.mean(np.array(rpe_errors) ** 2))

    print(f"Ran {n_frames} frames ({DURATION_S:.0f}s). "
          f"{pipeline.n_updates_applied} feature updates applied, {pipeline.n_tracks_rejected} rejected.")
    print(f"ATE (RMSE position error, no alignment needed): {ate:.4f} m")
    print(f"RPE @ {RPE_DELTA_S:.1f}s (RMSE relative displacement error): {rpe:.4f} m")
    print(f"Final gyro bias:  {pipeline.state.b_g}  (reference: {DUMMY_BIAS_GYRO})")
    print(f"Final accel bias: {pipeline.state.b_a}  (reference: {DUMMY_BIAS_ACCEL})")

    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(121, projection="3d")
    ax.plot(*gt_positions.T, color="tab:blue", label="Ground truth")
    ax.plot(*positions.T, color="tab:green", label="MSCKF estimate")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title(f"Full pipeline: {DURATION_S:.0f}s estimate vs ground truth")
    ax.legend()

    t_axis = (np.array(timestamps) - timestamps[0]) / 1e9
    ax2 = fig.add_subplot(122)
    ax2.plot(t_axis, position_errors)
    ax2.axhline(ate, color="black", linestyle="--", label=f"ATE = {ate:.3f}m")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Position error (m)")
    ax2.set_title("Absolute position error over time")
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
