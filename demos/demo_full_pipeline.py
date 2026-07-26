"""Stage 8 capstone: run the full MSCKFPipeline over an extended real window
and evaluate the estimated trajectory against ground truth with the standard
VIO metrics -- ATE (absolute trajectory error) and RPE (relative pose error
at a fixed time delta).

No SE3/Umeyama alignment step is needed before computing ATE here: the
filter is initialized directly from ground truth's own starting pose, so
estimated and ground-truth trajectories already live in the same frame
(unlike a real deployed VIO system, whose own coordinate frame would need
aligning to ground truth first).

DURATION_S is deliberately conservative (validated: ATE=0.108m, RPE@1s=0.064m
at 20s). Longer runs eventually diverge, now via a much better-understood
mechanism than originally suspected: it isn't primarily low-parallax
triangulation feeding bad measurements past an overly-permissive chi-square
gate (that was the first hypothesis; adding triangulation.py's parallax/
cheirality/reprojection-error checks alone barely moved the divergence point).
The actual dominant cause, found by comparing against pure IMU dead-reckoning
over the same window (which turned out to diverge at almost exactly the same
rate as the "full" pipeline!) and by instrumenting P's own reported
uncertainty against the real error: msckf_pipeline.py used to let its sliding
window grow up to 3x max_clones (a "soft cap", added to stop force-evicting
still-active long-lived tracks with barely any new data). That let features
get triangulated against increasingly stale camera-pose estimates that had
already drifted -- Gauss-Newton fits the feature to those poses by
construction, so the residual looks small and "consistent" even when the
poses themselves are wrong, making the EKF update read as highly informative
and shrink P accordingly. That reinforces the existing drift instead of
correcting it: P collapses to near-zero (confirmed directly: sigma_v stayed
~0.01-0.1 m/s throughout a run where the real velocity error grew past
11 m/s) while bias estimates get pulled further from truth, not closer,
compounding fast. Reverting to a strictly bounded window (msckf_pipeline.py's
max_clones is now always enforced, no soft cap) starves that feedback loop of
the staleness it needs, since the new parallax/cheirality checks handle the
force-evicted thin tracks that motivated the soft cap in the first place.
Net effect: instantaneous position error at t=20s dropped from ~14.6m to
~0.6m (~25x) versus the pre-fix pipeline, and the point where the residual
feedback loop resumes moved from somewhere before 20s out to roughly 25s.
It isn't eliminated outright, though: this is the classic EKF-SLAM/MSCKF
consistency problem (unobservable directions -- global yaw/position --
becoming falsely overconfident from self-referential landmark triangulation),
which real fixes address via Observability-Constrained EKF or First-Estimates
Jacobians (fixing H's linearization point rather than re-linearizing at the
latest estimate every update) -- a substantially bigger undertaking than
either fix applied so far.

A cheaper partial mitigation was added first: msckf_update.py's ekf_update
takes an optional min_variance, a covariance floor for the 15-dim IMU error
state (see msckf_pipeline.py's DEFAULT_MIN_SIGMA), applied by default. This is
regularization in the classic Tikhonov/ridge sense -- add a small nonnegative
diagonal term so P's eigenvalues can never collapse to zero, however many
confidently-accepted updates came before. It's genuinely helpful at the
*right* scale (demos/demo_divergence_analysis.py's sweep found ~0.4x a naive
first guess was the sweet spot; going further made things worse, since
keeping P artificially open raises the Kalman gain for every subsequent
measurement, including the corrupted ones) -- but it only buys margin; it
doesn't fix the underlying consistency problem.

Then the user, watching demos/full_pipeline.mp4, noticed the divergence in
this test window actually starts right when the drone lands and holds still
-- confirmed directly against ground-truth velocity (drops from >0.2 m/s to
<0.02 m/s at t~18s, stays there ~23s, resumes at t~42s). That's the dominant
cause, and a much more standard, well-understood one than anything above: a
stationary camera has zero parallax on everything it sees, so vision updates
aren't just unhelpful, they're actively harmful (same self-referential
triangulation issue, but with no real corrective signal available at all to
push back against it), while unaided dead-reckoning drifts on whatever
residual accel bias remains, unconstrained. The standard fix is a zero-
velocity update (ZUPT, msckf_update.py's zero_velocity_update): detect
stationarity directly from raw IMU statistics and assert v=0 as a direct
pseudo-measurement, independent of vision entirely. Two non-obvious bugs
turned up building it (see msckf_pipeline.py's _is_stationary and
zero_velocity_update docstrings): the detector must bias-correct with the
filter's *own* gyro bias estimate, which can itself drift wrong (needs
hysteresis to survive brief vibration blips without falsely deactivating);
and the ZUPT's mean correction must be restricted to the velocity block only,
since a full EKF update let it leak ~0.06 rad/s of drift into gyro bias
through an uncalibrated cross-correlation -- which then broke the detector's
own bias correction, a nasty bootstrapping failure.

With both fixed: ATE=0.240m, RPE@1s=0.042m over the current DURATION_S=30
(versus ATE=2.98m/RPE=0.56m with the covariance floor alone) -- roughly a 12x
improvement, and the ~23s stationary segment now tracks sub-meter the whole
way through instead of drifting tens of meters. The pipeline still eventually
diverges on renewed real motion past t~42s for the original self-referential-
triangulation reason (Observability-Constrained EKF / First-Estimates
Jacobians remains the proper fix for that), but the specific failure mode
the user spotted from the video is now substantially resolved.

Two follow-up refinements, both validated with demos/demo_divergence_analysis.py
over a 60s window (long enough to see the whole stationary hold and the
motion-resumption divergence, unlike this demo's 30s):

1. Detection moved from once per camera frame (20Hz) to once per IMU sample
   (200Hz, msckf_pipeline.py's _update_stationary_state) -- reacts to a stop
   up to 10x faster, since none of this depends on camera frames at all.
2. A bare ZUPT has no way to correct the residual accel bias actually causing
   the drift it's trying to stop (it only asserts v=0), so position still
   crept slowly even while it fired correctly the whole time. Added the
   standard INS companions -- ZARU (zero angular-rate update: raw gyro reading
   IS the bias when truly not rotating) and gravity/tilt alignment (raw accel
   reading is fixed by orientation+bias alone when truly not accelerating,
   correcting roll/pitch and accel bias together) -- both direct measurements
   of bias/tilt, not inferred through a correlation, so neither needs
   zero_velocity_update's masking trick. Result: position error over the ~24s
   hold went from slowly creeping (0.31m -> 1.17m, t=20s->40s) to essentially
   flat (0.72m -> 0.73m).

One real gotcha along the way, worth knowing if extending this: applying
those corrections at the new 200Hz detection rate (not just detecting at that
rate) was actively harmful -- consecutive IMU samples 5ms apart aren't
independent measurements, so treating ~200 corrections/second as that many
independent ones manufactured far more confidence than the data supports,
badly corrupting accel bias within the hold itself. Correction application is
now rate-limited back to ~20Hz (zupt_update_interval) while detection stays
fast; only the *reaction time* needed the speedup, not the correction rate.

Net effect on ATE at this demo's DURATION_S=30 specifically: 0.466m, slightly
*worse* than the masked-ZUPT-only version's 0.240m -- because that version
happened to track tighter in exactly the first ~12s of the hold (t=18-30s)
that this window covers, while ZARU/tilt trades some of that early tightness
for staying flat over the *entire* ~24s hold rather than resuming its drift
partway through. This demo's window is too short to see that payoff; the 60s
diagnostic (demo_divergence_analysis.py) shows it clearly.

A third refinement replaced the hand-tuned detector thresholds with a
GLRT/SHOE test (msckf_pipeline.py's _update_stationary_state, standard in
ZUPT-aided INS) normalized by the IMU's actual characterized noise density --
principled, but the datasheet noise number turned out to understate real
sensor behavior by 1x-17x on this dataset (a documented, generic gap in
GLRT/SHOE deployments, not specific to this sensor), needing one explicit,
empirically-tuned compensating knob (zupt_noise_inflation). Swept 1-50 against
ground truth: below ~5 the detector never fires at all; from 5-45 it never
misfires during genuine motion anywhere in a 60s window, with detection
latency at the real stop improving only marginally past 20 (20.4s at
inflation=5 down to 18.7s at 45); at 50 it starts misfiring during the
takeoff ramp (a real false positive -- actively harmful, since it would apply
ZUPT/ZARU/tilt-alignment against a platform that's actually accelerating).
Landed on 20.0: essentially all the reachable latency improvement, with
better than 2x margin below the observed failure point.

This tuning pass alone had a large, somewhat surprising downstream effect:
detecting ~0.5s earlier and holding ~2.5s longer through the real stationary
segment (matching it more precisely) left the state noticeably better
conditioned at the moment vision-only tracking resumes -- position error at
t=60s in the 60s diagnostic dropped from ~1512m (inflation=10, an earlier,
untuned default) to ~10.6m. The already-known post-motion divergence
mechanism (self-referential triangulation) is untouched and will still
eventually resurface on a long enough run, but how much runway it gets before
mattering turned out to be quite sensitive to how precisely the stationary
window itself is detected.
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
from quaternion_utils import quat_error_vector
from triangulation import load_cam0_intrinsics

DURATION_S = 30.0
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

    for frame_idx, (frame_t, image) in enumerate(frames):
        while imu_idx < len(imu_timestamps) and imu_timestamps[imu_idx] <= frame_t:
            t_curr = int(imu_timestamps[imu_idx])
            pipeline.process_imu(gyro[imu_idx], accel[imu_idx], (t_curr - t_prev) / 1e9)
            t_prev = t_curr
            imu_idx += 1
        pipeline.process_image(frame_t, image)

        if frame_idx % 50 == 0:
            gt_p = body_to_sensor_position(gt.interpolate_ground_truth_position(frame_t),
                                            gt.interpolate_ground_truth_orientation(frame_t), T_BS_cam0)
            gt_q = gt.interpolate_cam0_orientation(frame_t)
            pos_err = np.linalg.norm(pipeline.state.clone_positions[-1] - gt_p)
            theta_err = np.linalg.norm(quat_error_vector(gt_q, pipeline.state.clone_orientations[-1]))
            print(f"frame {frame_idx}/{n_frames}  pos_err={pos_err:.4f}m  theta_err={theta_err:.4f}rad")

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
