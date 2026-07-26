"""Diagnostic: does the residual long-run divergence (see demo_full_pipeline.py's
docstring) come from the filter trusting one of its measurement sources -- vision
updates, specifically -- more than it should?

Runs the full pipeline over a window long enough to pass the point where the
(now much-delayed, but not eliminated) divergence resumes, and plots four
things against each other on a shared time axis:

  1. Position error vs. the filter's own *predicted* position uncertainty
     (sqrt(trace(P_position)), the RMS expected error magnitude under the
     filter's Gaussian assumption, and its 3-sigma band). This is the standard
     EKF consistency check: if the real error stays inside the predicted
     band, the filter's confidence is honest. If the real error escapes the
     band and the gap keeps widening, the filter has become overconfident --
     it's trusting *something* (here: vision updates, triangulated against the
     filter's own possibly-already-wrong poses) more than the incoming
     information actually warrants.
  2. The same consistency check for orientation error (quat_error_vector
     against ground truth) vs. the predicted 3-sigma orientation bound --
     position and orientation error don't necessarily diverge in lockstep
     (yaw in particular is fundamentally unobservable in monocular VIO), so
     it's worth seeing both, not just inferring one from the other.
  3. The six IMU bias components (gyro x/y/z, accel x/y/z) against the
     independent reference bias -- if vision updates were correcting the
     filter honestly, bias estimates should stay near the reference or
     improve; if they drift away as error grows, that's corrupted correction,
     not helpful correction.
  4. The "magnitude" (sqrt(trace) = RMS expected error norm) of each of the
     five IMU error-state covariance blocks (theta, b_g, v, b_a, p), to see
     which blocks are collapsing (shrinking despite growing real error) vs.
     genuinely growing.

See msckf_pipeline.py's and demo_full_pipeline.py's docstrings for the full
history of this divergence investigation (the earlier finding that motivates
this plot: pure IMU dead-reckoning diverged at almost the exact same rate as
the "full" pipeline, and P's own reported velocity uncertainty stayed flat at
~0.01-0.1 m/s throughout a run where the real velocity error grew past 11 m/s
-- i.e. exactly this kind of overconfidence, just far more severe before the
sliding-window fix).

MSCKFPipeline now also applies a zero-velocity update (ZUPT) whenever it
detects the platform is stationary from raw IMU statistics -- added after the
user noticed, from the video demo, that the divergence in this test window
begins right when the drone lands and holds still (t~18-41s here). Watch the
bottom-left accel-bias panel and the top panel together: with ZUPT, bias
stays essentially flat and error stays sub-meter through the whole stationary
segment, instead of both drifting steadily once the covariance floor alone
was the only defense. To see the pre-ZUPT behavior this plot originally
diagnosed, patch MSCKFPipeline(..., enable_zupt=False) into main() below.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for the top-level modules below

import matplotlib.pyplot as plt
import numpy as np

import ground_truth as gt
from feature_tracker import iter_cam0_frames
from frames import load_T_BS
from imu_propagation import _load_imu_measurements
from msckf_pipeline import MSCKFPipeline
from msckf_state import load_imu_noise_params
from quaternion_utils import quat_error_vector
from triangulation import load_cam0_intrinsics

DURATION_S = 60.0  # long enough to pass the point (~t=25-30s) where the residual divergence resumes
MAX_CLONES = 20
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "divergence_analysis.png")

# the filter is initialized exactly at ground truth, so pos_err/theta_err start at ~0 (floating-
# point noise, not a real value) -- on a log-scale plot that briefly plunges to ~1e-16 and drags the
# whole y-axis down with it, squashing every other value against the top of the plot. The lines
# themselves are still drawn in full (that dip is real data, however uninteresting); only the
# y-axis autoscale ignores these first few samples, in the two error-vs-bound panels only. The
# onset/consistency analysis below always uses the full, untrimmed arrays regardless.
PLOT_SKIP_S = 0.5

# same independent "dummy bias" used throughout the demos -- see
# demos/demo_mean_gravity_corrected_accel.py for how it was derived
DUMMY_BIAS_GYRO = np.array([-0.00233173, 0.02172386, 0.07821335])
DUMMY_BIAS_ACCEL = np.array([-0.04066623, 0.1155297, 0.05121861])

INITIAL_SIGMA = dict(theta=1e-3, b_g=1e-2, v=1e-1, b_a=1e-1, p=1e-3)


def initial_covariance():
    diag = ([INITIAL_SIGMA["theta"] ** 2] * 3 + [INITIAL_SIGMA["b_g"] ** 2] * 3 + [INITIAL_SIGMA["v"] ** 2] * 3
            + [INITIAL_SIGMA["b_a"] ** 2] * 3 + [INITIAL_SIGMA["p"] ** 2] * 3)
    return np.diag(diag)


def _block_magnitude(diag_P, lo, hi):
    """sqrt(trace) of a diagonal covariance block -- the RMS expected norm of that
    block's error vector under the filter's own Gaussian assumption, directly
    comparable to an actual error norm computed the same way."""
    return np.sqrt(np.sum(diag_P[lo:hi]))


def _shade_stationary_intervals(ax, t, active):
    """Light-grey background bands marking where _update_stationary_state's GLRT judged
    the platform stationary -- i.e. where ZUPT/ZARU/gravity-alignment were actively
    correcting the state instead of vision. Makes it visible at a glance whether a given
    stretch of error growth (or flatness) happens during a stationary hold or real motion."""
    edges = np.diff(active.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if active[0]:
        starts = [0] + starts
    if active[-1]:
        ends = ends + [len(active) - 1]
    for i, (s, e) in enumerate(zip(starts, ends)):
        ax.axvspan(t[s], t[e], color="lightgrey", alpha=0.6, zorder=0,
                   label="Stationarity (ZUPT) active" if i == 0 else None)


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

    history = {"t": [], "pos_err": [], "sigma_p": [], "theta_err": [], "b_g": [], "b_a": [],
               "sigma_theta": [], "sigma_bg": [], "sigma_v": [], "sigma_ba": [],
               "n_updates": [], "n_rejected": [], "zupt_active": []}

    for frame_idx, (frame_t, image) in enumerate(frames):
        while imu_idx < len(imu_timestamps) and imu_timestamps[imu_idx] <= frame_t:
            t_curr = int(imu_timestamps[imu_idx])
            pipeline.process_imu(gyro[imu_idx], accel[imu_idx], (t_curr - t_prev) / 1e9)
            t_prev = t_curr
            imu_idx += 1
        pipeline.process_image(frame_t, image)

        s = pipeline.state
        gt_p = gt.interpolate_ground_truth_position(frame_t)
        gt_q = gt.interpolate_ground_truth_orientation(frame_t)
        diag_P = np.diag(s.P)[:15]

        history["t"].append(frame_idx / 20.0)
        history["pos_err"].append(np.linalg.norm(s.p - gt_p))
        history["sigma_p"].append(_block_magnitude(diag_P, 12, 15))
        history["theta_err"].append(np.linalg.norm(quat_error_vector(gt_q, s.q)))
        history["b_g"].append(s.b_g.copy())
        history["b_a"].append(s.b_a.copy())
        history["sigma_theta"].append(_block_magnitude(diag_P, 0, 3))
        history["sigma_bg"].append(_block_magnitude(diag_P, 3, 6))
        history["sigma_v"].append(_block_magnitude(diag_P, 6, 9))
        history["sigma_ba"].append(_block_magnitude(diag_P, 9, 12))
        history["n_updates"].append(pipeline.n_updates_applied)
        history["n_rejected"].append(pipeline.n_tracks_rejected)
        history["zupt_active"].append(pipeline._zupt_active)

        if frame_idx % 50 == 0:
            print(f"frame {frame_idx}/{n_frames}  pos_err={history['pos_err'][-1]:.4f}m  "
                  f"theta_err={history['theta_err'][-1]:.4f}rad")

    t = np.array(history["t"])
    pos_err = np.array(history["pos_err"])
    sigma_p = np.array(history["sigma_p"])
    theta_err = np.array(history["theta_err"])
    sigma_theta = np.array(history["sigma_theta"])
    b_g = np.array(history["b_g"])
    b_a = np.array(history["b_a"])
    zupt_active = np.array(history["zupt_active"])

    # when (if at all) does real error escape the filter's own 3-sigma confidence band?
    # early on, position sigma starts from a near-zero initial P0 and briefly lags behind
    # small transient errors before catching up -- that's an initial-condition artifact, not
    # divergence, so report the *sustained* escape point instead: the last moment the filter
    # was still consistent, after which it never recovers.
    ratio = pos_err / (3 * sigma_p)
    consistent = np.where(ratio <= 1.0)[0]
    sustained_onset = (consistent[-1] + 1) if len(consistent) > 0 and consistent[-1] + 1 < len(t) else None

    if sustained_onset is not None:
        t_onset = t[sustained_onset]
        final_ratio = ratio[-1]
        print(f"Position error permanently escapes the filter's own 3-sigma bound at t={t_onset:.1f}s "
              f"(never recovers after that point in this run).")
        print(f"By t={t[-1]:.1f}s: error={pos_err[-1]:.2f}m vs predicted 3-sigma={3*sigma_p[-1]:.2f}m "
              f"({final_ratio:.1f}x over) -- the filter is overconfident, not just wrong.")
    else:
        print("Position error never permanently escapes the filter's own 3-sigma bound over this run: consistent.")

    theta_ratio = theta_err / (3 * sigma_theta)
    theta_consistent = np.where(theta_ratio <= 1.0)[0]
    theta_sustained_onset = (theta_consistent[-1] + 1
                              if len(theta_consistent) > 0 and theta_consistent[-1] + 1 < len(t) else None)
    if theta_sustained_onset is not None:
        print(f"Orientation error permanently escapes its own 3-sigma bound at t={t[theta_sustained_onset]:.1f}s "
              f"(final: {theta_err[-1]:.4f} rad vs predicted 3-sigma={3*sigma_theta[-1]:.4f} rad, "
              f"{theta_ratio[-1]:.1f}x over).")
    else:
        print("Orientation error never permanently escapes its own 3-sigma bound over this run: consistent.")

    print(f"Total feature updates applied: {pipeline.n_updates_applied}, rejected: {pipeline.n_tracks_rejected}")
    print(f"Final gyro bias:  {pipeline.state.b_g}  (reference: {DUMMY_BIAS_GYRO}, "
          f"|error|={np.linalg.norm(pipeline.state.b_g - DUMMY_BIAS_GYRO):.4f})")
    print(f"Final accel bias: {pipeline.state.b_a}  (reference: {DUMMY_BIAS_ACCEL}, "
          f"|error|={np.linalg.norm(pipeline.state.b_a - DUMMY_BIAS_ACCEL):.4f})")

    fig, axes = plt.subplot_mosaic([["err_theta", "err_pos"], ["bg", "ba"], ["cov", "cov"]],
                                    figsize=(13, 11))

    plot_mask = t >= PLOT_SKIP_S

    ax_err = axes["err_pos"]
    _shade_stationary_intervals(ax_err, t, zupt_active)
    ax_err.semilogy(t, pos_err, color="tab:red", linewidth=1.8, label="Actual position error")
    ax_err.semilogy(t, 3 * sigma_p, color="tab:red", linestyle="--", linewidth=1.2,
                     label="Filter's own 3-sigma position bound")
    if sustained_onset is not None:
        ax_err.axvline(t[sustained_onset], color="black", linestyle=":", linewidth=1,
                        label=f"escapes 3-sigma bound (t={t[sustained_onset]:.1f}s)")
    y_lo = min(pos_err[plot_mask].min(), (3 * sigma_p)[plot_mask].min())
    y_hi = max(pos_err[plot_mask].max(), (3 * sigma_p)[plot_mask].max())
    ax_err.set_ylim(y_lo * 0.5, y_hi * 1.5)
    ax_err.set_ylabel("Position error / bound (m, log scale)")
    ax_err.set_title("Position: divergence vs. predicted uncertainty")
    ax_err.legend(fontsize=8)
    ax_err.grid(True, which="both", alpha=0.3)

    ax_err_theta = axes["err_theta"]
    _shade_stationary_intervals(ax_err_theta, t, zupt_active)
    ax_err_theta.semilogy(t, theta_err, color="tab:purple", linewidth=1.8, label="Actual orientation error")
    ax_err_theta.semilogy(t, 3 * sigma_theta, color="tab:purple", linestyle="--", linewidth=1.2,
                           label="Filter's own 3-sigma orientation bound")
    if theta_sustained_onset is not None:
        ax_err_theta.axvline(t[theta_sustained_onset], color="black", linestyle=":", linewidth=1,
                              label=f"escapes 3-sigma bound (t={t[theta_sustained_onset]:.1f}s)")
    y_lo = min(theta_err[plot_mask].min(), (3 * sigma_theta)[plot_mask].min())
    y_hi = max(theta_err[plot_mask].max(), (3 * sigma_theta)[plot_mask].max())
    ax_err_theta.set_ylim(y_lo * 0.5, y_hi * 1.5)
    ax_err_theta.set_ylabel("Orientation error / bound (rad, log scale)")
    ax_err_theta.set_title("Orientation: divergence vs. predicted uncertainty")
    ax_err_theta.legend(fontsize=8)
    ax_err_theta.grid(True, which="both", alpha=0.3)

    labels = ["x", "y", "z"]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    ax_bg = axes["bg"]
    for i in range(3):
        ax_bg.plot(t, b_g[:, i], color=colors[i], label=f"b_g {labels[i]}")
        ax_bg.axhline(DUMMY_BIAS_GYRO[i], color=colors[i], linestyle="--", alpha=0.4)
    ax_bg.set_title("Gyro bias (dashed = independent reference)")
    ax_bg.set_ylabel("rad/s")
    ax_bg.set_xlabel("Time (s)")
    ax_bg.legend(fontsize=8)

    ax_ba = axes["ba"]
    for i in range(3):
        ax_ba.plot(t, b_a[:, i], color=colors[i], label=f"b_a {labels[i]}")
        ax_ba.axhline(DUMMY_BIAS_ACCEL[i], color=colors[i], linestyle="--", alpha=0.4)
    ax_ba.set_title("Accel bias (dashed = independent reference)")
    ax_ba.set_ylabel("m/s^2")
    ax_ba.set_xlabel("Time (s)")
    ax_ba.legend(fontsize=8)

    ax_cov = axes["cov"]
    ax_cov.semilogy(t, history["sigma_theta"], label="theta (orientation), rad")
    ax_cov.semilogy(t, history["sigma_bg"], label="b_g (gyro bias), rad/s")
    ax_cov.semilogy(t, history["sigma_v"], label="v (velocity), m/s")
    ax_cov.semilogy(t, history["sigma_ba"], label="b_a (accel bias), m/s^2")
    ax_cov.semilogy(t, sigma_p, label="p (position), m")
    ax_cov.set_xlabel("Time (s)")
    ax_cov.set_ylabel("sqrt(trace(P_block)) (log scale)")
    ax_cov.set_title("Covariance magnitude per state block -- flat/shrinking while error grows = overconfidence")
    ax_cov.legend(fontsize=8, ncol=3)
    ax_cov.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=110)
    print(f"Saved plot to {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
