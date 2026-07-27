# VIO_kalman

A monocular Multi-State Constraint Kalman Filter (MSCKF) visual-inertial
odometry pipeline, built from scratch in Python and evaluated on the EuRoC
`MH_01_easy` dataset. This covers the full stack: feature detection/tracking,
triangulation, the structureless EKF update, and marginalization -- not just
the filter math.

MSCKF is *structureless*: feature 3D positions never enter the filter state.
Each feature is triangulated fresh from the current sliding window of camera
poses, used once to constrain those poses via a null-space-projected EKF
update, then discarded. Only the IMU state (orientation, velocity, position,
gyro/accel bias) and a bounded window of camera-pose "clones" are ever
estimated directly.

## Setup

```bash
pip install -r requirements.txt
```

Download the [EuRoC MAV dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)'s
`MH_01_easy` sequence (ASL format) and extract it so the following path
exists relative to the repo root:

```
machine_hall/MH_01_easy/MH_01_easy/mav0/
    cam0/            # images + timestamps
    cam1/            # unused (monocular only)
    imu0/            # gyro + accel + noise parameters
    leica0/          # raw external position tracker (reference only)
    state_groundtruth_estimate0/   # batch-optimized ground truth
```

`machine_hall/` is gitignored -- it's a ~1.5GB dataset, not part of the repo.

## Running it

```bash
python demos/demo_full_pipeline.py         # run the filter, print ATE/RPE, plot the result
python demos/demo_full_pipeline_video.py   # same, rendered to demos/full_pipeline.mp4
python demos/demo_divergence_analysis.py   # longer diagnostic run, see below
python -m pytest                           # full test suite (155 tests)
```

## Project structure

Core pipeline (repo root):

| Module | Responsibility |
|---|---|
| `msckf_pipeline.py` | Top-level `MSCKFPipeline` class -- wires everything below into one continuously-runnable filter |
| `msckf_state.py` | State representation (`MSCKFState`), IMU error-state propagation, clone augmentation/marginalization |
| `msckf_update.py` | EKF update (chi-square gating, QR compression, Joseph-form covariance update), plus the ZUPT/ZARU/gravity-alignment pseudo-measurements |
| `measurement_model.py` | Per-observation reprojection residual/Jacobians and the null-space projection that eliminates feature-position dependence |
| `triangulation.py` | Multi-view DLT + Gauss-Newton triangulation, with parallax/cheirality/reprojection-error validation |
| `feature_tracker.py` | Shi-Tomasi + pyramidal KLT feature tracking frontend |
| `imu_propagation.py` | Raw strapdown INS integration (used both by the filter and as a standalone dead-reckoning baseline) |
| `quaternion_utils.py` | Quaternion/rotation math shared throughout |
| `frames.py` | Sensor-to-body frame conversions via a `sensor.yaml` extrinsic |
| `ground_truth.py` | Ground-truth loading/interpolation for evaluation |
| `interpolation.py` | Generic monotonic-timestamp interpolators (linear + slerp) |
| `plot.py` | Standalone raw IMU data visualization |

Supporting directories:

- `demos/` -- the current, runnable capstone demos (see above).
- `tests/` -- the pytest suite, one file per core module.
- `test_demo/` -- earlier per-build-stage demo scripts, kept for reference; superseded by `demos/`.

`old.py` is an unrelated prior project (PX4/Pixhawk drone log analysis) that
happens to share this repo -- not part of the MSCKF pipeline.

## How it works

1. **Propagate** (`msckf_state.propagate`, every IMU sample): strapdown
   integration of the nominal state, plus error-state covariance propagation.
2. **Augment** (`msckf_state.augment`, every camera frame): clone the current
   IMU pose into the sliding window as a new camera pose.
3. **Track** (`feature_tracker.FeatureTracker`): KLT-track existing features,
   replenish with fresh Shi-Tomasi corners as needed.
4. **Triangulate + validate** (`triangulation.triangulate_and_validate`): once
   a track ends (or is about to be evicted), triangulate its 3D position from
   the observing camera poses, rejecting degenerate geometry (insufficient
   parallax, behind-camera solutions, excessive reprojection error).
5. **Gate + update** (`msckf_update`): null-space-project out the feature's
   own position dependence, chi-square gate the residual, then apply a
   Joseph-form EKF update using every accepted feature from that frame at once.
6. **Marginalize** (`msckf_state.marginalize_clones`): drop the oldest clones
   once the window exceeds `max_clones` (a strict, always-enforced cap).

**Stationary-period handling**: a camera that isn't moving gets zero parallax
on everything it sees, so vision updates become actively harmful rather than
just unhelpful. `_update_stationary_state` (a GLRT/SHOE detector on raw IMU
statistics) triggers a trio of direct IMU pseudo-measurements when the
platform is at rest -- zero-velocity (ZUPT), zero-angular-rate (ZARU), and
gravity/tilt alignment -- all independent of vision. See
`msckf_pipeline.py`'s module docstring for the full mechanism.

## Current results and known limitations

Over a 30s window (real flight, a ~24s stationary hold, resumed motion):
**ATE = 0.691m, RPE@1s = 0.146m** (`demos/demo_full_pipeline.py`). The filter
stays sub-meter and stable through the entire stationary hold, but still
eventually diverges on renewed motion.

That residual divergence is the classic **EKF-SLAM/MSCKF consistency
problem**: directions that are fundamentally unobservable in monocular VIO
(global position, global yaw) can become falsely confident when a feature is
triangulated against the filter's own, already-slightly-wrong camera poses --
the reprojection residual looks small and "consistent" by construction, so
the EKF update reads as informative and shrinks the covariance further,
reinforcing the drift instead of correcting it. `demos/demo_divergence_analysis.py`
runs a longer 60s window and plots real error against the filter's own
predicted uncertainty side by side, making this overconfidence directly
visible.

The textbook fix is **Observability-Constrained EKF / First-Estimates
Jacobians** (evaluating Jacobians at a fixed linearization point per state
variable, rather than re-linearizing at the latest estimate every update) --
a substantially larger undertaking than anything implemented here, and left
as the natural next step. Everything currently in place (the bounded sliding
window, the covariance floor, and the ZUPT/ZARU/gravity-alignment trio)
mitigates specific symptoms of this same underlying problem without fixing
it at the root.
