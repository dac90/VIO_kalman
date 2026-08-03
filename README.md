# VIO_kalman

A personal project undertaken immediately after my graduation, this project was developed to address the issues faced by VIO_first.

This is a monocular Multi-State Constraint Kalman Filter (MSCKF) visual-inertial odometry pipeline,
built from scratch in Python and evaluated on the EuRoC dataset. It is complete with a test suite, and a selection of demo programms for manual validation. 

This is also my first project to employ AI-accelerated programming for development,
though not for design decisions or testing, which was done by hand to correct several mistakes created by the agent.

Due to the lack of any objective yaw-axis information, the system inevitably deviates from ground truth. Particular difficulties also occur during sudden acceleration or camera "trembling". A future SLAM-based VIO system will be developed to address this.

## The MSCKF
The MSCKF is *structureless*: feature 3D positions never enter the filter state.
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
