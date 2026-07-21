"""MSCKF reprojection measurement model: per-observation residual and
Jacobians, and the null-space projection that lets a triangulated feature
constrain the filter's camera-clone poses without the feature itself ever
entering the state (the core "MSCKF trick").

Error-state convention matches msckf_state.py exactly: delta_theta_cam is a
GLOBAL/world-frame perturbation with R_true = (I + [delta_theta]x) @ R_hat;
delta_p_cam and the feature's delta_X_world are plain additive differences.
"""
import numpy as np

from msckf_state import N_CLONE_ERROR, N_IMU_ERROR
from quaternion_utils import quaternion_to_rotation_matrix, skew_symmetric


def reprojection_residual_and_jacobians(X_world, q_world_cam, p_world_cam, bearing_obs):
    """Residual and Jacobians for one feature observation from one camera clone.

    Returns (r (2,), H_clone (2x6), H_f (2x3)):
      r = bearing_obs - h(X_world, camera pose)     (observed minus predicted;
          note this is the opposite sign convention from triangulation.py's
          refine_triangulation, which isn't an EKF and doesn't need to match)
      H_clone = dr/d[delta_theta_cam, delta_p_cam]   (this clone's own 6-dim error block)
      H_f = dr/d(delta_X_world)                      (feature-position error)
    """
    R_wc = quaternion_to_rotation_matrix(q_world_cam)
    R_cw = R_wc.T
    Xc = R_cw @ (np.asarray(X_world) - np.asarray(p_world_cam))

    h = np.array([Xc[0] / Xc[2], Xc[1] / Xc[2]])
    r = np.asarray(bearing_obs, dtype=float) - h

    dh_dXc = np.array([
        [1.0 / Xc[2], 0.0, -Xc[0] / Xc[2] ** 2],
        [0.0, 1.0 / Xc[2], -Xc[1] / Xc[2] ** 2],
    ])

    H_f = -dh_dXc @ R_cw

    lever = np.asarray(X_world) - np.asarray(p_world_cam)
    dr_dtheta = -dh_dXc @ R_cw @ skew_symmetric(lever)
    dr_dp = dh_dXc @ R_cw  # == -H_f: translating the camera is geometrically
                           # equivalent to translating the feature the other way

    H_clone = np.concatenate([dr_dtheta, dr_dp], axis=1)
    return r, H_clone, H_f


def stack_feature_observations(X_world, camera_poses, bearings, clone_indices, n_clones):
    """Stack every observation of one feature into full-width residual/Jacobian blocks.

    camera_poses/bearings: parallel lists, one entry per observation.
    clone_indices: which of the n_clones camera clones (0-based, matching
    MSCKFState.clone_positions/clone_orientations order) each observation
    corresponds to.

    Returns (r (2M,), H_x (2M, 15+6*n_clones), H_f (2M, 3)).
    """
    n_obs = len(bearings)
    n_state = N_IMU_ERROR + N_CLONE_ERROR * n_clones

    r = np.zeros(2 * n_obs)
    H_x = np.zeros((2 * n_obs, n_state))
    H_f = np.zeros((2 * n_obs, 3))

    for k, ((q_cam, p_cam), bearing, clone_idx) in enumerate(zip(camera_poses, bearings, clone_indices)):
        r_k, H_clone_k, H_f_k = reprojection_residual_and_jacobians(X_world, q_cam, p_cam, bearing)
        col = N_IMU_ERROR + N_CLONE_ERROR * clone_idx
        r[2 * k:2 * k + 2] = r_k
        H_x[2 * k:2 * k + 2, col:col + N_CLONE_ERROR] = H_clone_k
        H_f[2 * k:2 * k + 2, :] = H_f_k

    return r, H_x, H_f


def null_space_project(r, H_x, H_f):
    """Project (r, H_x) onto the left null space of H_f, eliminating feature-position dependence.

    Uses a full QR decomposition of H_f (2M x 3, generically rank 3): its
    first 3 columns of Q span H_f's range, the remaining (2M-3) span the
    orthogonal complement -- exactly the left null space, i.e.
    null_basis.T @ H_f == 0. Because those columns are orthonormal, isotropic
    measurement noise on r stays isotropic on the projected residual too.

    Returns (r_o (2M-3,), H_o (2M-3, n_state)). Requires at least 2 observations
    (2M >= 4) for the null space to be non-trivial.
    """
    n_obs2 = H_f.shape[0]
    rank_f = H_f.shape[1]
    if n_obs2 <= rank_f:
        raise ValueError(f"need at least {rank_f + 1} observation-rows (>= 2 observations) to project out "
                          f"a {rank_f}-dim feature position, got {n_obs2}")

    Q, _ = np.linalg.qr(H_f, mode="complete")
    null_basis = Q[:, rank_f:]
    return null_basis.T @ r, null_basis.T @ H_x
