"""Multi-view feature triangulation: linear (DLT) initialization + nonlinear
(Gauss-Newton) refinement of a 3D point from its bearing observations across
several camera poses.

This is where lens distortion finally gets removed (see feature_tracker.py):
KLT tracks operate on raw distorted pixels, but triangulation needs
undistorted normalized image-plane coordinates (bearing = [x, y, 1] in the
camera's own frame, up to scale) for the linear-algebra/geometry to be valid.
"""
import os

import cv2
import numpy as np
import yaml

from quaternion_utils import quaternion_to_rotation_matrix

MAV0_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "machine_hall", "MH_01_easy", "MH_01_easy", "mav0")


def load_cam0_intrinsics(mav0_dir=MAV0_DIR):
    """(K, dist_coeffs) for cam0: K is the 3x3 pinhole intrinsic matrix, dist_coeffs is [k1, k2, p1, p2]."""
    path = os.path.join(mav0_dir, "cam0", "sensor.yaml")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    fu, fv, cu, cv_ = cfg["intrinsics"]
    K = np.array([[fu, 0.0, cu], [0.0, fv, cv_], [0.0, 0.0, 1.0]])
    dist_coeffs = np.array(cfg["distortion_coefficients"], dtype=float)
    return K, dist_coeffs


def undistort_normalized(pixels, K, dist_coeffs):
    """Raw distorted pixel coords (N,2) -> undistorted normalized image-plane coords (N,2), i.e. bearing = [x, y, 1]."""
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(pixels, K, dist_coeffs)
    return undistorted.reshape(-1, 2)


def _camera_projection_matrix(q_world_cam, p_world_cam):
    """3x4 camera-from-world projection matrix [R_cw | t_cw] such that lambda*[x,y,1] = P @ [X_world; 1]."""
    R_wc = quaternion_to_rotation_matrix(q_world_cam)
    R_cw = R_wc.T
    t_cw = -R_cw @ p_world_cam
    return np.concatenate([R_cw, t_cw.reshape(3, 1)], axis=1)


def linear_triangulate(camera_poses, bearings):
    """Initial DLT estimate of the 3D world point from >=2 views.

    camera_poses: list of (q_world_cam, p_world_cam), the world-from-camera
    orientation/position of each observing camera clone.
    bearings: matching list/array of (x, y) undistorted normalized image-plane
    coordinates (see undistort_normalized).

    Returns the triangulated point in world coordinates.
    """
    rows = []
    for (q_world_cam, p_world_cam), (x, y) in zip(camera_poses, bearings):
        P = _camera_projection_matrix(q_world_cam, p_world_cam)
        rows.append(x * P[2, :] - P[0, :])
        rows.append(y * P[2, :] - P[1, :])
    A = np.array(rows)

    _, _, Vt = np.linalg.svd(A)
    X_homogeneous = Vt[-1]
    return X_homogeneous[:3] / X_homogeneous[3]


def _project(X_world, q_world_cam, p_world_cam):
    """Predicted (x, y) normalized image-plane coords and the camera-frame point Xc."""
    R_wc = quaternion_to_rotation_matrix(q_world_cam)
    Xc = R_wc.T @ (X_world - p_world_cam)
    return np.array([Xc[0] / Xc[2], Xc[1] / Xc[2]]), Xc


def reprojection_residuals(X_world, camera_poses, bearings):
    """Stacked (predicted - observed) residual vector (2 * n_views,)."""
    residuals = []
    for (q_world_cam, p_world_cam), (x, y) in zip(camera_poses, bearings):
        (u, v), _ = _project(X_world, q_world_cam, p_world_cam)
        residuals.extend([u - x, v - y])
    return np.array(residuals)


def refine_triangulation(X_world_init, camera_poses, bearings, n_iters=10, convergence_tol=1e-10):
    """Gauss-Newton refinement of a triangulated point, minimizing squared reprojection error."""
    X = np.array(X_world_init, dtype=float)

    for _ in range(n_iters):
        J_rows = []
        r_rows = []
        for (q_world_cam, p_world_cam), (x, y) in zip(camera_poses, bearings):
            R_wc = quaternion_to_rotation_matrix(q_world_cam)
            R_cw = R_wc.T
            (u, v), Xc = _project(X, q_world_cam, p_world_cam)

            d_uv_d_Xc = np.array([
                [1.0 / Xc[2], 0.0, -Xc[0] / Xc[2] ** 2],
                [0.0, 1.0 / Xc[2], -Xc[1] / Xc[2] ** 2],
            ])
            J_rows.append(d_uv_d_Xc @ R_cw)
            r_rows.append([u - x, v - y])

        J = np.concatenate(J_rows, axis=0)
        r = np.concatenate(r_rows, axis=0)

        dX, *_ = np.linalg.lstsq(J, -r, rcond=None)
        X = X + dX
        if np.linalg.norm(dX) < convergence_tol:
            break

    return X


def triangulate_feature(camera_poses, bearings, n_iters=10):
    """Linear (DLT) initialization followed by Gauss-Newton refinement. Returns the world-frame point."""
    X_init = linear_triangulate(camera_poses, bearings)
    return refine_triangulation(X_init, camera_poses, bearings, n_iters=n_iters)


def check_parallax(camera_poses, bearings, min_orthogonal_translation=0.2):
    """True if there's enough triangulation-observable baseline between the first and last
    observing camera pose (matches MSCKF-VIO's Feature::checkMotion).

    Pure translation *along* the feature's initial bearing ray gives zero parallax -- depth
    along that ray stays unobservable no matter how far the camera travels -- so only the
    translation component orthogonal to the ray counts as usable baseline.
    """
    (q_first, p_first) = camera_poses[0]
    (_, p_last) = camera_poses[-1]

    x, y = bearings[0]
    ray_cam = np.array([x, y, 1.0])
    ray_cam = ray_cam / np.linalg.norm(ray_cam)
    ray_world = quaternion_to_rotation_matrix(q_first) @ ray_cam

    translation = p_last - p_first
    parallel = np.dot(translation, ray_world) * ray_world
    orthogonal = translation - parallel
    return np.linalg.norm(orthogonal) > min_orthogonal_translation


def check_cheirality(X_world, camera_poses):
    """True if the triangulated point is in front of every observing camera (positive depth).

    A linear (DLT) + Gauss-Newton solution isn't constrained to end up in front of every
    camera -- near-degenerate geometry can converge to a point behind one, which is an
    unambiguous sign the solution is wrong regardless of how small its reprojection
    residual happens to look.
    """
    for q_world_cam, p_world_cam in camera_poses:
        _, Xc = _project(X_world, q_world_cam, p_world_cam)
        if Xc[2] <= 0:
            return False
    return True


def triangulate_and_validate(camera_poses, bearings, min_orthogonal_translation=0.2,
                              max_reprojection_error=None, n_iters=10):
    """triangulate_feature, but returns None instead of a point for degenerate geometry.

    Three checks, all independent of the filter's own covariance P (unlike the chi-square
    gate in msckf_update.py, whose threshold scales with P -- so once P is even a little
    too large, a bad point can still pass it, adding more error, growing P further, and so
    on): insufficient parallax (check_parallax), a solution that ends up behind a camera
    (check_cheirality), and -- if max_reprojection_error is given -- a refined reprojection
    error too large to trust even though the first two checks passed.
    """
    if not check_parallax(camera_poses, bearings, min_orthogonal_translation):
        return None

    X_init = linear_triangulate(camera_poses, bearings)
    X_world = refine_triangulation(X_init, camera_poses, bearings, n_iters=n_iters)

    if not check_cheirality(X_world, camera_poses):
        return None

    if max_reprojection_error is not None:
        residuals = reprojection_residuals(X_world, camera_poses, bearings).reshape(-1, 2)
        if np.max(np.linalg.norm(residuals, axis=1)) > max_reprojection_error:
            return None

    return X_world
