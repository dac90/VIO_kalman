import numpy as np

from interpolation import QuaternionSequentialInterpolator, SequentialInterpolator
from quaternion_utils import axis_angle_to_quat, quat_multiply, slerp


def test_sequential_interpolator_linear_midpoint():
    ts = np.array([0, 10, 20], dtype=np.int64)
    positions = np.array([[0.0, 0.0, 0.0], [10.0, 20.0, -10.0], [20.0, 40.0, -20.0]])
    interp = SequentialInterpolator(ts, positions)
    assert np.allclose(interp.interpolate(5), [5.0, 10.0, -5.0])
    assert np.allclose(interp.interpolate(15), [15.0, 30.0, -15.0])


def test_sequential_interpolator_exact_sample_matches_exactly():
    ts = np.array([0, 10, 20], dtype=np.int64)
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    interp = SequentialInterpolator(ts, positions)
    assert np.allclose(interp.interpolate(10), [1.0, 2.0, 3.0])


def test_sequential_interpolator_clamps_outside_range():
    ts = np.array([0, 10], dtype=np.int64)
    positions = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    interp = SequentialInterpolator(ts, positions)
    assert np.allclose(interp.interpolate(-100), [1.0, 1.0, 1.0])
    assert np.allclose(interp.interpolate(1000), [2.0, 2.0, 2.0])


def test_sequential_interpolator_exact_match_tolerance_snaps_to_nearest():
    ts = np.array([0, 100, 200], dtype=np.int64)
    positions = np.array([[0.0], [1.0], [2.0]])
    interp = SequentialInterpolator(ts, positions, exact_match_tolerance=5)
    # within tolerance of t=100 -> snap exactly to that sample instead of interpolating
    assert np.allclose(interp.interpolate(103), [1.0])
    assert np.allclose(interp.interpolate(97), [1.0])


def test_sequential_interpolator_monotonic_queries_advance_pointer_correctly():
    ts = np.arange(0, 1000, 10, dtype=np.int64)
    positions = ts.astype(float).reshape(-1, 1)
    interp = SequentialInterpolator(ts, positions)
    # querying with a strictly increasing sequence of timestamps (the
    # documented usage pattern) should give the same answer as a fresh
    # interpolator queried once at that same point
    for t in [5, 55, 105, 500, 995]:
        fresh = SequentialInterpolator(ts, positions)
        assert np.allclose(interp.interpolate(t), fresh.interpolate(t))


def test_quaternion_interpolator_matches_slerp():
    ts = np.array([0, 100], dtype=np.int64)
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    q1 = axis_angle_to_quat(np.array([0.0, 0.0, np.pi / 2]))
    interp = QuaternionSequentialInterpolator(ts, np.array([q0, q1]))
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        t = alpha * 100
        expected = slerp(q0, q1, alpha)
        got = interp.interpolate(t)
        if np.dot(got, expected) < 0:
            got = -got
        assert np.allclose(got, expected, atol=1e-9)


def test_quaternion_interpolator_normalizes_input():
    ts = np.array([0, 10], dtype=np.int64)
    q0 = np.array([2.0, 0.0, 0.0, 0.0])  # deliberately not unit-norm
    q1 = np.array([0.0, 3.0, 0.0, 0.0])
    interp = QuaternionSequentialInterpolator(ts, np.array([q0, q1]))
    assert np.isclose(np.linalg.norm(interp.quaternions[0]), 1.0)
    assert np.isclose(np.linalg.norm(interp.quaternions[1]), 1.0)


def test_quaternion_interpolator_clamps_outside_range():
    ts = np.array([0, 10], dtype=np.int64)
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    q1 = axis_angle_to_quat(np.array([0.1, 0.0, 0.0]))
    interp = QuaternionSequentialInterpolator(ts, np.array([q0, q1]))
    assert np.allclose(interp.interpolate(-5), q0)
    assert np.allclose(interp.interpolate(50), q1)
