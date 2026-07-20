"""Sequential (monotonic-timestamp) interpolators shared across datasets."""
import numpy as np

from quaternion_utils import slerp


class SequentialInterpolator:
    """Linear interpolator over a monotonically increasing timestamp series.

    Keeps a pointer to the last bracket used so that repeated calls with
    increasing query timestamps resume searching from there instead of
    restarting from the beginning of the series each time.
    """

    def __init__(self, timestamps, positions, exact_match_tolerance=0):
        # keep timestamps in their native (typically int64) dtype: casting huge
        # nanosecond-scale integers to float64 loses up to ~256ns of precision,
        # since float64 only has 53 bits of exact integer range
        self.timestamps = np.asarray(timestamps)
        self.positions = np.asarray(positions, dtype=float)
        self.exact_match_tolerance = exact_match_tolerance
        self._idx = 0

    def interpolate(self, t):
        ts = self.timestamps
        n = len(ts)

        if t <= ts[0]:
            self._idx = 0
            return self.positions[0].copy()
        if t >= ts[-1]:
            self._idx = n - 2
            return self.positions[-1].copy()

        if t < ts[self._idx]:
            self._idx = 0
        while self._idx < n - 2 and ts[self._idx + 1] < t:
            self._idx += 1

        t0, t1 = ts[self._idx], ts[self._idx + 1]
        d0, d1 = abs(t - t0), abs(t - t1)

        # snap to whichever bracket endpoint is closer, not just t0 first:
        # otherwise a tolerance wider than half the sample spacing can snap
        # an exact match at t1 to the wrong (t0) neighbor instead
        if min(d0, d1) <= self.exact_match_tolerance:
            return (self.positions[self._idx] if d0 <= d1 else self.positions[self._idx + 1]).copy()

        alpha = (t - t0) / (t1 - t0)
        return self.positions[self._idx] + alpha * (self.positions[self._idx + 1] - self.positions[self._idx])


class QuaternionSequentialInterpolator:
    """Like SequentialInterpolator, but spherically interpolates (slerps) unit quaternions.

    Linearly interpolating Euler angles (or raw quaternion components) can
    produce artifacts near angle wraparound / antipodal quaternions -
    slerp avoids both by always taking the shortest rotational path between
    samples and staying on the unit quaternion manifold throughout.
    """

    def __init__(self, timestamps, quaternions, exact_match_tolerance=0):
        # see SequentialInterpolator for why timestamps keep their native dtype
        self.timestamps = np.asarray(timestamps)
        q = np.asarray(quaternions, dtype=float)
        self.quaternions = q / np.linalg.norm(q, axis=1, keepdims=True)
        self.exact_match_tolerance = exact_match_tolerance
        self._idx = 0

    def interpolate(self, t):
        ts = self.timestamps
        n = len(ts)

        if t <= ts[0]:
            self._idx = 0
            return self.quaternions[0].copy()
        if t >= ts[-1]:
            self._idx = n - 2
            return self.quaternions[-1].copy()

        if t < ts[self._idx]:
            self._idx = 0
        while self._idx < n - 2 and ts[self._idx + 1] < t:
            self._idx += 1

        t0, t1 = ts[self._idx], ts[self._idx + 1]
        d0, d1 = abs(t - t0), abs(t - t1)

        if min(d0, d1) <= self.exact_match_tolerance:
            return (self.quaternions[self._idx] if d0 <= d1 else self.quaternions[self._idx + 1]).copy()

        alpha = (t - t0) / (t1 - t0)
        return slerp(self.quaternions[self._idx], self.quaternions[self._idx + 1], alpha)
