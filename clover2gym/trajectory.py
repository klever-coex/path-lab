"""Participant-facing trajectory representation.

A trajectory is a sequence of position samples at a uniform time step. The
framework fits a clamped cubic spline through the samples, so velocity,
acceleration and jerk are well defined and identical for every participant.
Clamped boundary conditions mean v(0) = v(T) = 0 (start and end from hover).
"""
from __future__ import annotations

from typing import Self

import numpy as np


class Trajectory:
    """Position trajectory defined by uniform samples.

    Parameters
    ----------
    positions : (N, 3) array of positions in m, 2 <= N <= 1e6
    dt        : sampling period in s
    """

    MAX_SAMPLES = 1_000_000

    def __init__(self, positions: np.ndarray, dt: float):
        from scipy.interpolate import CubicSpline

        positions = np.asarray(positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < 2:
            raise ValueError("positions must be an (N, 3) array with N >= 2")
        if positions.shape[0] > self.MAX_SAMPLES:
            raise ValueError(
                f"{positions.shape[0]} samples exceeds MAX_SAMPLES={self.MAX_SAMPLES}; "
                "reduce N or increase dt"
            )
        if not (np.isfinite(positions).all()):
            raise ValueError("positions contain non-finite values")
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.positions = positions
        self.dt = float(dt)
        self.times = np.arange(len(positions)) * self.dt
        # clamped start (v(0) = 0: the drone takes off from hover); not-a-knot
        # end (nothing is forced at T: the race is over at the last gate, and
        # forcing either v(T) = 0 or a(T) = 0 fights planner profiles)
        self._spline = CubicSpline(
            self.times, positions, bc_type=((1, np.zeros(3)), "not-a-knot"), axis=0
        )

    # ------------------------------------------------------------------ #
    # constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_waypoints(cls, waypoints: np.ndarray, times: np.ndarray | None = None, dt: float = 0.01, cruise_speed: float = 3.0) -> Self:
        """Build a trajectory by spline-fitting waypoints.

        If ``times`` is None, waypoint times follow a trapezoidal speed
        profile: hover at the first/last waypoint and ``cruise_speed`` at the
        interior ones, so segment durations account for acceleration.
        Participants are expected to supply their own time parametrization for
        faster laps.
        """
        from scipy.interpolate import CubicSpline
        
        waypoints = np.asarray(waypoints, dtype=float)
        if waypoints.ndim != 2 or waypoints.shape[1] != 3 or waypoints.shape[0] < 2:
            raise ValueError("waypoints must be an (N, 3) array with N >= 2")
        if times is None:
            seg = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
            speeds = np.array([0.0] + [cruise_speed] * (len(waypoints) - 2) + [0.0])
            # constant-acceleration time per segment (trapezoid rule); the
            # floor keeps degenerate cases (e.g. a single segment between two
            # hover waypoints) from producing unbounded durations
            durations = 2.0 * seg / np.maximum(speeds[:-1] + speeds[1:], 1.0)
            times = np.concatenate(([0.0], np.cumsum(durations)))
        times = np.asarray(times, dtype=float)
        # Snap waypoint times onto the dt grid: knots that fall between samples
        # would make the refit spline jitter in its higher derivatives.
        times = np.maximum.accumulate(np.round(times / dt)) * dt
        if times.ndim != 1 or len(times) != len(waypoints) or np.any(np.diff(times) <= 0):
            raise ValueError("times must be increasing and match the waypoints")
        spline = CubicSpline(times, waypoints, bc_type="clamped", axis=0)
        n_samples = int(np.ceil(times[-1] / dt)) + 1
        sample_times = np.arange(n_samples) * dt
        sample_times = sample_times[sample_times <= times[-1] + 1e-9]
        return cls(spline(sample_times), dt)

    # ------------------------------------------------------------------ #
    # evaluation
    # ------------------------------------------------------------------ #
    @property
    def duration(self) -> float:
        return self.times[-1]

    def __len__(self) -> int:
        return len(self.positions)

    def _clip_t(self, t):
        return np.clip(t, 0.0, self.duration)

    def position(self, t) -> np.ndarray:
        return self._spline(self._clip_t(t))

    def velocity(self, t) -> np.ndarray:
        return self._spline(self._clip_t(t), 1)

    def acceleration(self, t) -> np.ndarray:
        return self._spline(self._clip_t(t), 2)

    def jerk(self, t) -> np.ndarray:
        return self._spline(self._clip_t(t), 3)

    def resample(self, dt: float) -> Self:
        n = int(np.floor(self.duration / dt)) + 1
        t = np.arange(n) * dt
        return type(self)(self.position(t), dt)

    # ------------------------------------------------------------------ #
    # yaw profile (framework-defined)
    # ------------------------------------------------------------------ #
    def yaw_profile(self, start_yaw: float = 0.0, min_speed: float = 0.5, smoothing: float = 1.2) -> tuple[np.ndarray, np.ndarray]:
        """Yaw along the trajectory: heading of the horizontal velocity.

        Where |v_xy| < min_speed (start/end hover) the last computed heading is
        held. The heading is unwrapped and Gaussian-smoothed over ``smoothing``
        seconds so that the target yaw is always continuous — a discontinuous
        yaw command would destabilize the tracking controller. Returns
        (times, yaw) suitable for ``np.interp``.
        """
        from scipy.ndimage import gaussian_filter1d

        v = self.velocity(self.times)
        yaw = np.empty(len(self.times))
        last = float(start_yaw)
        for i in range(len(self.times)):
            if np.linalg.norm(v[i, :2]) >= min_speed:
                target = np.arctan2(v[i, 1], v[i, 0])
                delta = (target - last + np.pi) % (2 * np.pi) - np.pi
                last = last + delta
            yaw[i] = last
        yaw = gaussian_filter1d(np.unwrap(yaw), sigma=smoothing / self.dt, mode="nearest")
        yaw = yaw - (yaw[0] - start_yaw)  # pin the start exactly to start_yaw
        return self.times, yaw


def load_trajectory(obj) -> Trajectory:
    """Coerce a participant's return value into a Trajectory."""
    if isinstance(obj, Trajectory):
        return obj
    if isinstance(obj, tuple) and len(obj) == 2:
        return Trajectory(obj[0], obj[1])
    raise TypeError(
        "solve() must return a clover2gym.Trajectory (or a (positions, dt) tuple)"
    )
