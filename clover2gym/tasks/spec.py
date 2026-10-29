"""Task specification: gates, track, constraints, timing.

These dataclasses are the single source of truth: the YAML handed to
participants, the static checker, the simulator and the generated problem
statement all read the same objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

import numpy as np

Vec3 = np.ndarray  # (3,) float64


# ---------------------------------------------------------------------- #
# Gates
# ---------------------------------------------------------------------- #
@dataclass
class Gate:
    """A racing gate.

    Parameters
    ----------
    center : (3,)  gate center (center of the opening) in world frame, m
    normal : (3,)  unit vector orthogonal to the gate plane; the track must be
                   crossed in the direction of the normal
    size   : (2,)  opening size (width, height) in m. For ``shape='rect'`` this
                   is the rectangle's full width/height; for ``shape='circle'``
                   the opening radius is ``size[0] / 2``.
    shape  :       'rect' or 'circle'
    """

    center: Vec3
    normal: Vec3
    size: Vec3
    shape: str = "rect"

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=float).reshape(3)
        n = np.asarray(self.normal, dtype=float).reshape(3)
        norm = float(np.linalg.norm(n))
        if norm < 1e-9:
            raise ValueError("gate normal must be non-zero")
        self.normal = n / norm
        self.size = np.asarray(self.size, dtype=float).reshape(2)
        if self.shape not in ("rect", "circle"):
            raise ValueError(f"unknown gate shape {self.shape!r}")
        if np.any(self.size <= 0):
            raise ValueError("gate size must be positive")

    @property
    def opening_radius(self) -> float:
        """Radius of the largest circle inscribed in the opening."""
        if self.shape == "circle":
            return float(self.size[0]) / 2
        return float(min(self.size[0], self.size[1])) / 2

    def to_dict(self) -> dict:
        return {
            "center": self.center.tolist(),
            "normal": self.normal.tolist(),
            "size": self.size.tolist(),
            "shape": self.shape,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        return cls(d["center"], d["normal"], d["size"], d.get("shape", "rect"))


# ---------------------------------------------------------------------- #
# Arena
# ---------------------------------------------------------------------- #
@dataclass
class Bounds:
    """Axis-aligned arena bounding box."""

    min: Vec3
    max: Vec3

    def __post_init__(self) -> None:
        self.min = np.asarray(self.min, dtype=float).reshape(3)
        self.max = np.asarray(self.max, dtype=float).reshape(3)
        if np.any(self.max <= self.min):
            raise ValueError("bounds max must be > min componentwise")

    def contains(self, points: np.ndarray, margin: float = 0.0) -> np.ndarray:
        pts = np.atleast_2d(points)
        return np.all((pts >= self.min + margin) & (pts <= self.max - margin), axis=1)

    def to_dict(self) -> dict:
        return {"min": self.min.tolist(), "max": self.max.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        return cls(d["min"], d["max"])


# ---------------------------------------------------------------------- #
# Constraints and timing
# ---------------------------------------------------------------------- #
@dataclass
class Constraints:
    """Model limits checked on the submitted trajectory (static stage).

    The thrust limits model the drone as a point mass with a thrust vector of
    bounded magnitude and direction: u(t) = a(t) - g, where g = (0, 0, -9.8).
    """

    v_max: float = 18.0            # m/s, norm of velocity
    a_max: float = 30.0            # m/s^2, norm of u (specific force)
    tilt_max_deg: float = 75.0     # max angle between u and the +z axis
    j_max: float = 80.0            # m/s^3, norm of jerk
    gate_margin: float = 0.10      # m, required inset from the opening edge
    collision_radius: float = 0.06  # m, drone collision cylinder radius
    collision_margin: float = 0.02  # m, extra clearance required beyond that
    start_tol: float = 0.05        # m, |p(0) - start_pos| tolerance
    start_vel_tol: float = 0.10    # m/s, |v(0)| tolerance
    z_min: float = 0.30            # m, floor clearance

    @property
    def clearance(self) -> float:
        """Required distance from the trajectory to any obstacle."""
        return self.collision_radius + self.collision_margin

    def to_dict(self) -> dict:
        return {
            "v_max": self.v_max,
            "a_max": self.a_max,
            "tilt_max_deg": self.tilt_max_deg,
            "j_max": self.j_max,
            "gate_margin": self.gate_margin,
            "collision_radius": self.collision_radius,
            "collision_margin": self.collision_margin,
            "start_tol": self.start_tol,
            "start_vel_tol": self.start_vel_tol,
            "z_min": self.z_min,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        return cls(**d)


@dataclass
class Timing:
    """Format requirements for the submitted trajectory samples."""

    dt_max: float = 0.01          # s, sampling period upper bound
    t_max: float = 60.0           # s, trajectory duration upper bound
    tail_after_last_gate: float = 0.3  # s, trajectory must extend past last gate

    def to_dict(self) -> dict:
        return {"dt_max": self.dt_max, "t_max": self.t_max, "tail_after_last_gate": self.tail_after_last_gate}

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------- #
# Track and task
# ---------------------------------------------------------------------- #
@dataclass
class Track:
    name: str
    gates: list[Gate]
    start_pos: Vec3
    start_yaw: float = 0.0
    arena: Bounds | None = None

    def __post_init__(self) -> None:
        self.start_pos = np.asarray(self.start_pos, dtype=float).reshape(3)
        if self.arena is None:
            pts = np.vstack([g.center for g in self.gates] + [self.start_pos])
            self.arena = Bounds(pts.min(axis=0) - 3.0, pts.max(axis=0) + 3.0)
        elif not isinstance(self.arena, Bounds):
            self.arena = Bounds(*self.arena)
        if len(self.gates) < 1:
            raise ValueError("track needs at least one gate")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "gates": [g.to_dict() for g in self.gates],
            "start_pos": self.start_pos.tolist(),
            "start_yaw": float(self.start_yaw),
            "arena": self.arena.to_dict() if self.arena else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        return cls(
            name=d["name"],
            gates=[Gate.from_dict(g) for g in d["gates"]],
            start_pos=d["start_pos"],
            start_yaw=d.get("start_yaw", 0.0),
            arena=Bounds.from_dict(d["arena"]) if d.get("arena") else None,
        )


@dataclass
class MotionPlanningTask:
    """The full task handed to a participant's ``solve()``."""

    track: Track
    constraints: Constraints = field(default_factory=Constraints)
    timing: Timing = field(default_factory=Timing)
    drone: str = "racer"
    controller: str = "geometric"  # built-in tracking controller (sim-level);
    # "geometric_ff" (plan-derived rate feedforward) is experimental: it helps
    # on smooth plans but destabilizes tracking on plans with fast tilt swings

    # Convenience accessors used by participants.
    @property
    def gates(self) -> list[Gate]:
        return self.track.gates

    @property
    def start_pos(self) -> Vec3:
        return self.track.start_pos

    @property
    def arena(self) -> Bounds:
        assert self.track.arena is not None
        return self.track.arena

    def to_dict(self) -> dict:
        return {
            "track": self.track.to_dict(),
            "constraints": self.constraints.to_dict(),
            "timing": self.timing.to_dict(),
            "drone": self.drone,
            "controller": self.controller,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        return cls(
            track=Track.from_dict(d["track"]),
            constraints=Constraints.from_dict(d.get("constraints", {})),
            timing=Timing.from_dict(d.get("timing", {})),
            drone=d.get("drone", "racer"),
            controller=d.get("controller", "geometric"),
        )

    def to_yaml(self, path) -> None:
        import yaml

        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def from_yaml(cls, path) -> Self:
        import yaml

        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))
