"""Static (simulation-free) validation of a submitted trajectory.

All limits come from the task's ``Constraints``/``Timing`` objects — the same
objects the problem statement is generated from, so the rules participants
read are exactly the rules enforced here.
"""
from __future__ import annotations

import numpy as np

from ..tasks.spec import MotionPlanningTask
from ..trajectory import Trajectory
from ..util.geometry import distance_to_gate, find_gate_crossings
from .report import CheckReport, Violation

G_VEC = np.array([0.0, 0.0, -9.8])


class StaticChecker:
    def __init__(self, task: MotionPlanningTask):
        self.task = task
        self.cons = task.constraints
        self.timing = task.timing

    def check(self, traj: Trajectory) -> CheckReport:
        violations: list[Violation] = []
        stats: dict = {}

        self._check_format(traj, violations, stats)
        traj = traj.resample(min(traj.dt, self.timing.dt_max, 0.01))
        times = traj.times
        pos = traj.positions
        vel = traj.velocity(times)
        acc = traj.acceleration(times)
        jerk = traj.jerk(times)

        self._check_start(traj, violations)
        self._check_bounds(pos, violations)
        self._check_limits(times, vel, acc, jerk, violations, stats)
        gate_times = self._check_gates(traj, violations, stats)
        self._check_clearance(times, pos, violations, stats)
        if gate_times:
            self._check_tail(traj, gate_times[-1], violations, stats)

        return CheckReport(stage="static", ok=not violations, violations=violations, stats=stats)

    # ------------------------------------------------------------------ #
    def _check_format(self, traj: Trajectory, out: list[Violation], stats: dict) -> None:
        stats["n_samples"] = len(traj)
        stats["dt"] = traj.dt
        stats["duration"] = traj.duration
        if traj.dt > self.timing.dt_max + 1e-12:
            out.append(
                Violation(
                    "format_dt",
                    f"sampling period {traj.dt:.4f} s exceeds dt_max",
                    value=traj.dt,
                    limit=self.timing.dt_max,
                )
            )
        if traj.duration > self.timing.t_max:
            out.append(
                Violation(
                    "format_duration",
                    f"duration {traj.duration:.2f} s exceeds t_max",
                    value=traj.duration,
                    limit=self.timing.t_max,
                )
            )

    def _check_start(self, traj: Trajectory, out: list[Violation]) -> None:
        p_err = float(np.linalg.norm(traj.position(0.0) - self.task.start_pos))
        if p_err > self.cons.start_tol:
            out.append(
                Violation(
                    "start_position",
                    f"|p(0) - start| = {p_err:.3f} m",
                    t=0.0,
                    value=p_err,
                    limit=self.cons.start_tol,
                )
            )
        v0 = float(np.linalg.norm(traj.velocity(0.0)))
        if v0 > self.cons.start_vel_tol:
            out.append(
                Violation(
                    "start_velocity",
                    f"|v(0)| = {v0:.3f} m/s, must start from hover",
                    t=0.0,
                    value=v0,
                    limit=self.cons.start_vel_tol,
                )
            )

    def _check_bounds(self, pos: np.ndarray, out: list[Violation]) -> None:
        arena = self.task.arena
        inside = arena.contains(pos)
        if not inside.all():
            i = int(np.argmax(~inside))
            out.append(
                Violation(
                    "arena_bounds",
                    f"trajectory leaves the arena at sample {i} ({pos[i].round(3).tolist()})",
                    value=float("nan"),
                    limit=float("nan"),
                )
            )
        z_min_seen = float(pos[:, 2].min())
        if z_min_seen < self.cons.z_min:
            i = int(np.argmin(pos[:, 2]))
            out.append(
                Violation(
                    "floor_clearance",
                    f"z = {z_min_seen:.3f} m at sample {i}, below z_min",
                    value=z_min_seen,
                    limit=self.cons.z_min,
                )
            )

    def _check_limits(
        self,
        times: np.ndarray,
        vel: np.ndarray,
        acc: np.ndarray,
        jerk: np.ndarray,
        out: list[Violation],
        stats: dict,
    ) -> None:
        speed = np.linalg.norm(vel, axis=1)
        u = acc - G_VEC  # specific force: a - g
        u_norm = np.linalg.norm(u, axis=1)
        tilt = np.degrees(np.arccos(np.clip(u[:, 2] / np.maximum(u_norm, 1e-9), -1, 1)))
        jerk_norm = np.linalg.norm(jerk, axis=1)

        stats["max_speed"] = float(speed.max())
        stats["max_acc_norm"] = float(u_norm.max())
        stats["max_tilt_deg"] = float(tilt.max())
        stats["max_jerk"] = float(jerk_norm.max())

        self._flag_max("speed_limit", "speed", speed, self.cons.v_max, times, out, "m/s")
        self._flag_max("thrust_limit", "|u| (specific force)", u_norm, self.cons.a_max, times, out, "m/s^2")
        self._flag_max("tilt_limit", "thrust tilt angle", tilt, self.cons.tilt_max_deg, times, out, "deg")
        self._flag_max("jerk_limit", "jerk norm", jerk_norm, self.cons.j_max, times, out, "m/s^3")

    def _flag_max(
        self,
        check: str,
        what: str,
        series: np.ndarray,
        limit: float,
        times: np.ndarray,
        out: list[Violation],
        unit: str,
    ) -> None:
        i = int(np.argmax(series))
        if series[i] > limit:
            out.append(
                Violation(
                    check,
                    f"{what} = {series[i]:.2f} {unit}",
                    t=float(times[i]),
                    value=float(series[i]),
                    limit=float(limit),
                )
            )

    def _check_gates(self, traj: Trajectory, out: list[Violation], stats: dict) -> list[float]:
        times = traj.times
        pos = traj.positions
        gate_times: list[float] = []
        start_idx = 0
        insets: list[float] = []
        for k, gate in enumerate(self.task.gates):
            crossings = find_gate_crossings(gate, times, pos, start_idx=start_idx)
            valid = [c for c in crossings if c.is_valid(gate, self.cons.gate_margin)]
            if not valid:
                best = max((c.inset(gate) for c in crossings), default=None)
                detail = f"best inset {best:.3f} m" if best is not None else "never crossed the gate plane in the normal direction"
                out.append(
                    Violation(
                        "gate_missed",
                        f"gate {k + 1} not passed in the normal direction with margin "
                        f"{self.cons.gate_margin} m ({detail})",
                        t=float(gate_times[-1]) if gate_times else 0.0,
                        value=float(best) if best is not None else None,
                        limit=self.cons.gate_margin,
                    )
                )
                insets.append(best if best is not None else float("-inf"))
                continue
            c = valid[0]
            gate_times.append(c.t)
            insets.append(c.inset(gate))
            start_idx = int(np.searchsorted(times, c.t))  # next gate must be crossed later
        stats["gate_times"] = [round(t, 3) for t in gate_times]
        stats["gate_insets_m"] = [round(v, 3) if v == v else None for v in insets]
        if gate_times:
            stats["planned_lap_time"] = round(gate_times[-1], 3)
        return gate_times

    def _check_clearance(self, times: np.ndarray, pos: np.ndarray, out: list[Violation], stats: dict) -> None:
        min_clear = np.inf
        worst = ("none", 0.0)
        for k, gate in enumerate(self.task.gates):
            d = distance_to_gate(gate, pos)
            i = int(np.argmin(d))
            if d[i] < min_clear:
                min_clear = float(d[i])
                worst = (f"gate {k + 1}", float(times[i]))
        stats["min_obstacle_clearance_m"] = round(min_clear, 4)
        if min_clear < self.cons.clearance:
            out.append(
                Violation(
                    "obstacle_clearance",
                    f"clearance {min_clear:.3f} m to {worst[0]} at t={worst[1]:.2f} s",
                    t=worst[1],
                    value=min_clear,
                    limit=self.cons.clearance,
                )
            )

    def _check_tail(self, traj: Trajectory, last_gate_time: float, out: list[Violation], stats: dict) -> None:
        stats["tail_after_last_gate"] = round(traj.duration - last_gate_time, 3)
        if traj.duration < last_gate_time + self.timing.tail_after_last_gate - 1e-6:
            out.append(
                Violation(
                    "tail_too_short",
                    f"trajectory ends {traj.duration - last_gate_time:.3f} s after the last gate, "
                    f"need >= {self.timing.tail_after_last_gate} s",
                    t=traj.duration,
                    value=traj.duration - last_gate_time,
                    limit=self.timing.tail_after_last_gate,
                )
            )
