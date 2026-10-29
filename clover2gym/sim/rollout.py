"""Closed-loop simulation of a trajectory with the built-in controller.

The drone tracks the submitted spline through the fixed GeometricController;
gates, collisions and arena bounds are checked on the *actual* simulated
states. The lap time (crossing of the last gate) is the competition score.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from ..model.drone import DroneModel
from ..tasks.spec import MotionPlanningTask
from ..trajectory import Trajectory
from ..util.geometry import distance_to_gate, find_gate_crossings
from ..validation.report import CheckReport, Violation
from .controller import TrackingController, load_controller
from .world import PYB_DT, RaceWorld

_GUI_INTERVAL = 1.0 / 60.0

_log = logging.getLogger(__name__)


@dataclass
class RolloutResult:
    report: CheckReport
    lap_time: float | None
    gate_times: list[float]
    log: dict = field(default_factory=dict)


def load_drone(name: str) -> DroneModel:
    if name != "racer":
        raise ValueError(f"unknown drone {name!r}")
    return DroneModel.racer()


def rollout(
    task: MotionPlanningTask,
    traj: Trajectory,
    gui: bool = False,
    extra_time: float = 3.0,
    controller: TrackingController | None = None,
    follow_camera: bool = True,
) -> RolloutResult:
    """Fly the trajectory in simulation and validate the actual flight.

    In GUI mode a chase camera follows the drone from behind (press ``f`` to
    toggle between chase and free camera).
    """
    drone = load_drone(task.drone)
    world = RaceWorld(task, drone, gui=gui)
    violations: list[Violation] = []
    stats: dict = {}
    try:
        return _run(task, traj, world, drone, gui, extra_time, controller, violations, stats, follow_camera)
    finally:
        if gui:
            import pybullet as p

            if p.isConnected(physicsClientId=world.client):
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=world.client)
        world.close()


def _run(task, traj, world, drone, gui, extra_time, controller, violations, stats, follow_camera=True) -> RolloutResult:
    cons = task.constraints
    _log.debug(
        "rollout: track=%s gates=%d traj_duration=%.2fs gui=%s",
        task.track.name, len(task.gates), traj.duration, gui,
    )
    ctrl = controller or load_controller(task.controller, drone)
    yaw_times, yaw_values = traj.yaw_profile(task.track.start_yaw)

    t_end = min(traj.duration + extra_time, task.timing.t_max + extra_time)
    n_steps = int(np.ceil(t_end / PYB_DT))

    rpm = np.full(4, drone.hover_rpm)
    log = {
        "t": np.zeros(n_steps),
        "pos": np.zeros((n_steps, 3)),
        "vel": np.zeros((n_steps, 3)),
        "quat": np.zeros((n_steps, 4)),
        "rpm": np.zeros((n_steps, 4)),
    }

    if gui:
        import pybullet as p

        # On macOS each debug command can redraw the whole window. Batch
        # initial annotations before enabling the complete scene again.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=world.client)
        _draw_path(world, traj)
        world.draw_gate_numbers()

    gate_times: list[float] = []
    next_gate = 0
    n_gates = len(task.gates)
    steps_done = 0
    lap_time = None
    tracking_err = np.zeros(n_steps)
    hud_id = -1
    next_hud_time = 0.0
    follow = bool(gui and follow_camera)
    follow_key_prev = False
    cam_yaw = float(task.track.start_yaw)
    if gui:
        import pybullet as p

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=world.client)

    state = world.drone_state()
    wall_start = time.perf_counter()
    next_gui_time = wall_start
    last_camera_time = -PYB_DT
    for i in range(n_steps):
        t = i * PYB_DT
        target_pos = traj.position(t)
        rpm_cmd = ctrl.compute(
            dt=PYB_DT,
            pos=state["pos"],
            quat_xyzw=state["quat"],
            vel=state["vel"],
            ang_vel_world=state["ang_vel"],
            target_pos=target_pos,
            target_vel=traj.velocity(t),
            target_acc=traj.acceleration(t),
            target_yaw=float(np.interp(t, yaw_times, yaw_values)),
        )

        # first-order motor dynamics: forces come from the lagged RPMs
        if drone.motor_tau > 0:
            rpm = rpm + (rpm_cmd - rpm) * PYB_DT / drone.motor_tau
        else:
            rpm = rpm_cmd.copy()
        rpm = np.clip(rpm, 0.0, drone.max_rpm)

        world.apply_rpms(rpm)
        # A slow renderer must drop frames, not slow down the physics clock.
        now = time.perf_counter() if gui else 0.0
        gui_frame = gui and now >= next_gui_time and now - wall_start <= (i + 1) * PYB_DT
        world.step()

        state = world.drone_state()
        log["t"][i] = t
        log["pos"][i] = state["pos"]
        log["vel"][i] = state["vel"]
        log["quat"][i] = state["quat"]
        log["rpm"][i] = rpm
        tracking_err[i] = float(np.linalg.norm(state["pos"] - target_pos))
        steps_done = i + 1

        if world.contacts():
            violations.append(
                Violation(
                    "collision",
                    f"drone collision at t={t:.3f} s",
                    t=t,
                )
            )
            break

        if not task.arena.contains(state["pos"].reshape(1, 3))[0]:
            violations.append(
                Violation(
                    "arena_bounds",
                    f"drone left the arena at t={t:.3f} s ({state['pos'].round(3).tolist()})",
                    t=t,
                )
            )
            break

        # gate progress on the actual trajectory
        if next_gate < n_gates:
            gate = task.gates[next_gate]
            crossings = find_gate_crossings(
                gate,
                log["t"][max(0, i - 1) : i + 1],
                log["pos"][max(0, i - 1) : i + 1],
            )
            for c in crossings:
                if c.is_valid(gate, cons.gate_margin):
                    gate_times.append(c.t)
                    next_gate += 1
                    _log.info("gate %d/%d passed at t=%.3f s", next_gate, n_gates, c.t)
                    break

        if next_gate == n_gates:
            lap_time = gate_times[-1]
            break

        if gui_frame:
            # 'f' toggles chase/free camera
            keys = p.getKeyboardEvents(physicsClientId=world.client)
            f_state = keys.get(ord("f"), 0)
            f_down = bool(f_state & p.KEY_IS_DOWN)
            if f_state & p.KEY_WAS_TRIGGERED or (f_down and not follow_key_prev):
                follow = not follow
            follow_key_prev = f_down

            if follow:
                # camera sits behind the velocity heading: cam yaw = heading - 90 deg
                v_xy = state["vel"][:2]
                if float(np.linalg.norm(v_xy)) > 0.5:
                    heading = float(np.arctan2(v_xy[1], v_xy[0]))
                    delta = (heading - cam_yaw + np.pi) % (2 * np.pi) - np.pi
                    cam_yaw += delta * min((t - last_camera_time) / 0.25, 1.0)
                p.resetDebugVisualizerCamera(
                    cameraDistance=5.0,
                    cameraYaw=float(np.degrees(cam_yaw)) - 90.0,
                    cameraPitch=-20.0,
                    cameraTargetPosition=state["pos"].tolist(),
                    physicsClientId=world.client,
                )

            if now >= next_hud_time:
                speed = float(np.linalg.norm(state["vel"]))
                txt = (
                    f"t={t:5.2f}s  gate {next_gate + 1}/{n_gates}  "
                    f"v={speed:4.1f} m/s  err={tracking_err[i]:4.2f} m"
                    + ("" if follow else "  [f: chase cam]")
                )
                hud_id = p.addUserDebugText(
                    txt,
                    textPosition=(state["pos"] + np.array([0.0, 0.0, 0.6])).tolist(),
                    textColorRGB=[0, 0, 0],
                    textSize=1.4,
                    lifeTime=0.0,
                    physicsClientId=world.client,
                    replaceItemUniqueId=hud_id,
                )
                next_hud_time = now + 0.1
            last_camera_time = t
            next_gui_time = now + _GUI_INTERVAL

        if gui:
            remaining = wall_start + (i + 1) * PYB_DT - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)

    # ------------------------------------------------------------------ #
    # report
    # ------------------------------------------------------------------ #
    log = {k: v[:steps_done] for k, v in log.items()}

    if lap_time is None and not violations:
        crossed = len(gate_times)
        violations.append(
            Violation(
                "sim_timeout",
                f"simulation ended after {t_end:.2f} s with {crossed}/{n_gates} gates passed",
                t=float(log["t"][-1]) if steps_done else 0.0,
                value=crossed,
                limit=n_gates,
            )
        )

    stats["lap_time"] = round(lap_time, 4) if lap_time is not None else None
    stats["gate_times"] = [round(t, 4) for t in gate_times]
    stats["steps"] = steps_done
    stats["sim_time"] = round(float(log["t"][-1]), 3) if steps_done else 0.0
    if steps_done > 1:
        pos = log["pos"]
        stats["max_speed_actual"] = round(
            float(np.linalg.norm(np.diff(pos, axis=0), axis=1).max() / PYB_DT), 3
        )
        stats["tracking_err_mean_m"] = round(float(tracking_err[:steps_done].mean()), 4)
        stats["tracking_err_max_m"] = round(float(tracking_err[:steps_done].max()), 4)
        min_clear = min(
            float(distance_to_gate(g, pos).min()) for g in task.gates
        )
        stats["min_clearance_actual_m"] = round(min_clear, 4)

    report = CheckReport(stage="sim", ok=not violations, violations=violations, stats=stats)
    if violations:
        _log.warning(
            "rollout failed on %s: %s", task.track.name, "; ".join(v.message for v in violations)
        )
    else:
        _log.info("rollout finished on %s: lap_time=%.3f s", task.track.name, lap_time)
    return RolloutResult(report=report, lap_time=lap_time, gate_times=gate_times, log=log)


def _draw_path(world: RaceWorld, traj: Trajectory) -> None:
    import pybullet as p

    pts = traj.position(np.arange(0, traj.duration, 0.2))
    for a, b in zip(pts[:-1], pts[1:]):
        p.addUserDebugLine(
            a.tolist(),
            b.tolist(),
            lineColorRGB=[0, 0.7, 0.3],
            lineWidth=2,
            physicsClientId=world.client,
        )
