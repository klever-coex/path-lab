"""PyBullet world: ground, procedural gates, arena, one drone.

The same gate geometry module (util.geometry) drives both the static checks
and the collision bodies built here, so "passing a gate" means exactly the
same thing in both validation stages.
"""
from __future__ import annotations

import time

import pybullet as p
from importlib.resources import files

import numpy as np

from ..model.drone import DroneModel
from ..tasks.spec import MotionPlanningTask
from ..util.geometry import gate_collision_boxes

PYB_FREQ = 240
PYB_DT = 1.0 / PYB_FREQ


class RaceWorld:
    """A simulated race course with a single drone."""

    def __init__(self, task: MotionPlanningTask, drone: DroneModel, gui: bool = False, spawn_drone: bool = True):
        self.task = task
        self.drone = drone
        self.gui = gui
        self.spawn_drone = spawn_drone

        if gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.setTimeStep(PYB_DT, physicsClientId=self.client)
        p.setRealTimeSimulation(0, physicsClientId=self.client)

        if gui:
            # Avoid redrawing after every one of the many gate bodies is
            # created. This otherwise makes GUI startup look hung on macOS.
            p.configureDebugVisualizer(
                p.COV_ENABLE_RENDERING, 0, physicsClientId=self.client
            )
            # The ExampleBrowser panels and synthetic-camera previews make
            # debug-text updates very expensive on macOS. Keep the 3D scene
            # and our own HUD, without the unused built-in panels.
            p.configureDebugVisualizer(
                p.COV_ENABLE_GUI, 0, physicsClientId=self.client
            )
        try:
            import pybullet_data

            p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
            self.ground_id = p.loadURDF("plane.urdf", physicsClientId=self.client)

            self.gate_body_ids = self._build_gates()
            if gui:
                self._build_arena_walls()
                p.resetDebugVisualizerCamera(
                    cameraDistance=18.0,
                    cameraYaw=45.0,
                    cameraPitch=-35.0,
                    cameraTargetPosition=[0.0, 0.0, 1.5],
                    physicsClientId=self.client,
                )

            self.drone_id = self._spawn_drone() if spawn_drone else None
        finally:
            if gui:
                p.configureDebugVisualizer(
                    p.COV_ENABLE_RENDERING, 1, physicsClientId=self.client
                )

    # ------------------------------------------------------------------ #
    # world construction
    # ------------------------------------------------------------------ #
    def _build_gates(self) -> list[int]:
        from scipy.spatial.transform import Rotation
        
        ids: list[int] = []
        for gate in self.task.gates:
            for center, half_extents, R in gate_collision_boxes(gate):
                shape = p.createCollisionShape(
                    p.GEOM_BOX, halfExtents=half_extents.tolist(), physicsClientId=self.client
                )
                visual = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=half_extents.tolist(),
                    rgbaColor=[0.9, 0.4, 0.1, 1],
                    physicsClientId=self.client,
                )
                quat = Rotation.from_matrix(R).as_quat().tolist()
                ids.append(
                    p.createMultiBody(
                        baseMass=0.0,
                        baseCollisionShapeIndex=shape,
                        baseVisualShapeIndex=visual,
                        basePosition=center.tolist(),
                        baseOrientation=quat,
                        physicsClientId=self.client,
                    )
                )
        return ids

    def _build_arena_walls(self) -> None:
        arena = self.task.arena
        center = (arena.min + arena.max) / 2
        size = arena.max - arena.min
        for axis in range(3):
            for side in (-1, 1):
                half = size / 2
                half[axis] = 0.05
                pos = center.copy()
                pos[axis] += side * (size[axis] / 2 + 0.05)
                visual = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=np.abs(half).tolist(),
                    rgbaColor=[0.2, 0.6, 0.9, 0.15],
                    physicsClientId=self.client,
                )
                p.createMultiBody(
                    baseMass=0.0,
                    baseVisualShapeIndex=visual,
                    basePosition=pos.tolist(),
                    physicsClientId=self.client,
                )

    def _spawn_drone(self) -> int:
        track = self.task.track
        quat = p.getQuaternionFromEuler([0.0, 0.0, float(track.start_yaw)])
        urdf = files("clover2gym") / "model" / "assets" / f"{self.task.drone}.urdf"
        return p.loadURDF(
            str(urdf),
            basePosition=track.start_pos.tolist(),
            baseOrientation=quat,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=self.client,
        )

    # ------------------------------------------------------------------ #
    # simulation interface
    # ------------------------------------------------------------------ #
    def apply_rpms(self, rpms) -> None:
        """Apply propeller forces/torques for the given (possibly lagged) RPMs."""
        d = self.drone
        forces = d.rpm_to_force(rpms)
        for i in range(4):
            p.applyExternalForce(
                self.drone_id,
                i,
                forceObj=[0.0, 0.0, float(forces[i])],
                posObj=[0.0, 0.0, 0.0],
                flags=p.LINK_FRAME,
                physicsClientId=self.client,
            )
            tau_z = float(d.prop_dirs[i] * d.km * rpms[i] ** 2)
            p.applyExternalTorque(
                self.drone_id,
                i,
                torqueObj=[0.0, 0.0, tau_z],
                flags=p.LINK_FRAME,
                physicsClientId=self.client,
            )

    def step(self) -> None:
        p.stepSimulation(physicsClientId=self.client)

    def drone_state(self) -> dict:
        pos, quat = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.client)
        vel, ang_vel = p.getBaseVelocity(self.drone_id, physicsClientId=self.client)
        return {
            "pos": np.asarray(pos, dtype=float),
            "quat": np.asarray(quat, dtype=float),  # pybullet order: [x, y, z, w]
            "vel": np.asarray(vel, dtype=float),
            "ang_vel": np.asarray(ang_vel, dtype=float),  # world frame
        }

    def contacts(self) -> list[tuple[int, int]]:
        """Contacts of the drone with anything (bodyB, linkIndexB) per contact point."""
        return [(c[2], c[4]) for c in p.getContactPoints(bodyA=self.drone_id, physicsClientId=self.client)]

    def close(self) -> None:
        try:
            p.disconnect(physicsClientId=self.client)
        except p.error:
            pass

    # ------------------------------------------------------------------ #
    # interactive viewer
    # ------------------------------------------------------------------ #
    def draw_gate_numbers(self) -> None:
        """Label gates with their crossing order (GUI only)."""
        if not self.gui:
            return
        for k, gate in enumerate(self.task.gates):
            p.addUserDebugText(
                str(k + 1),
                textPosition=(gate.center + gate.normal * 0.8).tolist(),
                textColorRGB=[0.8, 0.2, 0.1],
                textSize=1.8,
                physicsClientId=self.client,
            )

    def _move_camera_from_keys(self, keys: dict[int, int]) -> None:
        """Apply portable keyboard camera controls to the debug visualizer."""
        is_down = lambda key: bool(keys.get(key, 0) & p.KEY_IS_DOWN)
        camera_keys = (
            p.B3G_LEFT_ARROW,
            p.B3G_RIGHT_ARROW,
            p.B3G_UP_ARROW,
            p.B3G_DOWN_ARROW,
            p.B3G_PAGE_UP,
            p.B3G_PAGE_DOWN,
            ord("w"),
            ord("W"),
            ord("a"),
            ord("A"),
            ord("s"),
            ord("S"),
            ord("d"),
            ord("D"),
        )
        if not any(is_down(key) for key in camera_keys):
            return

        camera = p.getDebugVisualizerCamera(physicsClientId=self.client)
        yaw, pitch, distance = map(float, camera[8:11])
        target = np.asarray(camera[11], dtype=float)

        yaw += 1.5 * (is_down(p.B3G_LEFT_ARROW) - is_down(p.B3G_RIGHT_ARROW))
        pitch = float(
            np.clip(
                pitch + 1.5 * (is_down(p.B3G_UP_ARROW) - is_down(p.B3G_DOWN_ARROW)),
                -89.0,
                89.0,
            )
        )
        if is_down(p.B3G_PAGE_UP):
            distance = max(0.1, distance * 0.96)
        if is_down(p.B3G_PAGE_DOWN):
            distance = min(1000.0, distance / 0.96)

        pan_forward = np.asarray(camera[5], dtype=float)
        pan_forward[2] = 0.0
        forward_norm = float(np.linalg.norm(pan_forward))
        if forward_norm > 1e-9:
            pan_forward /= forward_norm
        pan_right = np.asarray(camera[6], dtype=float)
        pan_right[2] = 0.0
        right_norm = float(np.linalg.norm(pan_right))
        if right_norm > 1e-9:
            pan_right /= right_norm
        pan_step = max(0.03, distance * 0.015)
        target += pan_forward * pan_step * (
            (is_down(ord("w")) or is_down(ord("W")))
            - (is_down(ord("s")) or is_down(ord("S")))
        )
        target += pan_right * pan_step * (
            (is_down(ord("d")) or is_down(ord("D")))
            - (is_down(ord("a")) or is_down(ord("A")))
        )

        p.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=target.tolist(),
            physicsClientId=self.client,
        )

    def _move_camera_from_mouse(
        self, events: tuple[tuple, ...], *, shift_down: bool = False
    ) -> None:
        """Implement mouse camera controls without relying on PyBullet's GUI bindings."""
        if not hasattr(self, "_mouse_buttons"):
            self._mouse_buttons: set[int] = set()
            self._mouse_pos: tuple[float, float] | None = None

        for event in events:
            if len(event) < 5:
                continue
            event_type, x, y, button, state = event[:5]
            x, y = float(x), float(y)

            if button >= 0:
                if state & (p.KEY_WAS_TRIGGERED | p.KEY_IS_DOWN):
                    self._mouse_buttons.add(button)
                if state & p.KEY_WAS_RELEASED:
                    self._mouse_buttons.discard(button)

            previous = self._mouse_pos
            self._mouse_pos = (x, y)
            if event_type != 1 or previous is None or not self._mouse_buttons:
                continue

            dx, dy = x - previous[0], y - previous[1]
            if dx == 0.0 and dy == 0.0:
                continue

            camera = p.getDebugVisualizerCamera(physicsClientId=self.client)
            yaw, pitch, distance = map(float, camera[8:11])
            target = np.asarray(camera[11], dtype=float)

            if 0 in self._mouse_buttons and not shift_down:  # left drag: orbit
                yaw -= dx * 0.35
                pitch = float(np.clip(pitch + dy * 0.35, -89.0, 89.0))
            if 1 in self._mouse_buttons or (
                0 in self._mouse_buttons and shift_down
            ):  # middle drag or Shift + left drag: pan
                right = np.asarray(camera[6], dtype=float)
                right /= max(float(np.linalg.norm(right)), 1e-9)
                up = np.asarray(camera[4], dtype=float)
                up /= max(float(np.linalg.norm(up)), 1e-9)
                target += (-right * dx + up * dy) * distance * 0.002
            if 2 in self._mouse_buttons:  # right drag: zoom
                distance = float(np.clip(distance * np.exp(dy * 0.01), 0.1, 1000.0))

            p.resetDebugVisualizerCamera(
                cameraDistance=distance,
                cameraYaw=yaw,
                cameraPitch=pitch,
                cameraTargetPosition=target.tolist(),
                physicsClientId=self.client,
            )

    def view(self, message: str = "course loaded") -> None:
        """Interactive course viewer.

        PyBullet's macOS GUI processes window events on the thread making API
        calls. Keep polling it instead of blocking on terminal input so camera
        controls and window close events remain responsive.
        """
        if not self.gui:
            raise RuntimeError("view() requires gui=True")
        self.draw_gate_numbers()
        self._mouse_buttons = set()
        self._mouse_pos = None
        p.configureDebugVisualizer(
            p.COV_ENABLE_MOUSE_PICKING, 0, physicsClientId=self.client
        )
        print(
            f"{message}\n"
            "mouse: left-drag orbit, middle-drag or Shift+left-drag pan, "
            "right-drag or wheel zoom\n"
            "keyboard: arrows orbit, WASD pan, Page Up/Down zoom; q/Esc quits",
            flush=True,
        )
        p.setRealTimeSimulation(1, physicsClientId=self.client)
        try:
            while p.isConnected(physicsClientId=self.client):
                keys = p.getKeyboardEvents(physicsClientId=self.client)
                if any(
                    keys.get(key, 0) & p.KEY_WAS_TRIGGERED
                    for key in (ord("q"), ord("Q"), 27)
                ):
                    break
                self._move_camera_from_keys(keys)
                self._move_camera_from_mouse(
                    p.getMouseEvents(physicsClientId=self.client),
                    shift_down=bool(keys.get(p.B3G_SHIFT, 0) & p.KEY_IS_DOWN),
                )
                time.sleep(1.0 / 60.0)
        finally:
            if p.isConnected(physicsClientId=self.client):
                p.setRealTimeSimulation(0, physicsClientId=self.client)
            self.close()
