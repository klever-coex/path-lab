"""Tracking controllers: the interface, the registry, and implementations.

The controller is part of the task definition (like the drone model): every
participant's trajectory is flown through the *same* controller, referenced
by name from the track yaml. Implementations register themselves via
``@register_controller`` and must follow the ``TrackingController`` protocol.
"""
from __future__ import annotations

import logging
from typing import ClassVar, Protocol

import numpy as np
from scipy.spatial.transform import Rotation

from ..model.drone import DroneModel

_log = logging.getLogger(__name__)

G_VEC = np.array([0.0, 0.0, -9.8])


class TrackingController(Protocol):
    """A trajectory tracking controller: state + plan reference -> motor RPMs."""

    name: ClassVar[str]

    def reset(self) -> None: ...

    def compute(
        self,
        dt: float,
        pos: np.ndarray,
        quat_xyzw: np.ndarray,
        vel: np.ndarray,
        ang_vel_world: np.ndarray,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
        target_acc: np.ndarray,
        target_yaw: float,
    ) -> np.ndarray:
        """Return per-motor RPM commands for one control step (240 Hz)."""
        ...


CONTROLLERS: dict[str, type] = {}


def register_controller(cls):
    """Class decorator: add a controller implementation to the registry."""
    CONTROLLERS[cls.name] = cls
    return cls


def load_controller(name: str, drone: DroneModel) -> TrackingController:
    if name not in CONTROLLERS:
        raise ValueError(f"unknown controller {name!r}; known: {sorted(CONTROLLERS)}")
    return CONTROLLERS[name](drone)


def _vee(S: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix -> vector."""
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def _desired_attitude(thrust_vec: np.ndarray, yaw: float) -> tuple[np.ndarray, np.ndarray]:
    """Desired body-z direction and rotation matrix from a thrust vector + yaw."""
    z_b_des = thrust_vec / max(float(np.linalg.norm(thrust_vec)), 1e-9)
    x_c = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    y_b_des = np.cross(z_b_des, x_c)
    y_norm = np.linalg.norm(y_b_des)
    y_b_des = y_b_des / max(y_norm, 1e-9)
    x_b_des = np.cross(y_b_des, z_b_des)
    return z_b_des, np.column_stack((x_b_des, y_b_des, z_b_des))


# ---------------------------------------------------------------------- #
# implementations
# ---------------------------------------------------------------------- #
@register_controller
class GeometricController:
    """SE(3) cascaded controller, no rate feedforward.

    Position loop: PD + acceleration feedforward -> thrust vector.
    Attitude loop: geometric error on SO(3) -> body torques -> RPMs.
    """

    name: ClassVar[str] = "geometric"

    def __init__(
        self,
        drone: DroneModel,
        kp_pos: tuple = (16.0, 16.0, 22.0),
        kd_pos: tuple = (8.0, 8.0, 9.4),
        k_att: tuple = (1.6, 1.6, 0.5),
        k_rate: tuple = (0.22, 0.22, 0.05),
    ):
        self.drone = drone
        self.kp_pos = np.array(kp_pos)
        self.kd_pos = np.array(kd_pos)
        self.k_att = np.array(k_att)
        self.k_rate = np.array(k_rate)
        self.reset()

    def reset(self) -> None:
        self._prev_rpm_cmd = None

    def compute(
        self,
        dt: float,
        pos: np.ndarray,
        quat_xyzw: np.ndarray,
        vel: np.ndarray,
        ang_vel_world: np.ndarray,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
        target_acc: np.ndarray,
        target_yaw: float,
    ) -> np.ndarray:
        R = Rotation.from_quat(quat_xyzw).as_matrix()  # body -> world
        omega = R.T @ ang_vel_world  # body rates

        # --- position loop -------------------------------------------- #
        a_des = (
            target_acc
            + self.kp_pos * (target_pos - pos)
            + self.kd_pos * (target_vel - vel)
        )
        thrust_vec = self.drone.mass * (a_des - G_VEC)
        _, R_des = _desired_attitude(thrust_vec, target_yaw)

        # project the desired thrust on the *current* body z (Lee convention)
        collective = float(np.dot(thrust_vec, R[:, 2]))

        # --- attitude loop --------------------------------------------- #
        e_R = 0.5 * _vee(R_des.T @ R - R.T @ R_des)
        e_omega = omega  # no rate feedforward: the loop always lags slews
        torque = -self.k_att * e_R - self.k_rate * e_omega
        return self._allocate(collective, torque)

    def _allocate(self, collective: float, torque: np.ndarray) -> np.ndarray:
        forces = self.drone.allocate(collective, torque)
        forces = np.clip(forces, 0.0, self.drone.max_thrust / 4.0)
        return self.drone.force_to_rpm(forces)


@register_controller
class GeometricFeedforwardController(GeometricController):
    """Geometric controller with plan-derived angular-rate feedforward.

    Same cascade as ``geometric``, plus a desired body-rate term built from
    the *plan itself*: the rotation rate of the feedforward thrust direction
    (from ``a_ff``) and the yaw profile rate. Differentiating the state
    feedback instead (as one might naively do through R_des) amplifies noise;
    the plan signals are smooth, so a light filter is enough. This removes
    the steady-state attitude lag on fast tilt slews (hairpins, chicanes).
    """

    name: ClassVar[str] = "geometric_ff"

    def __init__(self, drone: DroneModel, ff_filter_tau: float = 0.06, ff_yaw: bool = True, **kwargs):
        super().__init__(drone, **kwargs)
        # bandwidth-matched feedforward: only rotations slower than the
        # attitude loop's tracking bandwidth are fed forward; faster plan
        # transients (spline artifacts) are left to the PD to shape
        self.ff_filter_tau = ff_filter_tau
        self.ff_yaw = ff_yaw

    def reset(self) -> None:
        super().reset()
        self._prev_yaw = None
        self._prev_z_ff = None
        self._omega_des_world_filt = np.zeros(3)

    def compute(
        self,
        dt: float,
        pos: np.ndarray,
        quat_xyzw: np.ndarray,
        vel: np.ndarray,
        ang_vel_world: np.ndarray,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
        target_acc: np.ndarray,
        target_yaw: float,
    ) -> np.ndarray:
        R = Rotation.from_quat(quat_xyzw).as_matrix()  # body -> world
        omega = R.T @ ang_vel_world  # body rates

        # --- position loop -------------------------------------------- #
        a_des = (
            target_acc
            + self.kp_pos * (target_pos - pos)
            + self.kd_pos * (target_vel - vel)
        )
        thrust_vec = self.drone.mass * (a_des - G_VEC)

        # feedforward thrust direction from the PLAN acceleration only
        thrust_ff = self.drone.mass * (target_acc - G_VEC)
        z_ff = thrust_ff / max(float(np.linalg.norm(thrust_ff)), 1e-9)

        # --- plan-derived angular rate feedforward --------------------- #
        omega_des_world = np.zeros(3)
        if self._prev_z_ff is not None and dt > 0:
            # rotation carrying z_ff to the new value: omega = z x z_dot
            z_dot = (z_ff - self._prev_z_ff) / dt
            omega_des_world = np.cross(self._prev_z_ff, z_dot)
            if self._prev_yaw is not None and self.ff_yaw:
                d_yaw = (target_yaw - self._prev_yaw + np.pi) % (2 * np.pi) - np.pi
                omega_des_world = omega_des_world + (d_yaw / dt) * z_ff
            alpha = min(dt / self.ff_filter_tau, 1.0) if self.ff_filter_tau > 0 else 1.0
            self._omega_des_world_filt += alpha * (omega_des_world - self._omega_des_world_filt)
            omega_des_world = self._omega_des_world_filt
        self._prev_z_ff = z_ff
        self._prev_yaw = target_yaw

        # --- attitude target and loops --------------------------------- #
        z_b_des, R_des = _desired_attitude(thrust_vec, target_yaw)
        collective = float(np.dot(thrust_vec, R[:, 2]))
        e_R = 0.5 * _vee(R_des.T @ R - R.T @ R_des)
        e_omega = omega - R.T @ R_des @ omega_des_world
        torque = -self.k_att * e_R - self.k_rate * e_omega
        return self._allocate(collective, torque)
