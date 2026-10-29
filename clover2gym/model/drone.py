"""Drone model: parameters parsed from a URDF file.

Follows the ``<properties>`` tag convention of gym-pybullet-drones: all
physics parameters live in one custom XML attribute block so that a drone is
fully described by its URDF and this parser.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Self

import numpy as np

G = 9.8  # m/s^2, matches the gravity set in the simulator


@dataclass
class DroneModel:
    """Physics model of a quadrotor.

    Attributes
    ----------
    mass : float               total mass in kg
    kf, km : float             thrust / reaction-torque coefficients (F = kf*w^2, M_z = km*w^2)
    thrust2weight : float      max total thrust / (m*g)
    max_speed : float          catalogue max speed in m/s
    motor_tau : float          first-order RPM lag time constant in s
    collision_radius : float   radius of the collision cylinder in m
    prop_positions : (4,3)     propeller positions in the body frame
    prop_dirs : (4,)           sign of each propeller's reaction torque along +z
    """

    urdf_path: Path
    name: str
    mass: float
    kf: float
    km: float
    thrust2weight: float
    max_speed: float
    motor_tau: float
    collision_radius: float
    prop_positions: np.ndarray
    prop_dirs: np.ndarray
    inertia: np.ndarray

    def __post_init__(self) -> None:
        self._build_allocation()

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_urdf(cls, path: str | Path) -> Self:
        path = Path(path)
        root = ET.parse(path).getroot()

        props = root.find("properties")
        if props is None or root.find("link[@name='base_link']") is None:
            raise ValueError(f"{path}: expected <properties> tag and base_link")

        base = root.find("link[@name='base_link']")
        assert base is not None
        inertial = base.find("inertial")
        assert inertial is not None
        mass = float(inertial.find("mass").attrib["value"])  # type: ignore[union-attr]
        inertia = np.array(
            [
                [float(inertial.find("inertia").attrib[k]) for k in ("ixx", "ixy", "ixz")],  # type: ignore[union-attr]
                [float(inertial.find("inertia").attrib[k]) for k in ("ixy", "iyy", "iyz")],  # type: ignore[union-attr]
                [float(inertial.find("inertia").attrib[k]) for k in ("ixz", "iyz", "izz")],  # type: ignore[union-attr]
            ]
        )

        prop_positions = []
        for link in root.findall("link"):
            if link.attrib["name"].startswith("prop"):
                origin = link.find("inertial/origin")
                assert origin is not None
                prop_positions.append([float(v) for v in origin.attrib["xyz"].split()])

        a = props.attrib
        return cls(
            urdf_path=path,
            name=root.attrib.get("name", path.stem),
            mass=mass,
            kf=float(a["kf"]),
            km=float(a["km"]),
            thrust2weight=float(a["thrust2weight"]),
            max_speed=float(a["max_speed_kmh"]) / 3.6,
            motor_tau=float(a.get("motor_tau", "0")),
            collision_radius=float(a.get("collision_radius", "0.06")),
            prop_positions=np.array(prop_positions, dtype=float),
            prop_dirs=np.array([float(v) for v in a.get("prop_dirs", "-1 1 -1 1").split()]),
            inertia=inertia,
        )

    @classmethod
    def racer(cls) -> Self:
        """The default racing drone shipped with the framework."""
        return cls.from_urdf(files("clover2gym") / "model" / "assets" / "racer.urdf")

    # ------------------------------------------------------------------ #
    # derived limits
    # ------------------------------------------------------------------ #
    @property
    def max_thrust(self) -> float:
        return self.thrust2weight * self.mass * G

    @property
    def max_rpm(self) -> float:
        return np.sqrt(self.max_thrust / (4 * self.kf))

    @property
    def hover_rpm(self) -> float:
        return np.sqrt(self.mass * G / (4 * self.kf))

    def rpm_to_force(self, rpm: np.ndarray | float) -> np.ndarray:
        return self.kf * np.asarray(rpm) ** 2

    def force_to_rpm(self, force: np.ndarray | float) -> np.ndarray:
        return np.sqrt(np.asarray(force, dtype=float) / self.kf)

    # ------------------------------------------------------------------ #
    # actuator allocation
    # ------------------------------------------------------------------ #
    def _build_allocation(self) -> None:
        r = self.prop_positions
        # Column i of B maps motor i thrust f_i to the wrench [Fz, tau_x, tau_y, tau_z]:
        #   r_i x (f_i * e_z) = f_i * (r_y, -r_x, 0); reaction torque = dir_i * km/kf * f_i * e_z
        B = np.column_stack(
            [
                np.ones(4),
                r[:, 1],
                -r[:, 0],
                self.prop_dirs * self.km / self.kf,
            ]
        ).T  # shape (4 wrench, 4 motors)
        self._alloc_B = B
        self._alloc_inv = np.linalg.inv(B)

    def allocate(self, thrust: float, torque: np.ndarray) -> np.ndarray:
        """Map a wrench (collective thrust, body torques) to per-motor thrusts."""
        wrench = np.concatenate(([thrust], np.asarray(torque, dtype=float)))
        return self._alloc_inv @ wrench

    def wrench_from_rpms(self, rpms: np.ndarray) -> tuple[float, np.ndarray]:
        """Map per-motor RPMs to (collective thrust, body torques)."""
        w = self._alloc_B @ self.rpm_to_force(rpms)
        return float(w[0]), w[1:]
