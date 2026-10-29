"""clover2gym: PyBullet-based framework for drone programming competition tasks.

Public API for task authors and participants. Importing this package does not
import pybullet; simulation modules are imported lazily by the runner.

Library logging follows the "no handler configuration" rule: modules log via
``logging.getLogger(__name__)`` and the application (the CLI, user scripts)
decides formatting and level; nothing is printed below WARNING by default.
"""
import logging

from .model.drone import DroneModel
from .tasks.spec import Bounds, Constraints, Gate, MotionPlanningTask, Timing, Track
from .trajectory import Trajectory, load_trajectory

__version__ = "0.1.0"

__all__ = [
    "Bounds",
    "Constraints",
    "DroneModel",
    "Gate",
    "MotionPlanningTask",
    "Timing",
    "Track",
    "Trajectory",
    "load_trajectory",
]

logging.getLogger(__name__).addHandler(logging.NullHandler())
