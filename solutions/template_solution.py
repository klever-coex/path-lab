import numpy as np

from clover2gym import Trajectory
from clover2gym.tasks.spec import MotionPlanningTask

SPEED = 3.0
SHOULDER = 1.5


def solve(task: MotionPlanningTask) -> Trajectory:
    wps = [np.asarray(task.start_pos, dtype=float)]
    for g in task.gates:
        wps.append(g.center - g.normal * SHOULDER)
        wps.append(g.center + g.normal * SHOULDER)
    last = task.gates[-1]
    wps.append(last.center + last.normal * 3.0)
    return Trajectory.from_waypoints(np.array(wps), dt=task.timing.dt_max, cruise_speed=SPEED)
