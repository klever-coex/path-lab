"""Full evaluation pipeline: static check, then simulation, then score."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .sim.rollout import RolloutResult, rollout
from .tasks.spec import MotionPlanningTask
from .trajectory import Trajectory
from .validation.report import CheckReport
from .validation.static import StaticChecker

_log = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    track_name: str
    static: CheckReport
    sim: CheckReport | None
    lap_time: float | None
    rollout: RolloutResult | None = None

    @property
    def passed(self) -> bool:
        if not self.static.ok:
            return False
        return self.sim is not None and self.sim.ok

    @property
    def stage_failed(self) -> str | None:
        if not self.static.ok:
            return "static"
        if self.sim is not None and not self.sim.ok:
            return "sim"
        return None

    def to_dict(self) -> dict:
        return {
            "track": self.track_name,
            "passed": self.passed,
            "stage_failed": self.stage_failed,
            "lap_time": self.lap_time,
            "static": self.static.to_dict(),
            "sim": self.sim.to_dict() if self.sim else None,
        }

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(__import__("json").dumps(self.to_dict(), indent=2, ensure_ascii=False))


def evaluate(task: MotionPlanningTask, traj: Trajectory, gui: bool = False, follow_camera: bool = True) -> EvaluationResult:
    """Run the two-stage check: static validation, then a simulated flight."""
    static = StaticChecker(task).check(traj)
    if not static.ok:
        _log.warning(
            "static check failed on %s: %s",
            task.track.name,
            "; ".join(v.message for v in static.violations),
        )
        return EvaluationResult(
            track_name=task.track.name, static=static, sim=None, lap_time=None
        )
    result = rollout(task, traj, gui=gui, follow_camera=follow_camera)
    return EvaluationResult(
        track_name=task.track.name,
        static=static,
        sim=result.report,
        lap_time=result.lap_time,
        rollout=result,
    )


def evaluate_set(tasks: list[MotionPlanningTask], traj_by_task: dict[str, Trajectory]) -> dict:
    """Evaluate a set of (hidden) tracks; a failed track scores t_max seconds."""
    results = []
    total = 0.0
    for task in tasks:
        res = evaluate(task, traj_by_task.get(task.track.name))
        penalty = task.timing.t_max if not res.passed else (res.lap_time or task.timing.t_max)
        total += penalty
        results.append({"result": res, "penalty_seconds": penalty})
    return {"total_time": total, "results": results}
