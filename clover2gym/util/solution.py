import importlib.util
import sys
from pathlib import Path

from ..tasks.spec import MotionPlanningTask
from ..trajectory import Trajectory, load_trajectory


class Solution:
    """A participant's solution module exposing ``solve(task)``."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        spec = importlib.util.spec_from_file_location(
            self.path.stem,
            self.path,
            submodule_search_locations=[str(self.path.parent)],
        )
        self.module = importlib.util.module_from_spec(spec)
        sys.modules[self.path.stem] = self.module
        spec.loader.exec_module(self.module)
        if not hasattr(self.module, "solve"):
            raise SystemExit(f"{self.path}: no solve(task) function found")

    def solve(self, task: MotionPlanningTask) -> Trajectory:
        obj = self.module.solve(task)
        try:
            return load_trajectory(obj)
        except TypeError as e:
            raise SystemExit(f"solve() returned an unusable object: {e}")
