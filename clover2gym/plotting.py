"""Matplotlib plots of a track, the planned trajectory and the actual flight."""
from __future__ import annotations

import numpy as np

from .scoring import EvaluationResult
from .tasks.spec import MotionPlanningTask
from .trajectory import Trajectory
from .util.geometry import gate_frame


def plot_evaluation(
    task: MotionPlanningTask,
    traj: Trajectory,
    result: EvaluationResult | None = None,
    path: str = "evaluation.png",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 6))
    ax_top = fig.add_subplot(1, 3, (1, 2))
    ax_3d = fig.add_subplot(1, 3, 3, projection="3d")

    plan = traj.positions
    ax_top.plot(plan[:, 0], plan[:, 1], "-", color="tab:green", lw=1.2, label="planned")
    if result is not None and result.rollout is not None and len(result.rollout.log["t"]):
        actual = result.rollout.log["pos"]
        ax_top.plot(actual[:, 0], actual[:, 1], "-", color="tab:red", lw=1.0, alpha=0.8, label="actual")

    for k, gate in enumerate(task.gates):
        u, v, n = gate_frame(gate)
        R = np.column_stack((u, v, n))
        w2, h2 = gate.size / 2
        corners_2d = np.array([[-w2, -h2], [w2, -h2], [w2, h2], [-w2, h2], [-w2, -h2]])
        if gate.shape == "circle":
            th = np.linspace(0, 2 * np.pi, 24)
            corners_2d = np.column_stack((w2 * np.cos(th), w2 * np.sin(th)))
        corners_2d = np.column_stack((corners_2d, np.zeros(len(corners_2d))))
        corners = gate.center + (R @ corners_2d.T).T
        ax_top.plot(corners[:, 0], corners[:, 1], "-", color="tab:orange", lw=2)
        ax_top.annotate(str(k + 1), (gate.center[0], gate.center[1]), ha="center", fontsize=9)
        ax_3d.plot(corners[:, 0], corners[:, 1], corners[:, 2], "-", color="tab:orange", lw=1.5)

    ax_top.plot(*task.start_pos[:2], "k^", ms=10, label="start")
    ax_top.set_aspect("equal")
    ax_top.grid(alpha=0.3)
    ax_top.legend()
    ax_top.set_title("top view")

    ax_3d.plot(plan[:, 0], plan[:, 1], plan[:, 2], "-", color="tab:green", lw=1.0)
    if result is not None and result.rollout is not None and len(result.rollout.log["t"]):
        a = result.rollout.log["pos"]
        ax_3d.plot(a[:, 0], a[:, 1], a[:, 2], "-", color="tab:red", lw=0.8, alpha=0.8)
    ax_3d.set_title("3D view")

    arena = task.arena
    ax_top.set_xlim(arena.min[0] - 1, arena.max[0] + 1)
    ax_top.set_ylim(arena.min[1] - 1, arena.max[1] + 1)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
