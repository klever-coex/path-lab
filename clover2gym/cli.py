"""Command-line interface: run, check, viz, run-set."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from .tasks.spec import MotionPlanningTask
from .util.solution import Solution

app = typer.Typer(
    help=__doc__,
    add_completion=False,
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.callback()
def configure_logging(
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose", "-v", count=True,
            help="log internal events: -v for INFO (gate splits), -vv for DEBUG",
        ),
    ] = 0,
) -> None:
    logging.basicConfig(
        level=max(logging.DEBUG, logging.WARNING - 10 * verbose),
        format="%(levelname)s %(name)s: %(message)s",
    )


# ---------------------------------------------------------------------- #
# commands
# ---------------------------------------------------------------------- #
@app.command("run", help="full check: static + simulated flight")
def cmd_run(
    task: Annotated[Path, typer.Option(help="track yaml")],
    solution: Annotated[Path, typer.Option(help="solution py with solve(task)")],
    gui: Annotated[bool, typer.Option("--gui", help="show the flight in pybullet GUI")] = False,
    controller: Annotated[str | None, typer.Option(help="override the task's tracking controller by name")] = None,
    out: Annotated[Path | None, typer.Option(help="write JSON report here")] = None,
    plot: Annotated[Path | None, typer.Option(help="write a PNG plot here")] = None,
    no_follow: Annotated[bool, typer.Option("--no-follow", help="disable the chase camera with --gui (or press 'f' live)")] = False,
) -> None:
    from .scoring import evaluate  # import here to avoid pybullet dependency for other commands

    task = MotionPlanningTask.from_yaml(task)
    if controller:
        task.controller = controller
    traj = Solution(solution).solve(task)
    result = evaluate(task, traj, gui=gui, follow_camera=not no_follow)

    print(f"track:        {task.track.name}")
    print(f"static check: {'PASS' if result.static.ok else 'FAIL'}")
    for v in result.static.violations:
        print(f"  [static] {v.message} (t={v.t}, limit={v.limit})")
    if result.sim is not None:
        print(f"sim check:    {'PASS' if result.sim.ok else 'FAIL'}")
        for v in result.sim.violations:
            print(f"  [sim] {v.message}")
        print(f"lap time:     {result.lap_time:.3f} s" if result.lap_time else "lap time:     —")
        for key in ("tracking_err_max_m", "tracking_err_mean_m", "max_speed_actual", "min_clearance_actual_m"):
            if key in result.sim.stats:
                print(f"  {key}: {result.sim.stats[key]}")
    print(f"RESULT:       {'PASS' if result.passed else 'FAIL'}"
          + (f" ({result.lap_time:.3f} s)" if result.lap_time else ""))

    if out:
        result.to_json(out)
        print(f"report:       {out}")
    if plot:
        try:
            from .plotting import plot_evaluation

            plot_evaluation(task, traj, result, path=plot)
            print(f"plot:         {plot}")
        except ImportError:
            print("matplotlib not installed; skipping plot")
    raise typer.Exit(code=0 if result.passed else 1)


@app.command("check", help="fast static check only (no pybullet)")
def cmd_check(
    task: Annotated[Path, typer.Option(help="track yaml")],
    solution: Annotated[Path, typer.Option(help="solution py with solve(task)")],
    out: Annotated[Path | None, typer.Option(help="write JSON report here")] = None,
) -> None:
    task = MotionPlanningTask.from_yaml(task)
    traj = Solution(solution).solve(task)
    from .validation.static import StaticChecker

    report = StaticChecker(task).check(traj)
    print(report.to_markdown())
    if out:
        report.to_json(out)
    raise typer.Exit(code=0 if report.ok else 1)


@app.command("run-set", help="evaluate a solution on a set of tracks")
def cmd_run_set(
    task_dir: Annotated[Path, typer.Option(help="directory with track yaml files")],
    solution: Annotated[Path, typer.Option(help="solution py with solve(task)")],
) -> None:
    from .scoring import evaluate_set  # import here to avoid pybullet dependency for other commands

    task_files = sorted(Path(task_dir).glob("*.yaml"))
    if not task_files:
        raise SystemExit(f"no *.yaml tracks in {task_dir}")
    _solution = Solution(solution)
    tasks, traj_map = [], {}
    for f in task_files:
        task = MotionPlanningTask.from_yaml(f)
        tasks.append(task)
        traj_map[task.track.name] = _solution.solve(task)
    summary = evaluate_set(tasks, traj_map)

    print(f"{'track':<20} {'result':<8} {'lap, s':>8}")
    for item in summary["results"]:
        r = item["result"]
        lap = f"{r.lap_time:.3f}" if r.lap_time else "—"
        print(f"{r.track_name:<20} {'PASS' if r.passed else 'FAIL':<8} {lap:>8}")
    print(f"total: {summary['total_time']:.3f} s (failed tracks count as t_max)")


@app.command("viz", help="visualize a track and optionally a solution's planned path")
def cmd_viz(
    task: Annotated[Path, typer.Option(help="track yaml")],
    solution: Annotated[Path | None, typer.Option(help="draw the planned path of this solution")] = None,
    camera: Annotated[
        str | None,
        typer.Option(
            help="initial course/path preview camera: dist,yaw,pitch,tx,ty,tz "
            "(default 18,45,-35,0,0,1.5); left-drag orbits, wheel zooms, Ctrl+left pans",
        ),
    ] = None,
) -> None:
    camera_values = None
    if camera is not None:
        try:
            dist, yaw, pitch, tx, ty, tz = map(float, camera.split(","))
        except ValueError:
            raise typer.BadParameter("expected six numbers: dist,yaw,pitch,tx,ty,tz", param_hint="--camera") from None
        camera_values = (dist, yaw, pitch, [tx, ty, tz])

    task = MotionPlanningTask.from_yaml(task)

    from .model.drone import DroneModel
    from .sim.world import RaceWorld

    world = RaceWorld(task, DroneModel.racer(), gui=True, spawn_drone=False)
    if camera_values is not None:
        import pybullet as p

        p.resetDebugVisualizerCamera(*camera_values, physicsClientId=world.client)
    if solution:
        from .sim.rollout import _draw_path

        _draw_path(world, Solution(solution).solve(task))
    world.view()


if __name__ == "__main__":
    app()
