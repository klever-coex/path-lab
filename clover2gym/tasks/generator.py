"""Seeded procedural generation of racing tracks.

Used to publish practice tracks and to mint the hidden evaluation set from
the same distribution.
"""
from __future__ import annotations

import numpy as np

from ..tasks.spec import Bounds, Gate, Track

SIZES = [(1.8, 1.4), (1.7, 1.3), (1.6, 1.25), (1.9, 1.5)]
CIRCLE_PROB = 0.2


def generate_track(
    seed: int,
    name: str | None = None,
    n_gates: int = 10,
    min_seg: float = 5.0,
    max_seg: float = 9.0,
    max_turn_deg: float = 55.0,
    min_gate_separation: float = 3.5,
) -> Track:
    rng = np.random.default_rng(seed)
    heading = rng.uniform(0, 2 * np.pi)
    pos = np.array([0.0, 0.0, 1.3])
    gates: list[Gate] = []

    for k in range(n_gates):
        for _attempt in range(50):
            turn = np.deg2rad(rng.uniform(-max_turn_deg, max_turn_deg))
            heading_new = heading + turn
            d = rng.uniform(min_seg, max_seg)
            center = pos[:2] + d * np.array([np.cos(heading_new), np.sin(heading_new)])
            z = float(np.clip(pos[2] + rng.uniform(-1.0, 1.0), 1.1, 3.2))
            if all(np.hypot(*(center - g.center[:2])) >= min_gate_separation for g in gates):
                break
        heading = heading_new
        pos = np.array([center[0], center[1], z])

        normal = np.array([np.cos(heading), np.sin(heading), rng.uniform(-0.05, 0.05)])
        size = SIZES[rng.integers(len(SIZES))]
        shape = "circle" if rng.random() < CIRCLE_PROB else "rect"
        if shape == "circle":
            size = (size[0], size[0])
        gates.append(Gate(center=[center[0], center[1], z], normal=normal, size=size, shape=shape))

    all_pts = np.vstack([g.center for g in gates] + [[0.0, 0.0, 1.3]])
    arena = Bounds(all_pts.min(axis=0) - 4.0, all_pts.max(axis=0) + 4.0)
    arena.min[2] = 0.0
    arena.max[2] = max(arena.max[2], 6.0)

    return Track(
        name=name or f"gen{seed}",
        gates=gates,
        start_pos=[0.0, 0.0, 1.3],
        start_yaw=float(heading),
        arena=arena,
    )
