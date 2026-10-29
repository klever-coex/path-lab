"""Gate geometry: local frames, plane crossings, distances to gate frames.

Used by both the static checker and the simulator, so the definition of
"passed the gate" is exactly the same in both stages.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..tasks.spec import Gate

BEAM_THICKNESS = 0.05  # m, structural thickness of gate frame beams


def gate_frame(gate: Gate) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right-handed orthonormal frame (u, v, n) of a gate.

    n is the gate normal (direction of travel), v is the in-plane "up"
    (world +z projected onto the plane; world +x for horizontal gates),
    u = v x n completes the frame so that u x v = n.
    """
    n = gate.normal
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(ref, n))) > 0.99:  # horizontal gate
        ref = np.array([1.0, 0.0, 0.0])
    v = ref - np.dot(ref, n) * n
    v = v / np.linalg.norm(v)
    u = np.cross(v, n)
    return u, v, n


def to_gate_local(gate: Gate, points: np.ndarray) -> np.ndarray:
    """World points -> gate-local coordinates (a, b, s): u, v, normal axes."""
    u, v, n = gate_frame(gate)
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    rel = pts - gate.center
    return np.column_stack((rel @ u, rel @ v, rel @ n))


@dataclass
class GateCrossing:
    """A crossing of the gate plane in the normal direction."""

    t: float                 # time of crossing
    point: np.ndarray        # (3,) interpolated crossing point
    local_xy: np.ndarray     # (2,) coordinates in the gate plane
    speed: float             # |velocity| at the crossing
    speed_n: float           # velocity component along the normal

    def inset(self, gate: Gate) -> float:
        """Distance from the opening edge (>= 0 means inside the opening)."""
        a, b = self.local_xy
        if gate.shape == "circle":
            return gate.size[0] / 2 - float(np.hypot(a, b))
        return min(gate.size[0] / 2 - abs(a), gate.size[1] / 2 - abs(b))

    def is_valid(self, gate: Gate, margin: float) -> bool:
        return self.inset(gate) >= margin and self.speed_n > 0.0


def find_gate_crossings(
    gate: Gate, times: np.ndarray, positions: np.ndarray, start_idx: int = 0
) -> list[GateCrossing]:
    """All plane crossings in the normal direction (s: - -> +), in time order."""
    local = to_gate_local(gate, positions)
    s = local[:, 2]
    crossings: list[GateCrossing] = []
    for i in range(max(start_idx, 0), len(s) - 1):
        if s[i] <= 0.0 < s[i + 1]:
            alpha = -s[i] / (s[i + 1] - s[i])
            t = times[i] + alpha * (times[i + 1] - times[i])
            point = positions[i] + alpha * (positions[i + 1] - positions[i])
            seg = (positions[i + 1] - positions[i]) / (times[i + 1] - times[i])
            xy = to_gate_local(gate, point)[0, :2]
            crossings.append(
                GateCrossing(
                    t=float(t),
                    point=point,
                    local_xy=xy,
                    speed=float(np.linalg.norm(seg)),
                    speed_n=float(np.dot(seg, gate.normal)),
                )
            )
    return crossings


def distance_to_gate(gate: Gate, points: np.ndarray) -> np.ndarray:
    """Distance from world points to the gate's structural frame (beams)."""
    local = to_gate_local(gate, points)
    a, b, s = local[:, 0], local[:, 1], local[:, 2]
    if gate.shape == "circle":
        # torus with circular cross-section of radius BEAM_THICKNESS/2
        r = gate.size[0] / 2
        rho = np.hypot(a, b)
        d_ring = np.maximum(np.abs(rho - r) - BEAM_THICKNESS / 2, 0.0)
        d_axis = np.maximum(np.abs(s) - BEAM_THICKNESS / 2, 0.0)
        return np.hypot(d_ring, d_axis)
    # rect gate: 4 axis-aligned boxes in gate-local coords
    w2, h2 = gate.size[0] / 2, gate.size[1] / 2
    th2 = BEAM_THICKNESS / 2
    boxes = [
        ((0.0, h2 + th2), (w2 + th2, th2)),   # top
        ((0.0, -(h2 + th2)), (w2 + th2, th2)),  # bottom
        ((-(w2 + th2), 0.0), (th2, h2 + th2)),  # left
        (((w2 + th2), 0.0), (th2, h2 + th2)),   # right
    ]
    dist = np.full(len(a), np.inf)
    for (cu, cv), (hu, hv) in boxes:
        du = np.maximum(np.abs(a - cu) - hu, 0.0)
        dv = np.maximum(np.abs(b - cv) - hv, 0.0)
        ds = np.maximum(np.abs(s) - th2, 0.0)
        dist = np.minimum(dist, np.sqrt(du**2 + dv**2 + ds**2))
    return dist


def gate_collision_boxes(gate: Gate) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Collision/visual boxes approximating the gate frame.

    Returns a list of (center_world, half_extents, rotation_matrix) tuples,
    one per box. Rect gates use 4 exact beams; circle gates use 12 segments.
    """
    u, v, n = gate_frame(gate)
    R = np.column_stack((u, v, n))  # gate-local -> world
    th = BEAM_THICKNESS
    out: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    if gate.shape == "rect":
        w2, h2 = gate.size[0] / 2, gate.size[1] / 2
        beams = [
            ((0.0, h2 + th / 2), (w2 + th / 2, th / 2)),
            ((0.0, -(h2 + th / 2)), (w2 + th / 2, th / 2)),
            ((-(w2 + th / 2), 0.0), (th / 2, h2 + th / 2)),
            (((w2 + th / 2), 0.0), (th / 2, h2 + th / 2)),
        ]
        for (cu, cv), (hu, hv) in beams:
            center = gate.center + R @ np.array([cu, cv, 0.0])
            out.append((center, np.array([hu, hv, th / 2]), R))
    else:
        r = gate.size[0] / 2
        n_seg = 12
        mid_r = r + th / 2
        seg_len = 2 * mid_r * np.sin(np.pi / n_seg)
        for k in range(n_seg):
            theta = 2 * np.pi * (k + 0.5) / n_seg
            local_center = np.array([mid_r * np.cos(theta), mid_r * np.sin(theta), 0.0])
            tangential = np.array([-np.sin(theta), np.cos(theta), 0.0])
            radial = np.array([np.cos(theta), np.sin(theta), 0.0])
            # right-handed frame: tangential x (-radial) = +z
            box_axes = np.column_stack((tangential, -radial, [0.0, 0.0, 1.0]))
            center = gate.center + R @ local_center
            out.append((center, np.array([seg_len / 2 + th / 2, th / 2, th / 2]), R @ box_axes))
    return out
