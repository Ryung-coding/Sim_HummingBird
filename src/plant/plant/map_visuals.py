"""Non-colliding visual landmarks for the mapping-only corridor."""

import mujoco
import numpy as np

from plant.plant import to_zdown


COLORS = (
    (0.95, 0.95, 0.95, 1.0),
    (0.05, 0.05, 0.05, 1.0),
    (0.95, 0.15, 0.10, 1.0),
    (0.10, 0.85, 0.25, 1.0),
    (0.15, 0.35, 0.95, 1.0),
    (0.95, 0.80, 0.10, 1.0),
)


def add_map_visuals(spec, obstacles):
    for obstacle in obstacles:
        name = obstacle["name"]
        if name not in ("left_feature_wall", "right_feature_wall", "end_marker",
                        "near_feature_panel", "mid_feature_panel"):
            continue
        center = np.asarray(obstacle["center"], dtype=float)
        size = np.asarray(obstacle["size"], dtype=float)
        rng = np.random.default_rng(sum(ord(c) for c in name))
        if name in ("end_marker", "near_feature_panel", "mid_feature_panel"):
            cols = 8 if name == "end_marker" else 5
            rows = 7 if name == "end_marker" else 6
            spacing_y = 0.135 if name == "end_marker" else 0.075
            spacing_z = 0.17 if name == "end_marker" else 0.20
            for iy in range(cols):
                for iz in range(rows):
                    xyz = [center[0] - size[0] / 2.0 - 0.012,
                           center[1] + (iy - (cols - 1) / 2.0) * spacing_y,
                           center[2] + (iz - (rows - 1) / 2.0) * spacing_z]
                    spec.worldbody.add_geom(
                        name=f"{name}_tile_{iy}_{iz}", type=mujoco.mjtGeom.mjGEOM_BOX,
                        pos=to_zdown(xyz), size=[0.006, 0.028 if cols == 5 else 0.050,
                                                        0.070 if cols == 5 else 0.060],
                        rgba=COLORS[int(rng.integers(len(COLORS)))],
                        contype=0, conaffinity=0, group=0)
        else:
            inner_y = center[1] - np.sign(center[1]) * (size[1] / 2.0 + 0.012)
            for ix in range(25):
                for iz in range(4):
                    xyz = [center[0] + (ix - 12.0) * 0.53,
                           inner_y,
                           center[2] + (iz - 1.5) * 0.32]
                    spec.worldbody.add_geom(
                        name=f"{name}_tile_{ix}_{iz}", type=mujoco.mjtGeom.mjGEOM_BOX,
                        pos=to_zdown(xyz), size=[0.060, 0.006, 0.065],
                        rgba=COLORS[int(rng.integers(len(COLORS)))],
                        contype=0, conaffinity=0, group=0)
