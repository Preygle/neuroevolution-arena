from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ArenaMap:
    name: str
    terrain: np.ndarray


def build_training_map_pool(grid_size: int) -> list[ArenaMap]:
    return [
        ArenaMap("open-plaza", _open_plaza(grid_size)),
        ArenaMap("crossfire", _crossfire(grid_size)),
        ArenaMap("ring-run", _ring_run(grid_size)),
        ArenaMap("double-trench", _double_trench(grid_size)),
        ArenaMap("pillar-field", _pillar_field(grid_size)),
        ArenaMap("lane-control", _lane_control(grid_size)),
        ArenaMap("broken-bridge", _broken_bridge(grid_size)),
        ArenaMap("diamond-fort", _diamond_fort(grid_size)),
    ]


def _open_plaza(grid_size: int) -> np.ndarray:
    terrain = np.zeros((grid_size, grid_size), dtype=bool)
    center = grid_size // 2
    offsets = (-3, 3)
    for dx in offsets:
        for dy in offsets:
            terrain[center + dx, center + dy] = True
            terrain[center + dx + 1, center + dy] = True
    return terrain


def _crossfire(grid_size: int) -> np.ndarray:
    terrain = np.zeros((grid_size, grid_size), dtype=bool)
    center = grid_size // 2
    terrain[:, center] = True
    terrain[center, :] = True
    for offset in (-4, -1, 1, 4):
        terrain[center + offset, center] = False
        terrain[center, center + offset] = False
    terrain[center, center] = False
    return terrain


def _ring_run(grid_size: int) -> np.ndarray:
    terrain = np.zeros((grid_size, grid_size), dtype=bool)
    low = 3
    high = grid_size - 4
    terrain[low : high + 1, low] = True
    terrain[low : high + 1, high] = True
    terrain[low, low : high + 1] = True
    terrain[high, low : high + 1] = True
    gates = ((low, grid_size // 2), (high, grid_size // 2), (grid_size // 2, low), (grid_size // 2, high))
    for x, y in gates:
        terrain[x, y] = False
        if x == low or x == high:
            terrain[x, y - 1 : y + 2] = False
        else:
            terrain[x - 1 : x + 2, y] = False
    return terrain


def _double_trench(grid_size: int) -> np.ndarray:
    terrain = np.zeros((grid_size, grid_size), dtype=bool)
    rows = (grid_size // 2 - 2, grid_size // 2 + 2)
    for row in rows:
        terrain[row, 2 : grid_size - 2] = True
    for gate in (3, grid_size // 2, grid_size - 4):
        for row in rows:
            terrain[row, gate] = False
            terrain[row, gate - 1] = False
    return terrain


def _pillar_field(grid_size: int) -> np.ndarray:
    terrain = np.zeros((grid_size, grid_size), dtype=bool)
    anchors = (3, grid_size // 2, grid_size - 4)
    for x in anchors:
        for y in anchors:
            if (x, y) == (grid_size // 2, grid_size // 2):
                continue
            terrain[x : x + 2, y : y + 2] = True
    return terrain


def _lane_control(grid_size: int) -> np.ndarray:
    terrain = np.zeros((grid_size, grid_size), dtype=bool)
    cols = (grid_size // 3, 2 * grid_size // 3)
    for col in cols:
        terrain[2 : grid_size - 2, col] = True
    for gate in (3, grid_size // 2, grid_size - 4):
        for col in cols:
            terrain[gate, col] = False
            terrain[gate - 1, col] = False
    return terrain


def _broken_bridge(grid_size: int) -> np.ndarray:
    terrain = np.zeros((grid_size, grid_size), dtype=bool)
    row = grid_size // 2
    terrain[row, 1 : grid_size - 1] = True
    for gate in (2, grid_size // 2 - 1, grid_size - 3):
        terrain[row, gate : gate + 2] = False
    terrain[2 : grid_size - 2, grid_size // 2] = True
    terrain[row - 1 : row + 2, grid_size // 2] = False
    return terrain


def _diamond_fort(grid_size: int) -> np.ndarray:
    terrain = np.zeros((grid_size, grid_size), dtype=bool)
    center = grid_size // 2
    radius = max(3, grid_size // 4)
    for x in range(grid_size):
        for y in range(grid_size):
            distance = abs(x - center) + abs(y - center)
            if distance == radius:
                terrain[x, y] = True
    terrain[center, center - radius] = False
    terrain[center, center + radius] = False
    terrain[center - radius, center] = False
    terrain[center + radius, center] = False
    return terrain
