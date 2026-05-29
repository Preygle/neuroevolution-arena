from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from behavior_mutation_arena.config import EnemyKind, PowerUpType


@dataclass(frozen=True)
class ChestSpawn:
    position: tuple[int, int]
    powerups: tuple[PowerUpType, ...]

    @property
    def powerup(self) -> PowerUpType:
        return self.powerups[0]


@dataclass(frozen=True)
class EnemySpawn:
    position: tuple[int, int]
    kind: EnemyKind


@dataclass(frozen=True)
class DungeonFloor:
    index: int
    name: str
    theme: str
    terrain: np.ndarray
    slow_tiles: np.ndarray
    hazard_tiles: np.ndarray
    heal_tiles: np.ndarray
    gate_position: tuple[int, int]
    start_positions: tuple[tuple[int, int], ...]
    chests: tuple[ChestSpawn, ...]
    enemies: tuple[EnemySpawn, ...]
    gate_locked_until_boss: bool = False


@dataclass(frozen=True)
class StitchedDungeonLayout:
    floor_offsets: tuple[tuple[int, int], ...]
    floor_shape: tuple[int, int]
    global_shape: tuple[int, int]
    entrance_anchor: tuple[int, int]

    def floor_offset(self, floor_index: int) -> tuple[int, int]:
        return self.floor_offsets[floor_index]

    def to_global(self, floor_index: int, position: tuple[int, int]) -> tuple[int, int]:
        offset_x, offset_y = self.floor_offset(floor_index)
        return offset_x + position[0], offset_y + position[1]


def build_dungeon_floors(grid_size: int) -> list[DungeonFloor]:
    return [
        _verdant_entry(grid_size),
        _glacier_pass(grid_size),
        _ember_forges(grid_size),
        _cryptic_stacks(grid_size),
        _warden_keep(grid_size),
        _swamp_descent(grid_size),
        _crystal_caverns(grid_size),
        _storm_bastion(grid_size),
        _inferno_ascent(grid_size),
        _abyss_throne(grid_size),
    ]


def build_stitched_dungeon_layout(floors: list[DungeonFloor]) -> StitchedDungeonLayout:
    if not floors:
        raise ValueError("at least one dungeon floor is required")

    entrance_anchor = _entrance_anchor(floors[0])
    offsets: list[tuple[int, int]] = [(0, 0)]
    for floor_index in range(1, len(floors)):
        previous_floor = floors[floor_index - 1]
        previous_offset = offsets[floor_index - 1]
        current_anchor = _entrance_anchor(floors[floor_index])
        offsets.append(
            (
                previous_offset[0] + previous_floor.gate_position[0] - current_anchor[0],
                previous_offset[1] + previous_floor.gate_position[1] - current_anchor[1],
            )
        )

    min_x = min(offset[0] for offset in offsets)
    min_y = min(offset[1] for offset in offsets)
    if min_x < 0 or min_y < 0:
        offsets = [(x - min_x, y - min_y) for x, y in offsets]

    rows, columns = floors[0].terrain.shape
    max_x = max(offset[0] + rows for offset in offsets)
    max_y = max(offset[1] + columns for offset in offsets)
    return StitchedDungeonLayout(
        floor_offsets=tuple(offsets),
        floor_shape=(rows, columns),
        global_shape=(max_x, max_y),
        entrance_anchor=entrance_anchor,
    )


def _entrance_anchor(floor: DungeonFloor) -> tuple[int, int]:
    rows = [position[0] for position in floor.start_positions]
    columns = [position[1] for position in floor.start_positions]
    return (round(sum(rows) / len(rows)), round(sum(columns) / len(columns)))


def _start_positions() -> tuple[tuple[int, int], ...]:
    return ((2, 2), (2, 4), (4, 2), (4, 4), (3, 3))


def _blank_layers(grid_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    terrain = np.ones((grid_size, grid_size), dtype=bool)
    slow = np.zeros((grid_size, grid_size), dtype=bool)
    hazard = np.zeros((grid_size, grid_size), dtype=bool)
    heal = np.zeros((grid_size, grid_size), dtype=bool)
    return terrain, slow, hazard, heal


def _carve_room(terrain: np.ndarray, top: int, left: int, bottom: int, right: int) -> None:
    terrain[top:bottom, left:right] = False


def _carve_hallway(terrain: np.ndarray, row: int, start_col: int, end_col: int, width: int = 2) -> None:
    left = min(start_col, end_col)
    right = max(start_col, end_col)
    terrain[row : row + width, left : right + 1] = False


def _carve_vertical(terrain: np.ndarray, col: int, start_row: int, end_row: int, width: int = 2) -> None:
    top = min(start_row, end_row)
    bottom = max(start_row, end_row)
    terrain[top : bottom + 1, col : col + width] = False


def _paint_zone(layer: np.ndarray, top: int, left: int, bottom: int, right: int) -> None:
    layer[top:bottom, left:right] = True


def _clear_zone(layer: np.ndarray, top: int, left: int, bottom: int, right: int) -> None:
    layer[top:bottom, left:right] = False


def _verdant_entry(grid_size: int) -> DungeonFloor:
    terrain, slow, hazard, heal = _blank_layers(grid_size)
    _carve_room(terrain, 1, 1, 7, 8)
    _carve_hallway(terrain, 3, 7, 17)
    _carve_room(terrain, 1, 16, 10, 24)
    _carve_vertical(terrain, 19, 8, 20)
    _carve_room(terrain, 18, 15, 26, 24)
    _carve_hallway(terrain, 21, 23, 31)
    _carve_room(terrain, 18, 30, 28, 35)
    _carve_vertical(terrain, 32, 25, 32)
    _carve_room(terrain, 28, 26, 35, 35)
    _paint_zone(slow, 2, 16, 8, 23)
    _paint_zone(heal, 19, 16, 25, 23)
    return DungeonFloor(
        index=1,
        name="Verdant Entry",
        theme="jungle",
        terrain=terrain,
        slow_tiles=slow,
        hazard_tiles=hazard,
        heal_tiles=heal,
        gate_position=(32, 32),
        start_positions=_start_positions(),
        chests=(
            ChestSpawn((5, 20), (PowerUpType.VITALITY, PowerUpType.DAMAGE)),
            ChestSpawn((22, 19), (PowerUpType.DAMAGE, PowerUpType.VITALITY)),
        ),
        enemies=(
            EnemySpawn((20, 21), EnemyKind.SKIRMISHER),
            EnemySpawn((31, 29), EnemyKind.ARCHER),
        ),
    )


def _glacier_pass(grid_size: int) -> DungeonFloor:
    terrain, slow, hazard, heal = _blank_layers(grid_size)
    _carve_room(terrain, 1, 1, 7, 8)
    _carve_vertical(terrain, 5, 6, 14)
    _carve_room(terrain, 12, 2, 20, 11)
    _carve_hallway(terrain, 16, 10, 24)
    _carve_room(terrain, 12, 23, 22, 31)
    _carve_vertical(terrain, 27, 21, 30)
    _carve_room(terrain, 28, 24, 35, 35)
    _paint_zone(slow, 12, 2, 20, 11)
    _paint_zone(slow, 12, 23, 22, 31)
    return DungeonFloor(
        index=2,
        name="Glacier Pass",
        theme="ice",
        terrain=terrain,
        slow_tiles=slow,
        hazard_tiles=hazard,
        heal_tiles=heal,
        gate_position=(32, 31),
        start_positions=_start_positions(),
        chests=(
            ChestSpawn((16, 6), (PowerUpType.SPEED, PowerUpType.VITALITY)),
            ChestSpawn((16, 27), (PowerUpType.DIAGONAL, PowerUpType.SPEED)),
        ),
        enemies=(
            EnemySpawn((16, 19), EnemyKind.ARCHER),
            EnemySpawn((30, 29), EnemyKind.SKIRMISHER),
        ),
    )


def _ember_forges(grid_size: int) -> DungeonFloor:
    terrain, slow, hazard, heal = _blank_layers(grid_size)
    _carve_room(terrain, 1, 1, 7, 8)
    _carve_hallway(terrain, 4, 7, 13)
    _carve_room(terrain, 1, 12, 10, 20)
    _carve_vertical(terrain, 15, 9, 18)
    _carve_room(terrain, 16, 11, 26, 21)
    _carve_hallway(terrain, 20, 20, 30)
    _carve_room(terrain, 14, 29, 24, 35)
    _carve_vertical(terrain, 32, 23, 31)
    _carve_room(terrain, 28, 26, 35, 35)
    _paint_zone(hazard, 16, 11, 26, 21)
    _paint_zone(hazard, 15, 29, 24, 34)
    _clear_zone(hazard, 19, 15, 22, 34)
    _clear_zone(hazard, 28, 31, 34, 34)
    _paint_zone(heal, 21, 30, 23, 34)
    return DungeonFloor(
        index=3,
        name="Ember Forges",
        theme="lava",
        terrain=terrain,
        slow_tiles=slow,
        hazard_tiles=hazard,
        heal_tiles=heal,
        gate_position=(32, 32),
        start_positions=_start_positions(),
        chests=(
            ChestSpawn((5, 16), (PowerUpType.DAMAGE, PowerUpType.VITALITY)),
            ChestSpawn((19, 32), (PowerUpType.RANGE, PowerUpType.SPEED)),
        ),
        enemies=(
            EnemySpawn((19, 16), EnemyKind.SKIRMISHER),
            EnemySpawn((19, 30), EnemyKind.ARCHER),
            EnemySpawn((29, 30), EnemyKind.SKIRMISHER),
        ),
    )


def _cryptic_stacks(grid_size: int) -> DungeonFloor:
    terrain, slow, hazard, heal = _blank_layers(grid_size)
    _carve_room(terrain, 1, 1, 7, 8)
    _carve_vertical(terrain, 4, 6, 18)
    _carve_room(terrain, 16, 1, 24, 10)
    _carve_hallway(terrain, 19, 9, 20)
    _carve_room(terrain, 14, 19, 24, 27)
    _carve_vertical(terrain, 23, 23, 30)
    _carve_room(terrain, 28, 18, 35, 27)
    _carve_hallway(terrain, 31, 26, 33)
    _carve_room(terrain, 25, 32, 35, 35)
    _paint_zone(slow, 16, 1, 24, 10)
    _paint_zone(heal, 28, 18, 35, 27)
    return DungeonFloor(
        index=4,
        name="Cryptic Stacks",
        theme="ruins",
        terrain=terrain,
        slow_tiles=slow,
        hazard_tiles=hazard,
        heal_tiles=heal,
        gate_position=(30, 33),
        start_positions=_start_positions(),
        chests=(
            ChestSpawn((20, 5), (PowerUpType.VITALITY, PowerUpType.DAMAGE)),
            ChestSpawn((18, 23), (PowerUpType.SPEED, PowerUpType.RANGE)),
        ),
        enemies=(
            EnemySpawn((20, 17), EnemyKind.ARCHER),
            EnemySpawn((31, 23), EnemyKind.SKIRMISHER),
        ),
    )


def _warden_keep(grid_size: int) -> DungeonFloor:
    terrain, slow, hazard, heal = _blank_layers(grid_size)
    _carve_room(terrain, 1, 1, 7, 8)
    _carve_hallway(terrain, 4, 7, 15)
    _carve_room(terrain, 1, 14, 10, 23)
    _carve_hallway(terrain, 8, 18, 26)
    _carve_room(terrain, 5, 25, 13, 34)
    _carve_vertical(terrain, 29, 12, 19)
    _carve_room(terrain, 17, 21, 34, 35)
    _paint_zone(slow, 17, 21, 24, 35)
    _paint_zone(heal, 14, 27, 19, 32)
    return DungeonFloor(
        index=5,
        name="Warden Keep",
        theme="stronghold",
        terrain=terrain,
        slow_tiles=slow,
        hazard_tiles=hazard,
        heal_tiles=heal,
        gate_position=(31, 32),
        start_positions=_start_positions(),
        chests=(
            ChestSpawn((5, 18), (PowerUpType.DAMAGE, PowerUpType.VITALITY)),
            ChestSpawn((8, 30), (PowerUpType.RANGE, PowerUpType.SPEED)),
        ),
        enemies=(
            EnemySpawn((12, 29), EnemyKind.ARCHER),
            EnemySpawn((25, 28), EnemyKind.MINI_BOSS),
        ),
        gate_locked_until_boss=True,
    )


def _swamp_descent(grid_size: int) -> DungeonFloor:
    terrain, slow, hazard, heal = _blank_layers(grid_size)
    _carve_room(terrain, 1, 1, 7, 8)
    _carve_vertical(terrain, 5, 6, 14)
    _carve_room(terrain, 12, 2, 19, 11)
    _carve_hallway(terrain, 15, 10, 18)
    _carve_room(terrain, 11, 17, 20, 26)
    _carve_vertical(terrain, 21, 19, 28)
    _carve_room(terrain, 27, 16, 35, 25)
    _carve_hallway(terrain, 31, 24, 33)
    _carve_room(terrain, 26, 32, 35, 35)
    _paint_zone(slow, 12, 2, 19, 11)
    _paint_zone(hazard, 27, 16, 35, 25)
    return DungeonFloor(
        index=6,
        name="Swamp Descent",
        theme="swamp",
        terrain=terrain,
        slow_tiles=slow,
        hazard_tiles=hazard,
        heal_tiles=heal,
        gate_position=(30, 33),
        start_positions=_start_positions(),
        chests=(
            ChestSpawn((15, 6), (PowerUpType.VITALITY, PowerUpType.DAMAGE)),
            ChestSpawn((15, 21), (PowerUpType.DIAGONAL, PowerUpType.SPEED)),
        ),
        enemies=(
            EnemySpawn((16, 18), EnemyKind.SKIRMISHER),
            EnemySpawn((30, 21), EnemyKind.ARCHER),
        ),
    )


def _crystal_caverns(grid_size: int) -> DungeonFloor:
    terrain, slow, hazard, heal = _blank_layers(grid_size)
    _carve_room(terrain, 1, 1, 7, 8)
    _carve_hallway(terrain, 3, 7, 18)
    _carve_room(terrain, 1, 17, 10, 27)
    _carve_vertical(terrain, 22, 9, 18)
    _carve_room(terrain, 16, 18, 26, 28)
    _carve_hallway(terrain, 21, 5, 18)
    _carve_room(terrain, 18, 4, 28, 14)
    _carve_vertical(terrain, 9, 27, 31)
    _carve_room(terrain, 28, 6, 35, 14)
    _paint_zone(heal, 18, 4, 28, 14)
    _paint_zone(heal, 16, 18, 26, 28)
    return DungeonFloor(
        index=7,
        name="Crystal Caverns",
        theme="crystal",
        terrain=terrain,
        slow_tiles=slow,
        hazard_tiles=hazard,
        heal_tiles=heal,
        gate_position=(31, 10),
        start_positions=_start_positions(),
        chests=(
            ChestSpawn((5, 22), (PowerUpType.RANGE, PowerUpType.DAMAGE)),
            ChestSpawn((23, 9), (PowerUpType.SPEED, PowerUpType.VITALITY)),
        ),
        enemies=(
            EnemySpawn((20, 22), EnemyKind.ARCHER),
            EnemySpawn((22, 9), EnemyKind.SKIRMISHER),
            EnemySpawn((31, 12), EnemyKind.SKIRMISHER),
        ),
    )


def _storm_bastion(grid_size: int) -> DungeonFloor:
    terrain, slow, hazard, heal = _blank_layers(grid_size)
    _carve_room(terrain, 1, 1, 7, 8)
    _carve_vertical(terrain, 4, 6, 16)
    _carve_room(terrain, 14, 1, 22, 10)
    _carve_hallway(terrain, 17, 9, 22)
    _carve_room(terrain, 13, 21, 22, 31)
    _carve_vertical(terrain, 26, 21, 30)
    _carve_room(terrain, 28, 19, 35, 35)
    _paint_zone(hazard, 14, 1, 22, 10)
    _paint_zone(slow, 13, 21, 22, 31)
    return DungeonFloor(
        index=8,
        name="Storm Bastion",
        theme="storm",
        terrain=terrain,
        slow_tiles=slow,
        hazard_tiles=hazard,
        heal_tiles=heal,
        gate_position=(31, 31),
        start_positions=_start_positions(),
        chests=(
            ChestSpawn((17, 5), (PowerUpType.DAMAGE, PowerUpType.VITALITY)),
            ChestSpawn((17, 26), (PowerUpType.DIAGONAL, PowerUpType.RANGE)),
        ),
        enemies=(
            EnemySpawn((17, 16), EnemyKind.ARCHER),
            EnemySpawn((31, 25), EnemyKind.SKIRMISHER),
            EnemySpawn((31, 30), EnemyKind.ARCHER),
        ),
    )


def _inferno_ascent(grid_size: int) -> DungeonFloor:
    terrain, slow, hazard, heal = _blank_layers(grid_size)
    _carve_room(terrain, 1, 1, 7, 8)
    _carve_hallway(terrain, 4, 7, 14)
    _carve_room(terrain, 1, 13, 9, 22)
    _carve_vertical(terrain, 18, 8, 18)
    _carve_room(terrain, 16, 14, 26, 24)
    _carve_hallway(terrain, 20, 23, 31)
    _carve_room(terrain, 14, 30, 24, 35)
    _carve_vertical(terrain, 32, 23, 31)
    _carve_room(terrain, 28, 26, 35, 35)
    _paint_zone(hazard, 16, 14, 26, 24)
    _paint_zone(hazard, 14, 30, 24, 35)
    return DungeonFloor(
        index=9,
        name="Inferno Ascent",
        theme="inferno",
        terrain=terrain,
        slow_tiles=slow,
        hazard_tiles=hazard,
        heal_tiles=heal,
        gate_position=(32, 32),
        start_positions=_start_positions(),
        chests=(
            ChestSpawn((5, 18), (PowerUpType.DAMAGE, PowerUpType.RANGE)),
            ChestSpawn((20, 32), (PowerUpType.VITALITY, PowerUpType.SPEED)),
        ),
        enemies=(
            EnemySpawn((20, 18), EnemyKind.SKIRMISHER),
            EnemySpawn((19, 30), EnemyKind.ARCHER),
            EnemySpawn((31, 31), EnemyKind.SKIRMISHER),
        ),
    )


def _abyss_throne(grid_size: int) -> DungeonFloor:
    terrain, slow, hazard, heal = _blank_layers(grid_size)
    _carve_room(terrain, 1, 1, 7, 8)
    _carve_hallway(terrain, 4, 7, 18)
    _carve_room(terrain, 1, 17, 10, 27)
    _carve_hallway(terrain, 8, 21, 30)
    _carve_room(terrain, 5, 29, 13, 35)
    _carve_vertical(terrain, 32, 12, 18)
    _carve_room(terrain, 17, 22, 35, 35)
    _paint_zone(hazard, 17, 22, 35, 35)
    _paint_zone(slow, 5, 29, 13, 35)
    return DungeonFloor(
        index=10,
        name="Abyss Throne",
        theme="abyss",
        terrain=terrain,
        slow_tiles=slow,
        hazard_tiles=hazard,
        heal_tiles=heal,
        gate_position=(31, 31),
        start_positions=_start_positions(),
        chests=(
            ChestSpawn((8, 23), (PowerUpType.SPEED, PowerUpType.VITALITY)),
            ChestSpawn((8, 32), (PowerUpType.RANGE, PowerUpType.DAMAGE)),
        ),
        enemies=(
            EnemySpawn((10, 31), EnemyKind.ARCHER),
            EnemySpawn((25, 27), EnemyKind.FINAL_BOSS),
            EnemySpawn((29, 26), EnemyKind.SKIRMISHER),
            EnemySpawn((29, 31), EnemyKind.ARCHER),
        ),
        gate_locked_until_boss=True,
    )
