from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame
from PIL import Image

from behavior_mutation_arena.config import ArenaConfig, EnemyKind, PowerUpType
from behavior_mutation_arena.core.dungeon_floors import (
    StitchedDungeonLayout,
    build_dungeon_floors,
    build_stitched_dungeon_layout,
)
from behavior_mutation_arena.core.replay import ReplayRecord
from behavior_mutation_arena.visual.renderer import (
    AGENT_COLOR,
    CELL_BACKGROUND,
    CHEST_COLORS,
    ENEMY_COLORS,
    GATE_LOCKED_COLOR,
    GATE_OPEN_COLOR,
    GRID_LINE,
    HAZARD_TILE_COLOR,
    HEAL_TILE_COLOR,
    SLOW_TILE_COLOR,
    TEXT_COLOR,
    WALL_COLOR,
)


def render_replay_gif(
    replay: ReplayRecord,
    output_path: str | Path,
    config: ArenaConfig,
    fps: int = 12,
    max_frames: int = 360,
    cell_size: int = 10,
) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not replay.frames:
        raise ValueError("replay has no frames to render")

    pygame.font.init()
    floors = build_dungeon_floors(config.grid_size)
    layout = build_stitched_dungeon_layout(floors)
    cell = max(1, min(3, int(cell_size))) if config.render_stitched_dungeon else max(4, int(cell_size))
    width = layout.global_shape[1] * cell if config.render_stitched_dungeon else config.grid_size * cell
    header_height = 92
    grid_rows = layout.global_shape[0] if config.render_stitched_dungeon else config.grid_size
    height = header_height + grid_rows * cell
    font = pygame.font.SysFont("consolas", max(12, min(18, cell + 4)))
    small_font = pygame.font.SysFont("consolas", max(10, min(14, cell + 2)))
    tiny_font = pygame.font.SysFont("consolas", max(8, min(12, cell + 1)))
    frame_indices = _sample_frame_indices(len(replay.frames), max_frames)
    images: list[Image.Image] = []

    for frame_number, replay_index in enumerate(frame_indices):
        snapshot = replay.frames[replay_index]
        surface = pygame.Surface((width, height))
        surface.fill((14, 18, 24))
        _draw_header(surface, snapshot, replay, frame_number, len(frame_indices), font, small_font)
        _draw_grid(surface, snapshot, config, 0, header_height, cell, tiny_font, floors, layout)
        raw = pygame.image.tostring(surface, "RGB")
        images.append(Image.frombytes("RGB", surface.get_size(), raw))

    duration_ms = int(1000 / max(1, fps))
    images[0].save(
        target,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    return target


def _sample_frame_indices(frame_count: int, max_frames: int) -> list[int]:
    if frame_count <= max(1, max_frames):
        return list(range(frame_count))
    stride = max(1, frame_count // max(1, max_frames))
    indices = list(range(0, frame_count, stride))[:max_frames]
    if indices[-1] != frame_count - 1:
        indices[-1] = frame_count - 1
    return indices


def _draw_header(
    surface: pygame.Surface,
    snapshot: dict[str, Any],
    replay: ReplayRecord,
    frame_number: int,
    frame_count: int,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
) -> None:
    pygame.draw.rect(surface, (18, 22, 28), pygame.Rect(0, 0, surface.get_width(), 92))
    lines = [
        f"Best replay | gen {replay.generation} | champion P{replay.champion_id} | fitness {replay.fitness:.1f}",
        f"Frame {frame_number + 1}/{frame_count} | step {snapshot.get('step', 0)}/{snapshot.get('episode_step_limit', 0)}",
        (
            f"Floor {snapshot.get('floor_index', '?')}: {snapshot.get('floor_name', '?')} | "
            f"alive {int(sum(snapshot.get('alive', [])))} | "
            f"chests {snapshot.get('chests_opened', 0)} | powerups {snapshot.get('powerups_picked', 0)}"
        ),
        (
            f"gate {snapshot.get('closest_gate_distance', '?')} | "
            f"best gate {snapshot.get('best_gate_distance', '?')} | "
            f"boss hp {snapshot.get('boss_health_remaining', 0):.1f}"
        ),
    ]
    y = 8
    for index, line in enumerate(lines):
        rendered = (font if index == 0 else small_font).render(line, True, TEXT_COLOR)
        surface.blit(rendered, (10, y))
        y += 22 if index == 0 else 18


def _draw_grid(
    surface: pygame.Surface,
    snapshot: dict[str, Any],
    config: ArenaConfig,
    origin_x: int,
    origin_y: int,
    cell_size: int,
    tiny_font: pygame.font.Font,
    floors: list[Any],
    layout: StitchedDungeonLayout,
) -> None:
    if config.render_stitched_dungeon:
        _draw_stitched_grid(surface, snapshot, config, origin_x, origin_y, cell_size, tiny_font, floors, layout)
        return

    terrain = snapshot["terrain_grid"]
    slow_tiles = snapshot["slow_tiles"]
    hazard_tiles = snapshot["hazard_tiles"]
    heal_tiles = snapshot["heal_tiles"]
    chest_grid = snapshot["chest_grid"]
    gate_position = tuple(int(value) for value in snapshot["gate_position"])
    gate_open = bool(snapshot["gate_open"])
    draw_grid_lines = cell_size >= 8

    for x in range(config.grid_size):
        for y in range(config.grid_size):
            rect = pygame.Rect(origin_x + y * cell_size, origin_y + x * cell_size, cell_size, cell_size)
            color = CELL_BACKGROUND
            if terrain[x, y]:
                color = WALL_COLOR
            elif hazard_tiles[x, y]:
                color = HAZARD_TILE_COLOR
            elif heal_tiles[x, y]:
                color = HEAL_TILE_COLOR
            elif slow_tiles[x, y]:
                color = SLOW_TILE_COLOR
            pygame.draw.rect(surface, color, rect)
            if draw_grid_lines:
                pygame.draw.rect(surface, GRID_LINE, rect, width=1)
            if (x, y) == gate_position:
                inset = rect.inflate(-cell_size * 0.28, -cell_size * 0.28)
                pygame.draw.rect(
                    surface,
                    GATE_OPEN_COLOR if gate_open else GATE_LOCKED_COLOR,
                    inset,
                    border_radius=max(2, cell_size // 4),
                )
            chest_value = int(chest_grid[x, y])
            if chest_value > 0:
                chest_type = PowerUpType(chest_value - 1)
                inset = rect.inflate(-cell_size * 0.42, -cell_size * 0.42)
                pygame.draw.rect(surface, CHEST_COLORS[chest_type], inset, border_radius=max(2, cell_size // 4))

    _draw_enemies(surface, snapshot, origin_x, origin_y, cell_size, tiny_font)
    _draw_agents(surface, snapshot, config, origin_x, origin_y, cell_size, tiny_font)


def _draw_stitched_grid(
    surface: pygame.Surface,
    snapshot: dict[str, Any],
    config: ArenaConfig,
    origin_x: int,
    origin_y: int,
    cell_size: int,
    tiny_font: pygame.font.Font,
    floors: list[Any],
    layout: StitchedDungeonLayout,
) -> None:
    current_floor_index = int(snapshot.get("floor_index", 1)) - 1
    opened_chests = {
        (int(floor_index), tuple(position))
        for floor_index, position in snapshot.get("opened_chest_positions", ())
    }
    for floor_index, floor in enumerate(floors):
        offset_x, offset_y = layout.floor_offset(floor_index)
        brightness = _floor_brightness(floor_index, current_floor_index)
        for x in range(config.grid_size):
            for y in range(config.grid_size):
                rect = pygame.Rect(origin_x + (offset_y + y) * cell_size, origin_y + (offset_x + x) * cell_size, cell_size, cell_size)
                color = CELL_BACKGROUND
                if floor.terrain[x, y]:
                    color = WALL_COLOR
                elif floor.hazard_tiles[x, y]:
                    color = HAZARD_TILE_COLOR
                elif floor.heal_tiles[x, y]:
                    color = HEAL_TILE_COLOR
                elif floor.slow_tiles[x, y]:
                    color = SLOW_TILE_COLOR
                pygame.draw.rect(surface, _scale_color(color, brightness), rect)
        bounds = pygame.Rect(
            origin_x + offset_y * cell_size,
            origin_y + offset_x * cell_size,
            config.grid_size * cell_size,
            config.grid_size * cell_size,
        )
        pygame.draw.rect(surface, (225, 232, 242) if floor_index == current_floor_index else (78, 90, 108), bounds, width=1)
        if cell_size >= 2:
            label = tiny_font.render(f"{floor_index + 1:02d}", True, TEXT_COLOR)
            surface.blit(label, (bounds.x + 2, bounds.y + 2))
        _draw_stitched_gate(surface, snapshot, floors, layout, floor_index, current_floor_index, origin_x, origin_y, cell_size)
        _draw_stitched_chests(surface, snapshot, opened_chests, config, floors, layout, floor_index, current_floor_index, origin_x, origin_y, cell_size)
        _draw_stitched_enemies(surface, snapshot, floors, layout, floor_index, current_floor_index, origin_x, origin_y, cell_size, tiny_font)
    _draw_stitched_agents(surface, snapshot, config, layout, current_floor_index, origin_x, origin_y, cell_size, tiny_font)


def _draw_stitched_gate(
    surface: pygame.Surface,
    snapshot: dict[str, Any],
    floors: list[Any],
    layout: StitchedDungeonLayout,
    floor_index: int,
    current_floor_index: int,
    origin_x: int,
    origin_y: int,
    cell_size: int,
) -> None:
    floor = floors[floor_index]
    gate_position = floor.gate_position
    gate_open = not floor.gate_locked_until_boss
    if floor_index == current_floor_index:
        gate_position = tuple(int(value) for value in snapshot["gate_position"])
        gate_open = bool(snapshot["gate_open"])
    gx, gy = layout.to_global(floor_index, gate_position)
    rect = pygame.Rect(origin_x + gy * cell_size, origin_y + gx * cell_size, cell_size, cell_size)
    pygame.draw.rect(surface, GATE_OPEN_COLOR if gate_open else GATE_LOCKED_COLOR, rect)


def _draw_stitched_chests(
    surface: pygame.Surface,
    snapshot: dict[str, Any],
    opened_chests: set[tuple[int, tuple[int, int]]],
    config: ArenaConfig,
    floors: list[Any],
    layout: StitchedDungeonLayout,
    floor_index: int,
    current_floor_index: int,
    origin_x: int,
    origin_y: int,
    cell_size: int,
) -> None:
    if floor_index == current_floor_index:
        chest_grid = snapshot["chest_grid"]
        chests = [
            (x, y, PowerUpType(int(chest_grid[x, y]) - 1))
            for x in range(config.grid_size)
            for y in range(config.grid_size)
            if int(chest_grid[x, y]) > 0
        ]
    else:
        chests = [
            (chest.position[0], chest.position[1], chest.powerup)
            for chest in floors[floor_index].chests
            if (floor_index, chest.position) not in opened_chests
        ]
    for x, y, powerup in chests:
        gx, gy = layout.to_global(floor_index, (x, y))
        rect = pygame.Rect(origin_x + gy * cell_size, origin_y + gx * cell_size, cell_size, cell_size)
        pygame.draw.rect(surface, CHEST_COLORS[powerup], rect)


def _draw_stitched_enemies(
    surface: pygame.Surface,
    snapshot: dict[str, Any],
    floors: list[Any],
    layout: StitchedDungeonLayout,
    floor_index: int,
    current_floor_index: int,
    origin_x: int,
    origin_y: int,
    cell_size: int,
    tiny_font: pygame.font.Font,
) -> None:
    if floor_index == current_floor_index:
        enemies = [
            (tuple(int(value) for value in position), EnemyKind(int(kind_value)), float(enemy_hp))
            for position, kind_value, enemy_hp in zip(snapshot["enemy_positions"], snapshot["enemy_kind"], snapshot["enemy_health"])
        ]
    else:
        enemies = [(enemy.position, enemy.kind, 0.0) for enemy in floors[floor_index].enemies]
    for (x, y), kind, enemy_hp in enemies:
        gx, gy = layout.to_global(floor_index, (x, y))
        center = (int(origin_x + gy * cell_size + cell_size / 2), int(origin_y + gx * cell_size + cell_size / 2))
        pygame.draw.circle(surface, ENEMY_COLORS[kind], center, max(1, cell_size // 2))
        if enemy_hp > 0 and cell_size >= 9:
            hp_surface = tiny_font.render(str(int(enemy_hp)), True, (14, 18, 24))
            surface.blit(hp_surface, hp_surface.get_rect(center=center))


def _draw_stitched_agents(
    surface: pygame.Surface,
    snapshot: dict[str, Any],
    config: ArenaConfig,
    layout: StitchedDungeonLayout,
    current_floor_index: int,
    origin_x: int,
    origin_y: int,
    cell_size: int,
    tiny_font: pygame.font.Font,
) -> None:
    for agent_id in range(config.population_size):
        if not snapshot["alive"][agent_id]:
            continue
        x, y = snapshot["positions"][agent_id]
        gx, gy = layout.to_global(current_floor_index, (int(x), int(y)))
        center = (int(origin_x + gy * cell_size + cell_size / 2), int(origin_y + gx * cell_size + cell_size / 2))
        pygame.draw.circle(surface, AGENT_COLOR, center, max(2, cell_size // 2))
        if cell_size >= 9:
            label = tiny_font.render(str(agent_id), True, (14, 18, 24))
            surface.blit(label, label.get_rect(center=center))


def _floor_brightness(floor_index: int, current_floor_index: int) -> float:
    if floor_index == current_floor_index:
        return 1.0
    if floor_index < current_floor_index:
        return 0.72
    return 0.48


def _scale_color(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(channel * factor))) for channel in color)


def _draw_enemies(
    surface: pygame.Surface,
    snapshot: dict[str, Any],
    origin_x: int,
    origin_y: int,
    cell_size: int,
    tiny_font: pygame.font.Font,
) -> None:
    enemy_positions = snapshot["enemy_positions"]
    enemy_kind = snapshot["enemy_kind"]
    enemy_health = snapshot["enemy_health"]
    for (x, y), kind_value, enemy_hp in zip(enemy_positions, enemy_kind, enemy_health):
        center = (int(origin_x + y * cell_size + cell_size / 2), int(origin_y + x * cell_size + cell_size / 2))
        kind = EnemyKind(int(kind_value))
        radius = max(2, cell_size // 3 if kind in {EnemyKind.SKIRMISHER, EnemyKind.ARCHER} else cell_size // 2)
        pygame.draw.circle(surface, ENEMY_COLORS[kind], center, radius)
        if cell_size >= 9:
            hp_surface = tiny_font.render(str(int(enemy_hp)), True, (14, 18, 24))
            surface.blit(hp_surface, hp_surface.get_rect(center=center))


def _draw_agents(
    surface: pygame.Surface,
    snapshot: dict[str, Any],
    config: ArenaConfig,
    origin_x: int,
    origin_y: int,
    cell_size: int,
    tiny_font: pygame.font.Font,
) -> None:
    positions = snapshot["positions"]
    alive = snapshot["alive"]
    health = snapshot["health"]
    energy = snapshot["energy"]
    effective_max_health_by_agent = snapshot.get("effective_max_health_by_agent")
    fallback_max_health = float(snapshot.get("effective_max_health", config.max_health))

    for agent_id in range(config.population_size):
        if not alive[agent_id]:
            continue
        x, y = positions[agent_id]
        center = (int(origin_x + y * cell_size + cell_size / 2), int(origin_y + x * cell_size + cell_size / 2))
        pygame.draw.circle(surface, AGENT_COLOR, center, max(2, cell_size // 3))
        if cell_size >= 9:
            label = tiny_font.render(str(agent_id), True, (14, 18, 24))
            surface.blit(label, label.get_rect(center=center))
        max_health = fallback_max_health
        if effective_max_health_by_agent is not None:
            max_health = float(effective_max_health_by_agent[agent_id])
        health_ratio = max(0.0, min(1.0, float(health[agent_id]) / max(1.0, max_health)))
        energy_ratio = max(0.0, min(1.0, float(energy[agent_id]) / max(1.0, config.max_energy)))
        bar_x = origin_x + y * cell_size + 1
        width = max(1, cell_size - 2)
        pygame.draw.rect(surface, (40, 40, 40), pygame.Rect(bar_x, origin_y + x * cell_size + 1, width, 2))
        pygame.draw.rect(surface, (40, 40, 40), pygame.Rect(bar_x, origin_y + x * cell_size + 4, width, 2))
        pygame.draw.rect(
            surface,
            (80, 220, 120),
            pygame.Rect(bar_x, origin_y + x * cell_size + 1, int(width * health_ratio), 2),
        )
        pygame.draw.rect(
            surface,
            (80, 160, 255),
            pygame.Rect(bar_x, origin_y + x * cell_size + 4, int(width * energy_ratio), 2),
        )
