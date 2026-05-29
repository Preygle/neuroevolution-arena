from __future__ import annotations

import math
import time

import pygame

from behavior_mutation_arena.config import ArenaConfig, EnemyKind, PowerUpType
from behavior_mutation_arena.core.dungeon_floors import build_dungeon_floors, build_stitched_dungeon_layout


CELL_BACKGROUND = (20, 24, 30)
GRID_LINE = (50, 58, 68)
TEXT_COLOR = (235, 239, 245)
WALL_COLOR = (70, 78, 94)
SLOW_TILE_COLOR = (115, 155, 210)
HAZARD_TILE_COLOR = (180, 78, 58)
HEAL_TILE_COLOR = (70, 140, 92)
GATE_OPEN_COLOR = (255, 215, 90)
GATE_LOCKED_COLOR = (134, 101, 165)
AGENT_COLOR = (230, 230, 230)
CHEST_COLORS = {
    PowerUpType.DAMAGE: (240, 126, 76),
    PowerUpType.RANGE: (93, 177, 255),
    PowerUpType.SPEED: (255, 214, 92),
    PowerUpType.DIAGONAL: (186, 107, 255),
    PowerUpType.VITALITY: (90, 204, 136),
}
ENEMY_COLORS = {
    EnemyKind.SKIRMISHER: (217, 93, 93),
    EnemyKind.ARCHER: (230, 158, 76),
    EnemyKind.MINI_BOSS: (165, 87, 219),
    EnemyKind.FINAL_BOSS: (255, 66, 66),
}


class ArenaRenderer:
    def __init__(self, config: ArenaConfig, instance_count: int = 1) -> None:
        self.config = config
        self.instance_count = max(1, instance_count)
        self.visible_instance_count = min(self.instance_count, config.max_render_instance_count)
        self.columns = max(1, config.render_columns)
        self.header_height = 54
        self.tile_header_height = 56
        self.padding = 10
        self.minimized = False
        self.floors = build_dungeon_floors(config.grid_size)
        self.stitched_layout = build_stitched_dungeon_layout(self.floors)
        pygame.init()
        pygame.display.set_caption("Dungeon Crawler Training")
        desktop_sizes = pygame.display.get_desktop_sizes()
        self.desktop_size = desktop_sizes[0] if desktop_sizes else (config.render_max_window_width, config.render_max_window_height)
        self.screen = pygame.display.set_mode(self.desktop_size, pygame.RESIZABLE)
        self.font = pygame.font.SysFont("consolas", 18)
        self.small_font = pygame.font.SysFont("consolas", 14)
        self.tiny_font = pygame.font.SysFont("consolas", 12)
        self.last_present_time = 0.0

    def page_status(self, total_instances: int) -> tuple[int, int]:
        page_size = self._page_size()
        total_pages = max(1, math.ceil(max(1, total_instances) / page_size))
        page_index = 0
        if total_pages > 1:
            page_index = int(time.time() / self.config.render_page_seconds) % total_pages
        return page_index + 1, total_pages

    def visible_instance_indices(self, total_instances: int) -> list[int]:
        page_size = self._page_size()
        total_pages = max(1, math.ceil(max(1, total_instances) / page_size))
        page_index = 0
        if total_pages > 1:
            page_index = int(time.time() / self.config.render_page_seconds) % total_pages
        start = page_index * page_size
        end = min(total_instances, start + page_size)
        return list(range(start, end))

    def draw(self, snapshot: dict[str, object], overlay_lines: list[str] | None = None) -> None:
        self.draw_many([snapshot], [overlay_lines or []], header_lines=[])

    def draw_many(
        self,
        snapshots: list[dict[str, object]],
        overlay_groups: list[list[str]] | None = None,
        header_lines: list[str] | None = None,
    ) -> None:
        self._poll_events()
        if self.minimized:
            return
        now = time.perf_counter()
        if now - self.last_present_time < 1.0 / max(1, self.config.render_fps):
            return
        self.screen.fill((14, 18, 24))
        overlay_groups = overlay_groups or [[] for _ in snapshots]
        layout = self._layout(snapshots)
        self._draw_header(header_lines or [], layout["screen_width"])

        for index, snapshot in enumerate(snapshots):
            row = index // layout["columns"]
            column = index % layout["columns"]
            tile_x = self.padding + column * (layout["tile_width"] + self.padding)
            tile_y = self.header_height + self.padding + row * (layout["tile_height"] + self.padding)
            frame_rect = pygame.Rect(tile_x - 3, tile_y - 3, layout["tile_width"] + 6, layout["tile_height"] + 6)
            pygame.draw.rect(self.screen, (24, 29, 37), frame_rect, border_radius=8)
            self._draw_tile_header(tile_x, tile_y, layout["tile_width"], overlay_groups[index])
            self._draw_snapshot_grid(
                snapshot,
                tile_x,
                tile_y + self.tile_header_height,
                layout["cell_size"],
            )

        pygame.display.flip()
        self.last_present_time = now

    def _draw_header(self, lines: list[str], width: int) -> None:
        header_rect = pygame.Rect(0, 0, width, self.header_height)
        pygame.draw.rect(self.screen, (18, 22, 28), header_rect)
        y_offset = 8
        for line in lines[:3]:
            surface = self.small_font.render(line, True, TEXT_COLOR)
            self.screen.blit(surface, (self.padding, y_offset))
            y_offset += 16

    def _draw_tile_header(self, origin_x: int, origin_y: int, width: int, lines: list[str]) -> None:
        header_rect = pygame.Rect(origin_x, origin_y, width, self.tile_header_height)
        pygame.draw.rect(self.screen, (18, 22, 28), header_rect, border_radius=6)
        y_offset = origin_y + 6
        for line in lines[:4]:
            surface = self.tiny_font.render(line, True, TEXT_COLOR)
            self.screen.blit(surface, (origin_x + 8, y_offset))
            y_offset += 12

    def _layout(self, snapshots: list[dict[str, object]]) -> dict[str, int]:
        screen_width, screen_height = self.screen.get_size()
        snapshot_count = len(snapshots)
        columns = 1 if snapshot_count <= 1 else min(self.columns, snapshot_count)
        grid_rows, grid_columns = self._grid_dimensions(snapshots)
        min_cell_size = 1 if self.config.render_stitched_dungeon else self.config.render_min_cell_size
        min_tile_height = grid_rows * min_cell_size + self.tile_header_height
        rows_fit = max(
            1,
            (screen_height - self.header_height - self.padding * 2) // max(1, min_tile_height + self.padding),
        )
        rows = max(1, min(rows_fit, math.ceil(snapshot_count / columns)))
        available_width = screen_width - self.padding * (columns + 1)
        available_height = screen_height - self.header_height - self.padding * (rows + 1) - rows * self.tile_header_height
        cell_size = max(
            min_cell_size,
            min(
                self.config.render_cell_size,
                available_width // max(1, columns * grid_columns),
                available_height // max(1, rows * grid_rows),
            ),
        )
        tile_width = grid_columns * cell_size
        tile_height = grid_rows * cell_size + self.tile_header_height
        return {
            "screen_width": screen_width,
            "columns": columns,
            "rows": rows,
            "cell_size": cell_size,
            "tile_width": tile_width,
            "tile_height": tile_height,
        }

    def _draw_snapshot_grid(self, snapshot: dict[str, object], origin_x: int, origin_y: int, cell_size: int) -> None:
        if self.config.render_stitched_dungeon:
            self._draw_stitched_snapshot_grid(snapshot, origin_x, origin_y, cell_size)
            return
        self._draw_floor_snapshot_grid(snapshot, origin_x, origin_y, cell_size)

    def _draw_floor_snapshot_grid(
        self,
        snapshot: dict[str, object],
        origin_x: int,
        origin_y: int,
        cell_size: int,
    ) -> None:
        terrain = snapshot["terrain_grid"]
        slow_tiles = snapshot["slow_tiles"]
        hazard_tiles = snapshot["hazard_tiles"]
        heal_tiles = snapshot["heal_tiles"]
        chest_grid = snapshot["chest_grid"]
        positions = snapshot["positions"]
        alive = snapshot["alive"]
        health = snapshot["health"]
        energy = snapshot["energy"]
        enemy_positions = snapshot["enemy_positions"]
        enemy_kind = snapshot["enemy_kind"]
        enemy_health = snapshot["enemy_health"]
        gate_position = tuple(int(value) for value in snapshot["gate_position"])
        gate_open = bool(snapshot["gate_open"])
        effective_max_health = float(snapshot["effective_max_health"])
        effective_max_health_by_agent = snapshot.get("effective_max_health_by_agent")
        draw_grid_lines = cell_size >= 8

        for x in range(self.config.grid_size):
            for y in range(self.config.grid_size):
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
                pygame.draw.rect(self.screen, color, rect)
                if draw_grid_lines:
                    pygame.draw.rect(self.screen, GRID_LINE, rect, width=1)

                if (x, y) == gate_position:
                    inset = rect.inflate(-cell_size * 0.28, -cell_size * 0.28)
                    pygame.draw.rect(
                        self.screen,
                        GATE_OPEN_COLOR if gate_open else GATE_LOCKED_COLOR,
                        inset,
                        border_radius=max(2, cell_size // 4),
                    )

                chest_value = int(chest_grid[x, y])
                if chest_value > 0:
                    chest_type = PowerUpType(chest_value - 1)
                    inset = rect.inflate(-cell_size * 0.42, -cell_size * 0.42)
                    pygame.draw.rect(
                        self.screen,
                        CHEST_COLORS[chest_type],
                        inset,
                        border_radius=max(2, cell_size // 4),
                    )

        if len(enemy_positions) > 0:
            for (x, y), kind_value, enemy_hp in zip(enemy_positions, enemy_kind, enemy_health):
                center = (
                    int(origin_x + y * cell_size + cell_size / 2),
                    int(origin_y + x * cell_size + cell_size / 2),
                )
                kind = EnemyKind(int(kind_value))
                radius = max(2, cell_size // 3 if kind in {EnemyKind.SKIRMISHER, EnemyKind.ARCHER} else cell_size // 2)
                pygame.draw.circle(self.screen, ENEMY_COLORS[kind], center, radius)
                if cell_size >= 10:
                    hp_surface = self.tiny_font.render(str(int(enemy_hp)), True, (14, 18, 24))
                    self.screen.blit(hp_surface, hp_surface.get_rect(center=center))

        for agent_id in range(self.config.population_size):
            if not alive[agent_id]:
                continue
            x, y = positions[agent_id]
            center = (
                int(origin_x + y * cell_size + cell_size / 2),
                int(origin_y + x * cell_size + cell_size / 2),
            )
            pygame.draw.circle(self.screen, AGENT_COLOR, center, max(2, cell_size // 3))
            if cell_size >= 10:
                label = self.tiny_font.render(str(agent_id), True, (14, 18, 24))
                self.screen.blit(label, label.get_rect(center=center))
            max_health = effective_max_health
            if effective_max_health_by_agent is not None:
                max_health = float(effective_max_health_by_agent[agent_id])
            health_ratio = max(0.0, min(1.0, float(health[agent_id]) / max_health))
            energy_ratio = max(0.0, min(1.0, float(energy[agent_id]) / self.config.max_energy))
            bar_x = origin_x + y * cell_size + 1
            width = max(1, cell_size - 2)
            pygame.draw.rect(self.screen, (40, 40, 40), pygame.Rect(bar_x, origin_y + x * cell_size + 1, width, 2))
            pygame.draw.rect(self.screen, (40, 40, 40), pygame.Rect(bar_x, origin_y + x * cell_size + 4, width, 2))
            pygame.draw.rect(self.screen, (80, 220, 120), pygame.Rect(bar_x, origin_y + x * cell_size + 1, int(width * health_ratio), 2))
            pygame.draw.rect(self.screen, (80, 160, 255), pygame.Rect(bar_x, origin_y + x * cell_size + 4, int(width * energy_ratio), 2))

    def _draw_stitched_snapshot_grid(
        self,
        snapshot: dict[str, object],
        origin_x: int,
        origin_y: int,
        cell_size: int,
    ) -> None:
        current_floor_index = int(snapshot.get("floor_index", 1)) - 1
        view_top, view_left, view_rows, view_columns = self._stitched_viewport(snapshot)
        opened_chests = {
            (int(floor_index), tuple(position))
            for floor_index, position in snapshot.get("opened_chest_positions", ())
        }
        draw_grid_lines = cell_size >= 5

        for floor_index, floor in enumerate(self.floors):
            if not self._floor_intersects_viewport(floor_index, view_top, view_left, view_rows, view_columns):
                continue
            offset_x, offset_y = self.stitched_layout.floor_offset(floor_index)
            brightness = self._floor_brightness(floor_index, current_floor_index)
            for x in range(self.config.grid_size):
                for y in range(self.config.grid_size):
                    global_x = offset_x + x
                    global_y = offset_y + y
                    if not self._in_viewport(global_x, global_y, view_top, view_left, view_rows, view_columns):
                        continue
                    rect = pygame.Rect(
                        origin_x + (global_y - view_left) * cell_size,
                        origin_y + (global_x - view_top) * cell_size,
                        cell_size,
                        cell_size,
                    )
                    color = CELL_BACKGROUND
                    if floor.terrain[x, y]:
                        color = WALL_COLOR
                    elif floor.hazard_tiles[x, y]:
                        color = HAZARD_TILE_COLOR
                    elif floor.heal_tiles[x, y]:
                        color = HEAL_TILE_COLOR
                    elif floor.slow_tiles[x, y]:
                        color = SLOW_TILE_COLOR
                    pygame.draw.rect(self.screen, self._scale_color(color, brightness), rect)
                    if draw_grid_lines:
                        pygame.draw.rect(self.screen, self._scale_color(GRID_LINE, brightness), rect, width=1)

            bounds = pygame.Rect(
                origin_x + (offset_y - view_left) * cell_size,
                origin_y + (offset_x - view_top) * cell_size,
                self.config.grid_size * cell_size,
                self.config.grid_size * cell_size,
            )
            pygame.draw.rect(
                self.screen,
                (225, 232, 242) if floor_index == current_floor_index else (78, 90, 108),
                bounds,
                width=max(1, min(3, cell_size)),
            )
            self._draw_floor_label(floor_index, floor.name, bounds, cell_size)
            self._draw_stitched_gate(
                snapshot,
                floor_index,
                current_floor_index,
                origin_x,
                origin_y,
                cell_size,
                view_top,
                view_left,
                view_rows,
                view_columns,
            )
            self._draw_stitched_chests(
                snapshot,
                opened_chests,
                floor_index,
                current_floor_index,
                origin_x,
                origin_y,
                cell_size,
                view_top,
                view_left,
                view_rows,
                view_columns,
            )
            self._draw_stitched_enemies(
                snapshot,
                floor_index,
                current_floor_index,
                origin_x,
                origin_y,
                cell_size,
                view_top,
                view_left,
                view_rows,
                view_columns,
            )

        self._draw_stitched_agents(
            snapshot,
            current_floor_index,
            origin_x,
            origin_y,
            cell_size,
            view_top,
            view_left,
            view_rows,
            view_columns,
        )

    def _draw_stitched_gate(
        self,
        snapshot: dict[str, object],
        floor_index: int,
        current_floor_index: int,
        origin_x: int,
        origin_y: int,
        cell_size: int,
        view_top: int,
        view_left: int,
        view_rows: int,
        view_columns: int,
    ) -> None:
        floor = self.floors[floor_index]
        gate_position = floor.gate_position
        gate_open = not floor.gate_locked_until_boss
        if floor_index == current_floor_index:
            gate_position = tuple(int(value) for value in snapshot["gate_position"])
            gate_open = bool(snapshot["gate_open"])
        gx, gy = self.stitched_layout.to_global(floor_index, gate_position)
        if not self._in_viewport(gx, gy, view_top, view_left, view_rows, view_columns):
            return
        rect = pygame.Rect(origin_x + (gy - view_left) * cell_size, origin_y + (gx - view_top) * cell_size, cell_size, cell_size)
        inset = rect.inflate(-cell_size * 0.2, -cell_size * 0.2)
        pygame.draw.rect(
            self.screen,
            GATE_OPEN_COLOR if gate_open else GATE_LOCKED_COLOR,
            inset,
            border_radius=max(1, cell_size // 4),
        )

    def _draw_stitched_chests(
        self,
        snapshot: dict[str, object],
        opened_chests: set[tuple[int, tuple[int, int]]],
        floor_index: int,
        current_floor_index: int,
        origin_x: int,
        origin_y: int,
        cell_size: int,
        view_top: int,
        view_left: int,
        view_rows: int,
        view_columns: int,
    ) -> None:
        floor = self.floors[floor_index]
        if floor_index == current_floor_index:
            chest_grid = snapshot["chest_grid"]
            chests = [
                (x, y, PowerUpType(int(chest_grid[x, y]) - 1))
                for x in range(self.config.grid_size)
                for y in range(self.config.grid_size)
                if int(chest_grid[x, y]) > 0
            ]
        else:
            chests = [
                (chest.position[0], chest.position[1], chest.powerup)
                for chest in floor.chests
                if (floor_index, chest.position) not in opened_chests
            ]
        for x, y, powerup in chests:
            gx, gy = self.stitched_layout.to_global(floor_index, (x, y))
            if not self._in_viewport(gx, gy, view_top, view_left, view_rows, view_columns):
                continue
            rect = pygame.Rect(origin_x + (gy - view_left) * cell_size, origin_y + (gx - view_top) * cell_size, cell_size, cell_size)
            inset = rect.inflate(-cell_size * 0.42, -cell_size * 0.42)
            pygame.draw.rect(self.screen, CHEST_COLORS[powerup], inset, border_radius=max(1, cell_size // 4))

    def _draw_stitched_enemies(
        self,
        snapshot: dict[str, object],
        floor_index: int,
        current_floor_index: int,
        origin_x: int,
        origin_y: int,
        cell_size: int,
        view_top: int,
        view_left: int,
        view_rows: int,
        view_columns: int,
    ) -> None:
        if floor_index == current_floor_index:
            enemy_positions = snapshot["enemy_positions"]
            enemy_kind = snapshot["enemy_kind"]
            enemy_health = snapshot["enemy_health"]
            enemies = [
                (tuple(int(value) for value in position), EnemyKind(int(kind_value)), float(enemy_hp))
                for position, kind_value, enemy_hp in zip(enemy_positions, enemy_kind, enemy_health)
            ]
        else:
            enemies = [(enemy.position, enemy.kind, 0.0) for enemy in self.floors[floor_index].enemies]
        for (x, y), kind, enemy_hp in enemies:
            gx, gy = self.stitched_layout.to_global(floor_index, (x, y))
            if not self._in_viewport(gx, gy, view_top, view_left, view_rows, view_columns):
                continue
            center = (
                int(origin_x + (gy - view_left) * cell_size + cell_size / 2),
                int(origin_y + (gx - view_top) * cell_size + cell_size / 2),
            )
            radius = max(1, cell_size // 3 if kind in {EnemyKind.SKIRMISHER, EnemyKind.ARCHER} else cell_size // 2)
            pygame.draw.circle(self.screen, ENEMY_COLORS[kind], center, radius)
            if enemy_hp > 0 and cell_size >= 9:
                hp_surface = self.tiny_font.render(str(int(enemy_hp)), True, (14, 18, 24))
                self.screen.blit(hp_surface, hp_surface.get_rect(center=center))

    def _draw_stitched_agents(
        self,
        snapshot: dict[str, object],
        current_floor_index: int,
        origin_x: int,
        origin_y: int,
        cell_size: int,
        view_top: int,
        view_left: int,
        view_rows: int,
        view_columns: int,
    ) -> None:
        positions = snapshot["positions"]
        alive = snapshot["alive"]
        health = snapshot["health"]
        energy = snapshot["energy"]
        effective_max_health = float(snapshot["effective_max_health"])
        effective_max_health_by_agent = snapshot.get("effective_max_health_by_agent")

        for agent_id in range(self.config.population_size):
            if not alive[agent_id]:
                continue
            x, y = positions[agent_id]
            gx, gy = self.stitched_layout.to_global(current_floor_index, (int(x), int(y)))
            if not self._in_viewport(gx, gy, view_top, view_left, view_rows, view_columns):
                continue
            center = (
                int(origin_x + (gy - view_left) * cell_size + cell_size / 2),
                int(origin_y + (gx - view_top) * cell_size + cell_size / 2),
            )
            pygame.draw.circle(self.screen, AGENT_COLOR, center, max(2, cell_size // 3))
            if cell_size >= 9:
                label = self.tiny_font.render(str(agent_id), True, (14, 18, 24))
                self.screen.blit(label, label.get_rect(center=center))
            max_health = effective_max_health
            if effective_max_health_by_agent is not None:
                max_health = float(effective_max_health_by_agent[agent_id])
            health_ratio = max(0.0, min(1.0, float(health[agent_id]) / max_health))
            energy_ratio = max(0.0, min(1.0, float(energy[agent_id]) / self.config.max_energy))
            bar_x = origin_x + (gy - view_left) * cell_size + 1
            width = max(1, cell_size - 2)
            pygame.draw.rect(self.screen, (40, 40, 40), pygame.Rect(bar_x, origin_y + (gx - view_top) * cell_size + 1, width, 2))
            pygame.draw.rect(self.screen, (40, 40, 40), pygame.Rect(bar_x, origin_y + (gx - view_top) * cell_size + 4, width, 2))
            pygame.draw.rect(
                self.screen,
                (80, 220, 120),
                pygame.Rect(bar_x, origin_y + (gx - view_top) * cell_size + 1, int(width * health_ratio), 2),
            )
            pygame.draw.rect(
                self.screen,
                (80, 160, 255),
                pygame.Rect(bar_x, origin_y + (gx - view_top) * cell_size + 4, int(width * energy_ratio), 2),
            )

    def _draw_floor_label(self, floor_index: int, floor_name: str, bounds: pygame.Rect, cell_size: int) -> None:
        if cell_size < 2:
            return
        label = f"{floor_index + 1:02d}"
        if cell_size >= 4:
            label = f"{floor_index + 1:02d} {floor_name}"
        surface = self.tiny_font.render(label, True, TEXT_COLOR)
        background = surface.get_rect(topleft=(bounds.x + 3, bounds.y + 3))
        background.inflate_ip(4, 2)
        pygame.draw.rect(self.screen, (14, 18, 24), background, border_radius=2)
        self.screen.blit(surface, (background.x + 2, background.y + 1))

    def _grid_dimensions(self, snapshots: list[dict[str, object]] | None = None) -> tuple[int, int]:
        if self.config.render_stitched_dungeon:
            if self.config.render_stitched_focus_current and snapshots:
                dimensions = [self._stitched_viewport(snapshot)[2:] for snapshot in snapshots]
                return (
                    max(rows for rows, _ in dimensions),
                    max(columns for _, columns in dimensions),
                )
            if self.config.render_stitched_focus_current:
                context = max(0, int(self.config.render_stitched_context_floors))
                count = min(len(self.floors), context * 2 + 1)
                return self._stitched_subset_dimensions(0, count - 1)
            return self.stitched_layout.global_shape
        return self.config.grid_size, self.config.grid_size

    def _stitched_viewport(self, snapshot: dict[str, object]) -> tuple[int, int, int, int]:
        if not self.config.render_stitched_focus_current:
            rows, columns = self.stitched_layout.global_shape
            return 0, 0, rows, columns

        current_floor_index = max(0, min(len(self.floors) - 1, int(snapshot.get("floor_index", 1)) - 1))
        context = max(0, int(self.config.render_stitched_context_floors))
        start = max(0, current_floor_index - context)
        end = min(len(self.floors) - 1, current_floor_index + context)
        top, left, bottom, right = self._stitched_subset_bounds(start, end)
        padding = 2
        global_rows, global_columns = self.stitched_layout.global_shape
        top = max(0, top - padding)
        left = max(0, left - padding)
        bottom = min(global_rows, bottom + padding)
        right = min(global_columns, right + padding)
        return top, left, bottom - top, right - left

    def _stitched_subset_dimensions(self, start: int, end: int) -> tuple[int, int]:
        top, left, bottom, right = self._stitched_subset_bounds(start, end)
        return bottom - top, right - left

    def _stitched_subset_bounds(self, start: int, end: int) -> tuple[int, int, int, int]:
        offsets = [self.stitched_layout.floor_offset(index) for index in range(start, end + 1)]
        top = min(offset[0] for offset in offsets)
        left = min(offset[1] for offset in offsets)
        bottom = max(offset[0] + self.config.grid_size for offset in offsets)
        right = max(offset[1] + self.config.grid_size for offset in offsets)
        return top, left, bottom, right

    def _floor_intersects_viewport(
        self,
        floor_index: int,
        view_top: int,
        view_left: int,
        view_rows: int,
        view_columns: int,
    ) -> bool:
        offset_x, offset_y = self.stitched_layout.floor_offset(floor_index)
        return not (
            offset_x + self.config.grid_size <= view_top
            or offset_x >= view_top + view_rows
            or offset_y + self.config.grid_size <= view_left
            or offset_y >= view_left + view_columns
        )

    def _in_viewport(
        self,
        x: int,
        y: int,
        view_top: int,
        view_left: int,
        view_rows: int,
        view_columns: int,
    ) -> bool:
        return view_top <= x < view_top + view_rows and view_left <= y < view_left + view_columns

    def _floor_brightness(self, floor_index: int, current_floor_index: int) -> float:
        if floor_index == current_floor_index:
            return 1.0
        if floor_index < current_floor_index:
            return 0.72
        return 0.48

    def _scale_color(self, color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
        return tuple(max(0, min(255, int(channel * factor))) for channel in color)

    def _page_size(self) -> int:
        screen_width, screen_height = self.screen.get_size()
        grid_rows, _ = self._grid_dimensions()
        min_cell_size = 1 if self.config.render_stitched_dungeon else self.config.render_min_cell_size
        rows_fit = max(
            1,
            (screen_height - self.header_height - self.padding * 2)
            // max(1, grid_rows * min_cell_size + self.tile_header_height + self.padding),
        )
        return max(1, self.columns * rows_fit)

    def _poll_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit
            if event.type == pygame.WINDOWMINIMIZED:
                self.minimized = True
            elif event.type == pygame.WINDOWRESTORED:
                self.minimized = False
            elif event.type == pygame.WINDOWSIZECHANGED:
                self.minimized = False

    def play_replay(self, replay_record: object, delay: float = 0.08) -> None:
        for frame in replay_record.frames:
            self.draw(
                frame,
                [
                    f"Replay generation {replay_record.generation}",
                    f"Champion {replay_record.champion_id}",
                    f"Fitness {replay_record.fitness:.2f}",
                ],
            )
            time.sleep(delay)

    def close(self) -> None:
        pygame.quit()
