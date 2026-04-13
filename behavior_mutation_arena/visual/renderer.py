from __future__ import annotations

import time

import pygame

from behavior_mutation_arena.config import ArenaConfig, ItemKind, WeaponKind


CELL_BACKGROUND = (24, 28, 34)
GRID_LINE = (58, 66, 78)
TEXT_COLOR = (235, 239, 245)
WALL_COLOR = (64, 72, 86)
ITEM_COLORS = {
    ItemKind.FOOD: (60, 179, 113),
    ItemKind.POISON: (205, 92, 92),
    ItemKind.MELEE: (225, 169, 95),
    ItemKind.RANGED: (86, 156, 214),
    ItemKind.RARE_MELEE: (255, 140, 0),
    ItemKind.RARE_RANGED: (148, 0, 211),
}
AGENT_COLORS = {
    WeaponKind.NONE: (230, 230, 230),
    WeaponKind.MELEE: (244, 196, 48),
    WeaponKind.RANGED: (90, 170, 255),
    WeaponKind.RARE_MELEE: (255, 125, 50),
    WeaponKind.RARE_RANGED: (182, 95, 255),
}


class ArenaRenderer:
    def __init__(self, config: ArenaConfig) -> None:
        self.config = config
        pygame.init()
        pygame.display.set_caption("Behavior Mutation Arena")
        self.cell_size = config.render_cell_size
        self.grid_pixels = config.grid_size * self.cell_size
        self.sidebar_width = 300
        self.screen = pygame.display.set_mode((self.grid_pixels + self.sidebar_width, self.grid_pixels))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)
        self.small_font = pygame.font.SysFont("consolas", 14)

    def draw(self, snapshot: dict[str, object], overlay_lines: list[str] | None = None) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit

        self.screen.fill((14, 18, 22))
        item_grid = snapshot["item_grid"]
        terrain_grid = snapshot.get("terrain_grid")
        positions = snapshot["positions"]
        alive = snapshot["alive"]
        health = snapshot["health"]
        energy = snapshot["energy"]
        weapon_kind = snapshot["weapon_kind"]
        kills = snapshot["kills"]
        rewards = snapshot["reward"]
        explored = snapshot.get("explored_cells")
        camping = snapshot.get("camping_steps")

        for x in range(self.config.grid_size):
            for y in range(self.config.grid_size):
                rect = pygame.Rect(
                    y * self.cell_size,
                    x * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                cell_color = WALL_COLOR if terrain_grid is not None and terrain_grid[x, y] else CELL_BACKGROUND
                pygame.draw.rect(self.screen, cell_color, rect)
                pygame.draw.rect(self.screen, GRID_LINE, rect, width=1)
                item = ItemKind(int(item_grid[x, y]))
                if item is not ItemKind.EMPTY:
                    inset = rect.inflate(-self.cell_size * 0.45, -self.cell_size * 0.45)
                    pygame.draw.rect(self.screen, ITEM_COLORS[item], inset, border_radius=4)

        for agent_id in range(self.config.population_size):
            if not alive[agent_id]:
                continue
            x, y = positions[agent_id]
            center = (
                int(y * self.cell_size + self.cell_size / 2),
                int(x * self.cell_size + self.cell_size / 2),
            )
            color = AGENT_COLORS[WeaponKind(int(weapon_kind[agent_id]))]
            pygame.draw.circle(self.screen, color, center, self.cell_size // 3)
            label = self.small_font.render(str(agent_id), True, (14, 18, 22))
            self.screen.blit(label, label.get_rect(center=center))
            health_ratio = max(0.0, min(1.0, float(health[agent_id]) / self.config.max_health))
            energy_ratio = max(0.0, min(1.0, float(energy[agent_id]) / self.config.max_energy))
            bar_x = y * self.cell_size + 4
            health_bar = pygame.Rect(bar_x, x * self.cell_size + 4, int((self.cell_size - 8) * health_ratio), 4)
            energy_bar = pygame.Rect(bar_x, x * self.cell_size + 10, int((self.cell_size - 8) * energy_ratio), 4)
            pygame.draw.rect(self.screen, (45, 45, 45), pygame.Rect(bar_x, x * self.cell_size + 4, self.cell_size - 8, 4))
            pygame.draw.rect(self.screen, (45, 45, 45), pygame.Rect(bar_x, x * self.cell_size + 10, self.cell_size - 8, 4))
            pygame.draw.rect(self.screen, (80, 220, 120), health_bar)
            pygame.draw.rect(self.screen, (90, 170, 255), energy_bar)

        sidebar_x = self.grid_pixels + 20
        header = self.font.render("Arena State", True, TEXT_COLOR)
        self.screen.blit(header, (sidebar_x, 20))
        lines = [
            f"Step: {snapshot['step']}/{snapshot['episode_step_limit']}",
            f"Alive: {int(sum(1 for flag in alive if flag))}",
            f"Map: {snapshot.get('map_name', 'unknown')}",
        ]
        if overlay_lines:
            lines.extend(overlay_lines)
        top_agents = sorted(
            [
                (
                    float(rewards[idx]),
                    int(kills[idx]),
                    int(explored[idx]) if explored is not None else 0,
                    -(int(camping[idx]) if camping is not None else 0),
                    idx,
                )
                for idx in range(self.config.population_size)
                if alive[idx]
            ],
            reverse=True,
        )[:8]
        lines.append("")
        lines.append("Top live agents")
        for reward, kill_count, explored_cells, camping_rank, agent_id in top_agents:
            camping_steps = -camping_rank
            lines.append(
                f"A{agent_id:02d}  R {reward:6.1f}  K {kill_count}  E {explored_cells:02d}  C {camping_steps:02d}"
            )

        y_offset = 60
        for line in lines:
            surface = self.small_font.render(line, True, TEXT_COLOR)
            self.screen.blit(surface, (sidebar_x, y_offset))
            y_offset += 22

        pygame.display.flip()
        self.clock.tick(self.config.render_fps)

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
