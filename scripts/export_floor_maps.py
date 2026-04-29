from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pygame

sys.path.append(str(Path(__file__).resolve().parents[1]))

from behavior_mutation_arena.config import ArenaConfig, EnemyKind, PowerUpType
from behavior_mutation_arena.core.dungeon_floors import DungeonFloor, build_dungeon_floors
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


BACKGROUND = (12, 15, 20)
PANEL = (18, 23, 31)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export static PNG maps for the dungeon campaign floors.")
    parser.add_argument("--out", type=Path, default=Path("static") / "floors")
    parser.add_argument("--cell-size", type=int, default=18)
    return parser


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def render_floor(floor: DungeonFloor, config: ArenaConfig, cell_size: int) -> pygame.Surface:
    margin = 24
    header_height = 62
    legend_height = 74
    grid_pixels = config.grid_size * cell_size
    width = grid_pixels + margin * 2
    height = header_height + grid_pixels + legend_height + margin
    surface = pygame.Surface((width, height))
    surface.fill(BACKGROUND)
    title_font = pygame.font.SysFont("consolas", 24, bold=True)
    small_font = pygame.font.SysFont("consolas", 14)
    tiny_font = pygame.font.SysFont("consolas", 12, bold=True)

    title = f"Floor {floor.index:02d}: {floor.name}"
    subtitle = f"Theme: {floor.theme} | Gate: {floor.gate_position} | Chests: {len(floor.chests)} | Enemies: {len(floor.enemies)}"
    surface.blit(title_font.render(title, True, TEXT_COLOR), (margin, 14))
    surface.blit(small_font.render(subtitle, True, (178, 190, 205)), (margin, 42))

    origin_x = margin
    origin_y = header_height
    for x in range(config.grid_size):
        for y in range(config.grid_size):
            rect = pygame.Rect(origin_x + y * cell_size, origin_y + x * cell_size, cell_size, cell_size)
            color = CELL_BACKGROUND
            if floor.terrain[x, y]:
                color = WALL_COLOR
            elif floor.hazard_tiles[x, y]:
                color = HAZARD_TILE_COLOR
            elif floor.heal_tiles[x, y]:
                color = HEAL_TILE_COLOR
            elif floor.slow_tiles[x, y]:
                color = SLOW_TILE_COLOR
            pygame.draw.rect(surface, color, rect)
            if cell_size >= 9:
                pygame.draw.rect(surface, GRID_LINE, rect, width=1)

    gate_x, gate_y = floor.gate_position
    gate_rect = pygame.Rect(origin_x + gate_y * cell_size, origin_y + gate_x * cell_size, cell_size, cell_size)
    pygame.draw.rect(
        surface,
        GATE_LOCKED_COLOR if floor.gate_locked_until_boss else GATE_OPEN_COLOR,
        gate_rect.inflate(-cell_size * 0.18, -cell_size * 0.18),
        border_radius=max(2, cell_size // 4),
    )

    for chest in floor.chests:
        x, y = chest.position
        base = pygame.Rect(origin_x + y * cell_size, origin_y + x * cell_size, cell_size, cell_size)
        inset = base.inflate(-cell_size * 0.28, -cell_size * 0.28)
        stripe_width = max(1, inset.width // max(1, len(chest.powerups)))
        for index, powerup in enumerate(chest.powerups):
            stripe = pygame.Rect(inset.x + index * stripe_width, inset.y, stripe_width, inset.height)
            if index == len(chest.powerups) - 1:
                stripe.width = inset.right - stripe.x
            pygame.draw.rect(surface, CHEST_COLORS[powerup], stripe)
        pygame.draw.rect(surface, (20, 24, 30), inset, width=2, border_radius=2)

    for enemy in floor.enemies:
        x, y = enemy.position
        center = (origin_x + y * cell_size + cell_size // 2, origin_y + x * cell_size + cell_size // 2)
        radius = max(3, cell_size // 3 if enemy.kind in {EnemyKind.SKIRMISHER, EnemyKind.ARCHER} else cell_size // 2)
        pygame.draw.circle(surface, ENEMY_COLORS[enemy.kind], center, radius)
        if cell_size >= 14:
            label = {EnemyKind.SKIRMISHER: "S", EnemyKind.ARCHER: "A", EnemyKind.MINI_BOSS: "M", EnemyKind.FINAL_BOSS: "F"}[
                enemy.kind
            ]
            text = tiny_font.render(label, True, (14, 18, 24))
            surface.blit(text, text.get_rect(center=center))

    for agent_id, (x, y) in enumerate(floor.start_positions):
        center = (origin_x + y * cell_size + cell_size // 2, origin_y + x * cell_size + cell_size // 2)
        pygame.draw.circle(surface, AGENT_COLOR, center, max(3, cell_size // 3))
        if cell_size >= 14:
            text = tiny_font.render(str(agent_id), True, (14, 18, 24))
            surface.blit(text, text.get_rect(center=center))

    legend_y = header_height + grid_pixels + 18
    legend_rect = pygame.Rect(margin, legend_y - 8, grid_pixels, legend_height - 12)
    pygame.draw.rect(surface, PANEL, legend_rect, border_radius=8)
    legend_lines = [
        "White spawn | Gold gate | Purple boss gate | Red/orange/purple enemies",
        "Chest stripes: D damage, R range, S speed, X diagonal, V vitality",
        "Terrain: hazard red, slow blue, heal green, walls gray",
    ]
    for index, line in enumerate(legend_lines):
        surface.blit(small_font.render(line, True, TEXT_COLOR), (margin + 14, legend_y + index * 18))
    return surface


def save_contact_sheet(floors: list[DungeonFloor], config: ArenaConfig, images: list[pygame.Surface], out_dir: Path) -> None:
    columns = 5
    thumb_width = 300
    thumb_height = 330
    rows = (len(images) + columns - 1) // columns
    sheet = pygame.Surface((columns * thumb_width, rows * thumb_height))
    sheet.fill(BACKGROUND)
    font = pygame.font.SysFont("consolas", 16, bold=True)
    for index, image in enumerate(images):
        row = index // columns
        column = index % columns
        target = pygame.Rect(column * thumb_width + 10, row * thumb_height + 34, thumb_width - 20, thumb_height - 44)
        scaled = pygame.transform.smoothscale(image, (target.width, target.height))
        sheet.blit(font.render(f"{floors[index].index:02d}. {floors[index].name}", True, TEXT_COLOR), (column * thumb_width + 12, row * thumb_height + 10))
        sheet.blit(scaled, target)
    pygame.image.save(sheet, out_dir / "all_floors_contact_sheet.png")


def main() -> None:
    args = build_parser().parse_args()
    config = ArenaConfig()
    floors = build_dungeon_floors(config.grid_size)
    args.out.mkdir(parents=True, exist_ok=True)
    pygame.init()
    pygame.font.init()
    images: list[pygame.Surface] = []
    for floor in floors:
        image = render_floor(floor, config, max(8, args.cell_size))
        images.append(image)
        filename = f"floor_{floor.index:02d}_{slugify(floor.name)}.png"
        pygame.image.save(image, args.out / filename)
    save_contact_sheet(floors, config, images, args.out)
    pygame.quit()
    print(f"exported {len(floors)} floors to {args.out}")


if __name__ == "__main__":
    main()
