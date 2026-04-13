from __future__ import annotations

from typing import Iterable

import numpy as np

from behavior_mutation_arena.config import (
    Action,
    ArenaConfig,
    ITEM_TO_WEAPON,
    ItemKind,
    WEAPON_STATS,
    WeaponKind,
)
from behavior_mutation_arena.core.map_pool import ArenaMap, build_training_map_pool
from behavior_mutation_arena.core.models import AgentMetrics, StepBatch


MOVE_DELTAS = {
    Action.MOVE_UP: (-1, 0),
    Action.MOVE_DOWN: (1, 0),
    Action.MOVE_LEFT: (0, -1),
    Action.MOVE_RIGHT: (0, 1),
}


class ArenaEnvironment:
    def __init__(self, config: ArenaConfig, seed: int | None = None) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed if seed is not None else config.seed)
        self.grid_size = config.grid_size
        self.population_size = config.population_size
        self.map_pool = build_training_map_pool(self.grid_size)
        self.current_generation = 0
        self.current_map = self.map_pool[0]
        self.forced_map_index: int | None = None
        self.occupancy = np.full((self.grid_size, self.grid_size), -1, dtype=np.int16)
        self.item_grid = np.full((self.grid_size, self.grid_size), ItemKind.EMPTY, dtype=np.int8)
        self.terrain_grid = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.positions = np.zeros((self.population_size, 2), dtype=np.int16)
        self.alive = np.ones(self.population_size, dtype=bool)
        self.health = np.full(self.population_size, config.initial_health, dtype=np.float32)
        self.energy = np.full(self.population_size, config.initial_energy, dtype=np.float32)
        self.weapon_kind = np.full(self.population_size, WeaponKind.NONE, dtype=np.int8)
        self.weapon_durability = np.zeros(self.population_size, dtype=np.float32)
        self.total_reward = np.zeros(self.population_size, dtype=np.float32)
        self.kills = np.zeros(self.population_size, dtype=np.int16)
        self.damage_dealt = np.zeros(self.population_size, dtype=np.float32)
        self.survival_steps = np.zeros(self.population_size, dtype=np.int16)
        self.explored_cells = np.zeros(self.population_size, dtype=np.int16)
        self.camping_steps = np.zeros(self.population_size, dtype=np.int16)
        self.camping_streak = np.zeros(self.population_size, dtype=np.int16)
        self.anchor_positions = np.zeros((self.population_size, 2), dtype=np.int16)
        self.visited_cells = np.zeros((self.population_size, self.grid_size, self.grid_size), dtype=bool)
        self.current_step = 0
        self.episode_step_limit = config.episode_steps_min

    def set_generation(self, generation: int) -> None:
        self.current_generation = generation

    def set_forced_map_index(self, map_index: int | None) -> None:
        self.forced_map_index = map_index

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.current_step = 0
        self.episode_step_limit = int(
            self.rng.integers(self.config.episode_steps_min, self.config.episode_steps_max + 1)
        )
        self.current_map = self._select_map()
        self.terrain_grid = self.current_map.terrain.copy()
        self.occupancy.fill(-1)
        self.item_grid.fill(ItemKind.EMPTY)
        self.alive.fill(True)
        self.health.fill(self.config.initial_health)
        self.energy.fill(self.config.initial_energy)
        self.weapon_kind.fill(WeaponKind.NONE)
        self.weapon_durability.fill(0.0)
        self.total_reward.fill(0.0)
        self.kills.fill(0)
        self.damage_dealt.fill(0.0)
        self.survival_steps.fill(0)
        self.explored_cells.fill(0)
        self.camping_steps.fill(0)
        self.camping_streak.fill(0)
        self.visited_cells.fill(False)
        self._spawn_agents()
        self.anchor_positions[:] = self.positions
        self._mark_initial_exploration()
        self._spawn_initial_items()
        return self.observe_all()

    def observe_all(self) -> np.ndarray:
        return np.vstack([self._observe_agent(agent_id) for agent_id in range(self.population_size)])

    def step(self, actions: np.ndarray) -> StepBatch:
        actions = np.asarray(actions, dtype=np.int64)
        rewards = np.zeros(self.population_size, dtype=np.float32)
        terminated = np.zeros(self.population_size, dtype=bool)
        truncated = np.zeros(self.population_size, dtype=bool)

        self.current_step += 1
        active_mask = self.alive.copy()
        self.survival_steps[active_mask] += 1
        rewards[active_mask] += self.config.survival_reward
        self.total_reward[active_mask] += self.config.survival_reward
        self.energy[active_mask] -= self.config.step_energy_cost
        starving = active_mask & (self.energy <= 0.0)
        self.health[starving] -= self.config.starvation_damage

        self._resolve_moves(actions)
        self._apply_exploration_reward(rewards, active_mask)
        self._apply_camping_penalty(rewards, active_mask)
        self._resolve_pickups(actions, rewards)
        terminated |= self._eliminate_dead_agents()
        self._resolve_attacks(actions, rewards, terminated)
        terminated |= self._eliminate_dead_agents()

        episode_done = self.current_step >= self.episode_step_limit or not self.alive.any()
        if episode_done:
            truncated |= self.alive
        observations = self.observe_all()
        info = {
            "episode_done": episode_done,
            "alive_count": int(self.alive.sum()),
            "current_step": self.current_step,
            "episode_step_limit": self.episode_step_limit,
            "map_name": self.current_map.name,
        }
        return StepBatch(observations, rewards, terminated, truncated, info)

    def snapshot(self) -> dict[str, np.ndarray | int | str]:
        return {
            "step": self.current_step,
            "episode_step_limit": self.episode_step_limit,
            "map_name": self.current_map.name,
            "item_grid": self.item_grid.copy(),
            "terrain_grid": self.terrain_grid.copy(),
            "positions": self.positions.copy(),
            "alive": self.alive.copy(),
            "health": self.health.copy(),
            "energy": self.energy.copy(),
            "weapon_kind": self.weapon_kind.copy(),
            "weapon_durability": self.weapon_durability.copy(),
            "reward": self.total_reward.copy(),
            "kills": self.kills.copy(),
            "explored_cells": self.explored_cells.copy(),
            "camping_steps": self.camping_steps.copy(),
        }

    def get_agent_metrics(self) -> list[AgentMetrics]:
        metrics: list[AgentMetrics] = []
        for agent_id in range(self.population_size):
            reward = float(self.total_reward[agent_id])
            survival = int(self.survival_steps[agent_id])
            kills = int(self.kills[agent_id])
            explored = int(self.explored_cells[agent_id])
            damage = float(self.damage_dealt[agent_id])
            camping = int(self.camping_steps[agent_id])
            fitness = (
                reward
                + kills * self.config.fitness_kill_weight
                + damage * self.config.fitness_damage_weight
                + explored * self.config.fitness_exploration_weight
                - camping * self.config.fitness_camping_weight
            )
            metrics.append(
                AgentMetrics(
                    agent_id=agent_id,
                    reward=reward,
                    survival_steps=survival,
                    kills=kills,
                    fitness=float(fitness),
                    explored_cells=explored,
                    damage_dealt=damage,
                    camping_steps=camping,
                )
            )
        return metrics

    def _select_map(self) -> ArenaMap:
        if self.forced_map_index is not None:
            return self.map_pool[self.forced_map_index % len(self.map_pool)]
        initial = min(self.config.curriculum_initial_map_count, len(self.map_pool))
        full = min(self.config.curriculum_full_map_count, len(self.map_pool))
        growth = max(1, self.config.curriculum_growth_generations)
        if full <= initial:
            active_count = full
        else:
            progress = min(1.0, self.current_generation / growth)
            active_count = initial + int(round((full - initial) * progress))
            active_count = max(initial, min(full, active_count))
        return self.map_pool[self.current_generation % active_count]

    def _spawn_agents(self) -> None:
        candidates = self._spawn_candidates()
        if len(candidates) < self.population_size:
            candidates = self._free_cells()
        choice_indices = self.rng.permutation(len(candidates))[: self.population_size]
        for agent_id, pick_index in enumerate(choice_indices):
            x, y = candidates[pick_index]
            self.positions[agent_id] = (x, y)
            self.occupancy[x, y] = agent_id

    def _spawn_candidates(self) -> np.ndarray:
        band = max(1, self.config.spawn_band_width)
        spawn_mask = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        spawn_mask[:band, :] = True
        spawn_mask[-band:, :] = True
        spawn_mask[:, :band] = True
        spawn_mask[:, -band:] = True
        return np.argwhere(spawn_mask & ~self.terrain_grid)

    def _mark_initial_exploration(self) -> None:
        for agent_id in range(self.population_size):
            x, y = self.positions[agent_id]
            self.visited_cells[agent_id, x, y] = True
            self.explored_cells[agent_id] = 1

    def _spawn_initial_items(self) -> None:
        for item_kind, count in self.config.item_spawn_counts().items():
            for _ in range(count):
                self._spawn_item(item_kind)

    def _spawn_item(self, item_kind: ItemKind) -> None:
        candidates = self._preferred_item_cells(item_kind)
        if len(candidates) == 0:
            candidates = self._free_cells()
        if len(candidates) == 0:
            return
        pick = candidates[self.rng.integers(0, len(candidates))]
        self.item_grid[pick[0], pick[1]] = int(item_kind)

    def _preferred_item_cells(self, item_kind: ItemKind) -> np.ndarray:
        free_mask = self._free_mask()
        coords = np.argwhere(free_mask)
        if len(coords) == 0:
            return coords
        center = self.grid_size // 2
        center_distance = np.abs(coords[:, 0] - center) + np.abs(coords[:, 1] - center)
        center_radius = max(3, self.grid_size // 4)
        edge_band = max(2, self.grid_size // 5)
        edge_mask = (
            (coords[:, 0] < edge_band)
            | (coords[:, 1] < edge_band)
            | (coords[:, 0] >= self.grid_size - edge_band)
            | (coords[:, 1] >= self.grid_size - edge_band)
        )
        if item_kind in {ItemKind.FOOD, ItemKind.RARE_MELEE, ItemKind.RARE_RANGED}:
            preferred = coords[center_distance <= center_radius]
            if len(preferred) > 0:
                return preferred
        elif item_kind in {ItemKind.MELEE, ItemKind.RANGED}:
            preferred = coords[(center_distance <= center_radius + 2) & ~edge_mask]
            if len(preferred) > 0:
                return preferred
        elif item_kind is ItemKind.POISON:
            preferred = coords[edge_mask]
            if len(preferred) > 0:
                return preferred
        return coords

    def _resolve_moves(self, actions: np.ndarray) -> None:
        for agent_id in self.rng.permutation(self.population_size):
            if not self.alive[agent_id]:
                continue
            action = Action(int(actions[agent_id]))
            if action not in MOVE_DELTAS:
                continue
            dx, dy = MOVE_DELTAS[action]
            x, y = self.positions[agent_id]
            nx, ny = x + dx, y + dy
            if not self._in_bounds(nx, ny):
                continue
            if self.terrain_grid[nx, ny]:
                continue
            if self.occupancy[nx, ny] != -1:
                continue
            self.occupancy[x, y] = -1
            self.occupancy[nx, ny] = agent_id
            self.positions[agent_id] = (nx, ny)

    def _apply_exploration_reward(self, rewards: np.ndarray, active_mask: np.ndarray) -> None:
        for agent_id in np.flatnonzero(active_mask):
            x, y = self.positions[agent_id]
            if self.visited_cells[agent_id, x, y]:
                continue
            self.visited_cells[agent_id, x, y] = True
            self.explored_cells[agent_id] += 1
            rewards[agent_id] += self.config.exploration_reward
            self.total_reward[agent_id] += self.config.exploration_reward

    def _apply_camping_penalty(self, rewards: np.ndarray, active_mask: np.ndarray) -> None:
        for agent_id in np.flatnonzero(active_mask):
            x, y = self.positions[agent_id]
            ax, ay = self.anchor_positions[agent_id]
            distance = abs(int(x) - int(ax)) + abs(int(y) - int(ay))
            if distance <= self.config.camp_radius:
                self.camping_streak[agent_id] += 1
            else:
                self.anchor_positions[agent_id] = (x, y)
                self.camping_streak[agent_id] = 0
            if self.camping_streak[agent_id] <= self.config.camp_threshold:
                continue
            penalty = self.config.camping_penalty
            if self._is_corner_cell(int(x), int(y)):
                penalty += self.config.corner_camping_penalty
            rewards[agent_id] -= penalty
            self.total_reward[agent_id] -= penalty
            self.camping_steps[agent_id] += 1

    def _resolve_pickups(self, actions: np.ndarray, rewards: np.ndarray) -> None:
        for agent_id in self.rng.permutation(self.population_size):
            if not self.alive[agent_id]:
                continue
            if Action(int(actions[agent_id])) is not Action.PICK_ITEM:
                continue
            x, y = self.positions[agent_id]
            item_kind = ItemKind(int(self.item_grid[x, y]))
            if item_kind is ItemKind.EMPTY:
                continue
            if item_kind is ItemKind.FOOD:
                self.energy[agent_id] = min(self.config.max_energy, self.energy[agent_id] + self.config.food_energy)
                rewards[agent_id] += self.config.food_reward
                self.total_reward[agent_id] += self.config.food_reward
            elif item_kind is ItemKind.POISON:
                self.health[agent_id] -= self.config.poison_damage
                rewards[agent_id] += self.config.poison_reward
                self.total_reward[agent_id] += self.config.poison_reward
            else:
                weapon_kind = ITEM_TO_WEAPON[item_kind]
                self.weapon_kind[agent_id] = int(weapon_kind)
                self.weapon_durability[agent_id] = WEAPON_STATS[weapon_kind].durability
            self.item_grid[x, y] = int(ItemKind.EMPTY)
            self._spawn_item(item_kind)

    def _resolve_attacks(self, actions: np.ndarray, rewards: np.ndarray, terminated: np.ndarray) -> None:
        for agent_id in self.rng.permutation(self.population_size):
            if not self.alive[agent_id]:
                continue
            if Action(int(actions[agent_id])) is not Action.ATTACK:
                continue
            stats = WEAPON_STATS[WeaponKind(int(self.weapon_kind[agent_id]))]
            target_id = self._select_attack_target(agent_id, stats.range)
            if target_id is None:
                continue
            damage = stats.damage
            self.health[target_id] -= damage
            self.damage_dealt[agent_id] += damage
            damage_reward = damage * self.config.attack_damage_reward_scale
            rewards[agent_id] += damage_reward
            self.total_reward[agent_id] += damage_reward
            if stats.durability > 0:
                self.weapon_durability[agent_id] -= 1.0
                if self.weapon_durability[agent_id] <= 0:
                    self.weapon_kind[agent_id] = int(WeaponKind.NONE)
                    self.weapon_durability[agent_id] = 0.0
            if self.health[target_id] <= 0 and self.alive[target_id]:
                self._eliminate_agents([target_id])
                terminated[target_id] = True
                self.kills[agent_id] += 1
                rewards[agent_id] += self.config.kill_reward
                self.total_reward[agent_id] += self.config.kill_reward

    def _select_attack_target(self, agent_id: int, attack_range: int) -> int | None:
        origin = self.positions[agent_id]
        candidates: list[tuple[int, int, int]] = []
        for other_id in range(self.population_size):
            if other_id == agent_id or not self.alive[other_id]:
                continue
            target = self.positions[other_id]
            distance = abs(int(origin[0]) - int(target[0])) + abs(int(origin[1]) - int(target[1]))
            if distance <= attack_range:
                candidates.append((distance, int(self.health[other_id]), other_id))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[0][2]

    def _eliminate_dead_agents(self) -> np.ndarray:
        dead_mask = self.alive & (self.health <= 0)
        if dead_mask.any():
            self._eliminate_agents(np.flatnonzero(dead_mask))
        return dead_mask

    def _eliminate_agents(self, agent_ids: Iterable[int]) -> None:
        for agent_id in agent_ids:
            if not self.alive[agent_id]:
                continue
            x, y = self.positions[agent_id]
            self.alive[agent_id] = False
            self.occupancy[x, y] = -1
            self.health[agent_id] = 0.0
            self.energy[agent_id] = max(0.0, self.energy[agent_id])

    def _observe_agent(self, agent_id: int) -> np.ndarray:
        observation = np.zeros(self.config.observation_dim, dtype=np.float32)
        cursor = 0
        cursor = self._encode_window(agent_id, self.config.local_vision, 1.0, observation, cursor)
        cursor = self._encode_window(
            agent_id,
            self.config.extended_vision,
            self.config.extended_accuracy,
            observation,
            cursor,
        )
        observation[cursor] = self.health[agent_id] / self.config.max_health
        observation[cursor + 1] = self.energy[agent_id] / self.config.max_energy
        weapon = WeaponKind(int(self.weapon_kind[agent_id]))
        observation[cursor + 2 + int(weapon)] = 1.0
        if weapon is not WeaponKind.NONE:
            observation[cursor + 7] = self.weapon_durability[agent_id] / max(1.0, WEAPON_STATS[weapon].durability)
        return observation

    def _encode_window(
        self,
        agent_id: int,
        window_size: int,
        accuracy: float,
        target: np.ndarray,
        cursor: int,
    ) -> int:
        radius = window_size // 2
        ox, oy = self.positions[agent_id]
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                channels = self._encode_cell(agent_id, int(ox + dx), int(oy + dy), accuracy)
                next_cursor = cursor + self.config.vision_channels
                target[cursor:next_cursor] = channels
                cursor = next_cursor
        return cursor

    def _encode_cell(self, agent_id: int, x: int, y: int, accuracy: float) -> np.ndarray:
        if not self._in_bounds(x, y):
            channels = np.zeros(self.config.vision_channels, dtype=np.float32)
            channels[0] = 1.0
            return channels
        if self.terrain_grid[x, y]:
            channels = np.zeros(self.config.vision_channels, dtype=np.float32)
            channels[0] = 1.0
            return channels
        if accuracy < 1.0 and self.rng.random() > accuracy and (x, y) != tuple(self.positions[agent_id]):
            return self._sample_noisy_cell()
        channels = np.zeros(self.config.vision_channels, dtype=np.float32)
        occupant = self.occupancy[x, y]
        if occupant == agent_id:
            channels[1] = 1.0
        elif occupant != -1:
            channels[2] = 1.0
        item_kind = ItemKind(int(self.item_grid[x, y]))
        if item_kind is ItemKind.FOOD:
            channels[3] = 1.0
        elif item_kind is ItemKind.POISON:
            channels[4] = 1.0
        elif item_kind is ItemKind.MELEE:
            channels[5] = 1.0
        elif item_kind is ItemKind.RANGED:
            channels[6] = 1.0
        elif item_kind is ItemKind.RARE_MELEE:
            channels[7] = 1.0
        elif item_kind is ItemKind.RARE_RANGED:
            channels[8] = 1.0
        return channels

    def _sample_noisy_cell(self) -> np.ndarray:
        channels = np.zeros(self.config.vision_channels, dtype=np.float32)
        sample = int(self.rng.integers(0, self.config.vision_channels))
        if sample > 0:
            channels[sample] = 1.0
        return channels

    def _free_mask(self) -> np.ndarray:
        return (self.occupancy == -1) & (self.item_grid == ItemKind.EMPTY) & ~self.terrain_grid

    def _free_cells(self) -> np.ndarray:
        return np.argwhere(self._free_mask())

    def _is_corner_cell(self, x: int, y: int) -> bool:
        corner_band = max(2, self.grid_size // 5)
        return (
            (x < corner_band and y < corner_band)
            or (x < corner_band and y >= self.grid_size - corner_band)
            or (x >= self.grid_size - corner_band and y < corner_band)
            or (x >= self.grid_size - corner_band and y >= self.grid_size - corner_band)
        )

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size
