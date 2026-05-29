from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from behavior_mutation_arena.config import (
    Action,
    ArenaConfig,
    ENEMY_STATS,
    EnemyKind,
    POWERUP_STATS,
    PowerUpType,
)
from behavior_mutation_arena.core.dungeon_floors import build_dungeon_floors
from behavior_mutation_arena.core.models import AgentMetrics, StepBatch


MOVE_DELTAS = {
    Action.MOVE_UP: (-1, 0),
    Action.MOVE_DOWN: (1, 0),
    Action.MOVE_LEFT: (0, -1),
    Action.MOVE_RIGHT: (0, 1),
    Action.MOVE_UP_LEFT: (-1, -1),
    Action.MOVE_UP_RIGHT: (-1, 1),
    Action.MOVE_DOWN_LEFT: (1, -1),
    Action.MOVE_DOWN_RIGHT: (1, 1),
}


@dataclass
class EnemyState:
    kind: EnemyKind
    position: tuple[int, int]
    health: float
    alive: bool = True


class ArenaEnvironment:
    def __init__(self, config: ArenaConfig, seed: int | None = None) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed if seed is not None else config.seed)
        self.grid_size = config.grid_size
        self.population_size = config.population_size
        self.floors = build_dungeon_floors(self.grid_size)
        self.current_generation = 0
        self.current_floor_index = 0
        self.current_floor = self.floors[0]
        self.positions = np.zeros((self.population_size, 2), dtype=np.int16)
        self.alive = np.ones(self.population_size, dtype=bool)
        self.health = np.full(self.population_size, config.initial_health, dtype=np.float32)
        self.energy = np.full(self.population_size, config.initial_energy, dtype=np.float32)
        self.damage_dealt = np.zeros(self.population_size, dtype=np.float32)
        self.boss_damage_dealt = np.zeros(self.population_size, dtype=np.float32)
        self.boss_hits = np.zeros(self.population_size, dtype=np.int16)
        self.total_reward = np.zeros(self.population_size, dtype=np.float32)
        self.survival_steps = np.zeros(self.population_size, dtype=np.int16)
        self.current_step = 0
        self.episode_step_limit = config.episode_steps_min
        self.attack_bonus = np.zeros(self.population_size, dtype=np.float32)
        self.range_bonus = np.zeros(self.population_size, dtype=np.float32)
        self.speed_bonus = np.zeros(self.population_size, dtype=np.int16)
        self.vitality_bonus = np.zeros(self.population_size, dtype=np.float32)
        self.diagonal_unlocked = np.zeros(self.population_size, dtype=bool)
        self.floors_cleared = 0
        self.max_floor_reached = 1
        self.bosses_defeated = 0
        self.mini_bosses_defeated = 0
        self.chests_opened = 0
        self.opened_chest_positions: set[tuple[int, tuple[int, int]]] = set()
        self.powerup_pickups = np.zeros(len(PowerUpType), dtype=np.int16)
        self.victory = False
        self.gate_open = True
        self.chest_lookup: dict[tuple[int, int], list[PowerUpType]] = {}
        self.enemy_states: list[EnemyState] = []
        self.team_wiped = False
        self.stalled_out = False
        self.steps_since_progress = 0
        self.closest_gate_distance = 0
        self.closest_chest_distance = -1
        self.closest_boss_distance = -1
        self.best_gate_distance_reached = -1
        self.best_gate_distance = 0
        self.best_chest_distance = -1
        self.best_boss_distance = -1
        self.gate_tile_visits = 0
        self.use_gate_attempts = 0
        self.invalid_use_gate_attempts = 0
        self.boss_first_hit_awarded = False
        self.floor5_entry_recorded = False
        self.floor5_entry_alive = 0
        self.floor5_entry_power_score = 0.0
        self.powered_agent_deaths = 0
        self.locked_gate_attempt_this_step = False

    def set_generation(self, generation: int) -> None:
        self.current_generation = generation

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.current_step = 0
        self.episode_step_limit = int(
            self.rng.integers(self.config.episode_steps_min, self.config.episode_steps_max + 1)
        )
        self.current_floor_index = 0
        self.current_floor = self.floors[0]
        self.alive.fill(True)
        self.health.fill(self.config.initial_health)
        self.energy.fill(self.config.initial_energy)
        self.damage_dealt.fill(0.0)
        self.boss_damage_dealt.fill(0.0)
        self.boss_hits.fill(0)
        self.total_reward.fill(0.0)
        self.survival_steps.fill(0)
        self.attack_bonus.fill(0.0)
        self.range_bonus.fill(0.0)
        self.speed_bonus.fill(0)
        self.vitality_bonus.fill(0.0)
        self.diagonal_unlocked.fill(False)
        self.floors_cleared = 0
        self.max_floor_reached = 1
        self.bosses_defeated = 0
        self.mini_bosses_defeated = 0
        self.chests_opened = 0
        self.opened_chest_positions.clear()
        self.powerup_pickups.fill(0)
        self.victory = False
        self.team_wiped = False
        self.stalled_out = False
        self.steps_since_progress = 0
        self.best_gate_distance_reached = -1
        self.gate_tile_visits = 0
        self.use_gate_attempts = 0
        self.invalid_use_gate_attempts = 0
        self.boss_first_hit_awarded = False
        self.floor5_entry_recorded = False
        self.floor5_entry_alive = 0
        self.floor5_entry_power_score = 0.0
        self.powered_agent_deaths = 0
        self.locked_gate_attempt_this_step = False
        self._load_floor(self.current_floor_index, preserve_team_state=False)
        return self.observe_all()

    def observe_all(self) -> np.ndarray:
        return np.vstack([self._observe_agent(agent_id) for agent_id in range(self.population_size)])

    def step(self, actions: np.ndarray) -> StepBatch:
        actions = np.asarray(actions, dtype=np.int64)
        rewards = np.zeros(self.population_size, dtype=np.float32)
        terminated = np.zeros(self.population_size, dtype=bool)
        truncated = np.zeros(self.population_size, dtype=bool)
        previous_damage = float(self.damage_dealt.sum())
        previous_chests = self.chests_opened
        previous_powerups = int(self.powerup_pickups.sum())
        previous_bosses = self.bosses_defeated
        previous_floors_cleared = self.floors_cleared
        previous_victory = self.victory
        self.locked_gate_attempt_this_step = False

        self.current_step += 1
        active_mask = self.alive.copy()
        self.survival_steps[active_mask] += 1
        rewards[active_mask] += self.config.step_penalty
        self.total_reward[active_mask] += self.config.step_penalty

        floor_transition = self._resolve_agent_actions(actions, rewards)
        self._apply_tile_effects(rewards)
        self.gate_tile_visits += self._count_gate_occupants()

        if not floor_transition and not self.victory:
            self._resolve_enemy_turn(rewards)

        death_mask = self.alive & (self.health <= 0.0)
        if death_mask.any():
            self._handle_agent_deaths(death_mask, rewards, terminated)

        progress_made = self._apply_distance_shaping(
            rewards,
            previous_damage=previous_damage,
            previous_chests=previous_chests,
            previous_powerups=previous_powerups,
            previous_bosses=previous_bosses,
            previous_floors_cleared=previous_floors_cleared,
            previous_victory=previous_victory,
            floor_transition=floor_transition,
        )
        if progress_made:
            self.steps_since_progress = 0
        else:
            self.steps_since_progress += 1
            if self.steps_since_progress % self.config.no_progress_penalty_interval == 0:
                self._add_team_reward(rewards, self.config.no_progress_penalty)

        episode_done = False
        if not self.alive.any():
            self.team_wiped = True
            self._add_team_reward(rewards, self.config.team_wipe_penalty)
            episode_done = True
            truncated[:] = False
        elif self.victory:
            episode_done = True
            truncated |= self.alive
        elif self.steps_since_progress >= self.config.no_progress_patience:
            self.stalled_out = True
            self._add_team_reward(rewards, self.config.no_progress_termination_penalty)
            episode_done = True
            truncated |= self.alive
        elif self.current_step >= self.episode_step_limit:
            episode_done = True
            truncated |= self.alive

        observations = self.observe_all()
        info = {
            "episode_done": episode_done,
            "alive_count": int(self.alive.sum()),
            "current_step": self.current_step,
            "episode_step_limit": self.episode_step_limit,
            "floor_index": self.current_floor_index + 1,
            "floor_name": self.current_floor.name,
            "bosses_defeated": self.bosses_defeated,
            "miniboss_defeated": int(self.mini_bosses_defeated > 0),
            "chests_opened": self.chests_opened,
            "powerups_picked": int(self.powerup_pickups.sum()),
            "damage_powerups": int(self.powerup_pickups[PowerUpType.DAMAGE]),
            "range_powerups": int(self.powerup_pickups[PowerUpType.RANGE]),
            "speed_powerups": int(self.powerup_pickups[PowerUpType.SPEED]),
            "diagonal_powerups": int(self.powerup_pickups[PowerUpType.DIAGONAL]),
            "vitality_powerups": int(self.powerup_pickups[PowerUpType.VITALITY]),
            "victory": self.victory,
            "gate_open": self.gate_open,
            "closest_gate_distance": self.closest_gate_distance,
            "best_gate_distance": self.best_gate_distance_reached,
            "closest_chest_distance": self.closest_chest_distance,
            "closest_boss_distance": self.closest_boss_distance,
            "gate_tile_visits": self.gate_tile_visits,
            "use_gate_attempts": self.use_gate_attempts,
            "invalid_use_gate_attempts": self.invalid_use_gate_attempts,
            "steps_since_progress": self.steps_since_progress,
            "stalled_out": self.stalled_out,
            "boss_damage": float(self.boss_damage_dealt.sum()),
            "boss_hits": int(self.boss_hits.sum()),
            "boss_health_remaining": self._boss_health_remaining(),
            "floor5_entry_alive": self.floor5_entry_alive,
            "floor5_entry_power_score": self.floor5_entry_power_score,
            "alive_attack_bonus": self._alive_mean_bonus(self.attack_bonus),
            "alive_range_bonus": self._alive_mean_bonus(self.range_bonus),
            "powered_agent_deaths": self.powered_agent_deaths,
        }
        return StepBatch(observations, rewards, terminated, truncated, info)

    def snapshot(self) -> dict[str, object]:
        chest_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        for (x, y), powerups in self.chest_lookup.items():
            if powerups:
                chest_grid[x, y] = int(powerups[0]) + 1
        enemy_positions = np.asarray([enemy.position for enemy in self.enemy_states if enemy.alive], dtype=np.int16)
        enemy_kind = np.asarray([int(enemy.kind) for enemy in self.enemy_states if enemy.alive], dtype=np.int8)
        enemy_health = np.asarray([enemy.health for enemy in self.enemy_states if enemy.alive], dtype=np.float32)
        return {
            "step": self.current_step,
            "episode_step_limit": self.episode_step_limit,
            "floor_index": self.current_floor_index + 1,
            "floor_name": self.current_floor.name,
            "theme": self.current_floor.theme,
            "terrain_grid": self.current_floor.terrain.copy(),
            "slow_tiles": self.current_floor.slow_tiles.copy(),
            "hazard_tiles": self.current_floor.hazard_tiles.copy(),
            "heal_tiles": self.current_floor.heal_tiles.copy(),
            "gate_position": np.asarray(self.current_floor.gate_position, dtype=np.int16),
            "gate_open": self.gate_open,
            "chest_grid": chest_grid,
            "opened_chest_positions": tuple(self.opened_chest_positions),
            "enemy_positions": enemy_positions,
            "enemy_kind": enemy_kind,
            "enemy_health": enemy_health,
            "positions": self.positions.copy(),
            "alive": self.alive.copy(),
            "health": self.health.copy(),
            "energy": self.energy.copy(),
            "reward": self.total_reward.copy(),
            "effective_max_health": float(np.max(self._effective_max_health())),
            "effective_max_health_by_agent": self._effective_max_health().copy(),
            "attack_bonus": self.attack_bonus.copy(),
            "range_bonus": self.range_bonus.copy(),
            "speed_bonus": self.speed_bonus.copy(),
            "diagonal_unlocked": self.diagonal_unlocked.copy(),
            "bosses_defeated": self.bosses_defeated,
            "miniboss_defeated": int(self.mini_bosses_defeated > 0),
            "chests_opened": self.chests_opened,
            "powerups_picked": int(self.powerup_pickups.sum()),
            "powerup_pickups": self.powerup_pickups.copy(),
            "floors_cleared": self.floors_cleared,
            "victory": self.victory,
            "closest_gate_distance": self.closest_gate_distance,
            "best_gate_distance": self.best_gate_distance_reached,
            "closest_chest_distance": self.closest_chest_distance,
            "closest_boss_distance": self.closest_boss_distance,
            "gate_tile_visits": self.gate_tile_visits,
            "use_gate_attempts": self.use_gate_attempts,
            "invalid_use_gate_attempts": self.invalid_use_gate_attempts,
            "steps_since_progress": self.steps_since_progress,
            "stalled_out": self.stalled_out,
            "boss_damage": self.boss_damage_dealt.copy(),
            "boss_hits": self.boss_hits.copy(),
            "boss_health_remaining": self._boss_health_remaining(),
            "floor5_entry_alive": self.floor5_entry_alive,
            "floor5_entry_power_score": self.floor5_entry_power_score,
            "alive_attack_bonus": self._alive_mean_bonus(self.attack_bonus),
            "alive_range_bonus": self._alive_mean_bonus(self.range_bonus),
            "powered_agent_deaths": self.powered_agent_deaths,
        }

    def get_agent_metrics(self) -> list[AgentMetrics]:
        metrics: list[AgentMetrics] = []
        for agent_id in range(self.population_size):
            reward = float(self.total_reward[agent_id])
            fitness = (
                reward
                + self.floors_cleared * self.config.floor_progress_weight
                + self.bosses_defeated * self.config.boss_weight
                + self.chests_opened * self.config.chest_weight
                + float(self.damage_dealt[agent_id]) * self.config.damage_weight
                + int(self.survival_steps[agent_id]) * self.config.survival_weight
                + int(self.victory) * self.config.victory_weight
            )
            metrics.append(
                AgentMetrics(
                    agent_id=agent_id,
                    reward=reward,
                    survival_steps=int(self.survival_steps[agent_id]),
                    floor_reached=self.max_floor_reached,
                    bosses_defeated=self.bosses_defeated,
                    chests_opened=self.chests_opened,
                    damage_dealt=float(self.damage_dealt[agent_id]),
                    gate_distance=float(self.closest_gate_distance),
                    fitness=float(fitness),
                    victory=int(self.victory),
                    powerups_picked=int(self.powerup_pickups.sum()),
                    damage_powerups=int(self.powerup_pickups[PowerUpType.DAMAGE]),
                    range_powerups=int(self.powerup_pickups[PowerUpType.RANGE]),
                    speed_powerups=int(self.powerup_pickups[PowerUpType.SPEED]),
                    diagonal_powerups=int(self.powerup_pickups[PowerUpType.DIAGONAL]),
                    vitality_powerups=int(self.powerup_pickups[PowerUpType.VITALITY]),
                    best_gate_distance=float(self.best_gate_distance_reached),
                    gate_tile_visits=float(self.gate_tile_visits),
                    use_gate_attempts=float(self.use_gate_attempts),
                    invalid_use_gate_attempts=float(self.invalid_use_gate_attempts),
                    boss_damage_dealt=float(self.boss_damage_dealt[agent_id]),
                    boss_hits=float(self.boss_hits[agent_id]),
                    boss_health_remaining=float(self._boss_health_remaining()),
                    floor5_entry_alive=float(self.floor5_entry_alive),
                    floor5_entry_power_score=float(self.floor5_entry_power_score),
                    alive_attack_bonus=float(self._alive_mean_bonus(self.attack_bonus)),
                    alive_range_bonus=float(self._alive_mean_bonus(self.range_bonus)),
                    powered_agent_deaths=float(self.powered_agent_deaths),
                    miniboss_defeated=float(self.mini_bosses_defeated > 0),
                )
            )
        return metrics

    def _resolve_agent_actions(self, actions: np.ndarray, rewards: np.ndarray) -> bool:
        for agent_id, action_value in enumerate(actions):
            if not self.alive[agent_id]:
                continue
            action = Action(int(action_value))
            if action in MOVE_DELTAS:
                self._move_agent(agent_id, action)
            elif action is Action.ATTACK:
                self._agent_attack(agent_id, rewards)
            elif action is Action.OPEN_CHEST:
                self._open_chest(agent_id, rewards)
            elif action is Action.USE_GATE:
                self.use_gate_attempts += 1
                if self._use_gate(agent_id, rewards):
                    return True
                if self.config.auto_use_gate and self.gate_open:
                    continue
                locked_attempt = (
                    not self.gate_open and self._agent_in_gate_radius(agent_id)
                )
                penalty = (
                    self.config.locked_gate_action_penalty
                    if locked_attempt
                    else self.config.invalid_gate_action_penalty
                )
                if locked_attempt:
                    self.locked_gate_attempt_this_step = True
                if penalty != 0.0:
                    rewards[agent_id] += penalty
                    self.total_reward[agent_id] += penalty
                self.invalid_use_gate_attempts += 1
        if self.config.auto_use_gate and self.gate_open:
            for agent_id in np.flatnonzero(self.alive):
                if self._agent_can_use_gate(int(agent_id)):
                    return self._use_gate(int(agent_id), rewards)
        return False

    def _move_agent(self, agent_id: int, action: Action) -> None:
        dx, dy = MOVE_DELTAS[action]
        if dx != 0 and dy != 0 and not self.diagonal_unlocked[agent_id]:
            return
        steps = 1 + int(self.speed_bonus[agent_id])
        for step_index in range(steps):
            cost = self.config.diagonal_energy_cost if dx != 0 and dy != 0 else self.config.step_energy_cost
            nx = int(self.positions[agent_id, 0] + dx)
            ny = int(self.positions[agent_id, 1] + dy)
            if not self._can_agent_enter(agent_id, nx, ny):
                break
            extra_cost = self.config.slow_tile_extra_cost if self.current_floor.slow_tiles[nx, ny] else 0.0
            self.positions[agent_id] = (nx, ny)
            self.energy[agent_id] = max(0.0, self.energy[agent_id] - cost - extra_cost)
            if self.current_floor.slow_tiles[nx, ny]:
                break
            if step_index == 0 and dx != 0 and dy != 0:
                break

    def _agent_attack(self, agent_id: int, rewards: np.ndarray) -> None:
        target_index = self._nearest_enemy_in_range(agent_id)
        if target_index is None:
            return
        enemy = self.enemy_states[target_index]
        stats = ENEMY_STATS[enemy.kind]
        raw_damage = self.config.attack_base_damage + float(self.attack_bonus[agent_id]) - stats.armor
        damage = min(max(1.0, raw_damage), max(0.0, enemy.health))
        enemy.health -= damage
        self.damage_dealt[agent_id] += damage
        self._add_agent_reward(agent_id, rewards, damage * self.config.attack_damage_reward_scale)
        if enemy.kind in {EnemyKind.MINI_BOSS, EnemyKind.FINAL_BOSS}:
            self.boss_damage_dealt[agent_id] += damage
            self.boss_hits[agent_id] += 1
            self._add_agent_reward(agent_id, rewards, damage * self.config.boss_damage_reward_scale)
            if not self.boss_first_hit_awarded:
                self.boss_first_hit_awarded = True
                self._add_agent_reward(agent_id, rewards, self.config.boss_first_hit_reward)
        if enemy.health <= 0.0 and enemy.alive:
            enemy.alive = False
            if enemy.kind in {EnemyKind.MINI_BOSS, EnemyKind.FINAL_BOSS}:
                self.bosses_defeated += 1
                if enemy.kind is EnemyKind.MINI_BOSS:
                    self.mini_bosses_defeated += 1
                self.gate_open = True
                bonus = self.config.final_boss_reward if enemy.kind is EnemyKind.FINAL_BOSS else self.config.mini_boss_reward
                self._add_team_reward(rewards, bonus)
            else:
                self._add_agent_reward(agent_id, rewards, stats.reward)

    def _open_chest(self, agent_id: int, rewards: np.ndarray) -> None:
        position = tuple(int(value) for value in self.positions[agent_id])
        powerups = self.chest_lookup.get(position)
        if not powerups:
            return
        powerup = powerups.pop(0)
        if not powerups:
            self.chest_lookup.pop(position, None)
        stats = POWERUP_STATS[powerup]
        self.attack_bonus[agent_id] += stats.attack_bonus
        self.range_bonus[agent_id] += stats.range_bonus
        self.speed_bonus[agent_id] += stats.speed_bonus
        self.diagonal_unlocked[agent_id] = self.diagonal_unlocked[agent_id] or stats.diagonal_unlocked
        if stats.vitality_bonus > 0.0:
            self.vitality_bonus[agent_id] += stats.vitality_bonus
            self.health[agent_id] = min(self._effective_max_health(agent_id), self.health[agent_id] + stats.vitality_bonus)
        chest_key = (self.current_floor_index, position)
        if chest_key not in self.opened_chest_positions:
            self.opened_chest_positions.add(chest_key)
            self.chests_opened += 1
        self.powerup_pickups[int(powerup)] += 1
        reward = self.config.chest_reward + stats.reward
        rewards[agent_id] += reward
        self.total_reward[agent_id] += reward

    def _use_gate(self, agent_id: int, rewards: np.ndarray) -> bool:
        if not self._agent_can_use_gate(agent_id):
            return False
        self.gate_tile_visits += 1
        if self.current_floor_index == self.config.num_floors - 1:
            self.victory = True
            self.floors_cleared = self.config.num_floors
            self.max_floor_reached = self.config.num_floors
            self._add_team_reward(rewards, self.config.victory_reward + self.config.gate_reward)
            return True
        self.floors_cleared = max(self.floors_cleared, self.current_floor_index + 1)
        self.max_floor_reached = max(self.max_floor_reached, self.current_floor_index + 2)
        self._add_team_reward(
            rewards,
            self.config.gate_reward + self.config.floor_clear_reward * float(self.current_floor_index + 1),
        )
        alive_power_score = self._alive_power_score()
        if alive_power_score > 0.0:
            self._add_team_reward(rewards, alive_power_score * self.config.powered_agent_transition_reward_scale)
        self.current_floor_index += 1
        self._load_floor(self.current_floor_index, preserve_team_state=True)
        return True

    def _apply_tile_effects(self, rewards: np.ndarray) -> None:
        for agent_id in range(self.population_size):
            if not self.alive[agent_id]:
                continue
            x, y = self.positions[agent_id]
            if self.current_floor.hazard_tiles[x, y]:
                self.health[agent_id] -= self.config.hazard_damage
                rewards[agent_id] += self.config.hazard_reward_penalty
                self.total_reward[agent_id] += self.config.hazard_reward_penalty
            if self.current_floor.heal_tiles[x, y]:
                self.health[agent_id] = min(
                    self._effective_max_health(agent_id),
                    self.health[agent_id] + self.config.heal_tile_amount,
                )

    def _resolve_enemy_turn(self, rewards: np.ndarray) -> None:
        for enemy in self.enemy_states:
            if not enemy.alive:
                continue
            stats = ENEMY_STATS[enemy.kind]
            target_id = self._nearest_alive_agent(enemy.position, max_distance=stats.aggro_range)
            if target_id is None:
                continue
            target_position = tuple(int(value) for value in self.positions[target_id])
            if self._distance(enemy.position, target_position) <= stats.attack_range:
                self.health[target_id] -= stats.damage
                continue
            self._move_enemy_toward(enemy, target_position)

    def _handle_agent_deaths(self, death_mask: np.ndarray, rewards: np.ndarray, terminated: np.ndarray) -> None:
        for agent_id in np.flatnonzero(death_mask):
            power_score = self._agent_power_score(int(agent_id))
            self.alive[agent_id] = False
            self.health[agent_id] = 0.0
            self.energy[agent_id] = 0.0
            penalty = self.config.death_penalty - power_score * self.config.powered_agent_death_penalty_scale
            rewards[agent_id] += penalty
            self.total_reward[agent_id] += penalty
            if power_score > 0.5:
                self.powered_agent_deaths += 1
            terminated[agent_id] = True

    def _load_floor(self, floor_index: int, preserve_team_state: bool) -> None:
        self.current_floor = self.floors[floor_index]
        self.current_floor_index = floor_index
        self.gate_open = not self.current_floor.gate_locked_until_boss
        self.boss_first_hit_awarded = False
        self.chest_lookup = {chest.position: list(chest.powerups) for chest in self.current_floor.chests}
        self.enemy_states = [
            EnemyState(kind=spawn.kind, position=spawn.position, health=ENEMY_STATS[spawn.kind].health)
            for spawn in self.current_floor.enemies
        ]
        for agent_id, position in enumerate(self.current_floor.start_positions):
            self.positions[agent_id] = position
            if preserve_team_state and self.alive[agent_id]:
                self.health[agent_id] = min(
                    self._effective_max_health(agent_id),
                    self.health[agent_id] + self.config.floor_transition_heal,
                )
                self.energy[agent_id] = min(self.config.max_energy, self.energy[agent_id] + 20.0)
        if floor_index == 4 and preserve_team_state and not self.floor5_entry_recorded:
            self.floor5_entry_recorded = True
            self.floor5_entry_alive = int(self.alive.sum())
            self.floor5_entry_power_score = self._alive_power_score()
        self.steps_since_progress = 0
        self.stalled_out = False
        self._reset_objective_tracking(reset_episode_tracking=not preserve_team_state)

    def _apply_distance_shaping(
        self,
        rewards: np.ndarray,
        previous_damage: float,
        previous_chests: int,
        previous_powerups: int,
        previous_bosses: int,
        previous_floors_cleared: int,
        previous_victory: bool,
        floor_transition: bool,
    ) -> bool:
        progress_made = floor_transition
        if float(self.damage_dealt.sum()) > previous_damage + 1e-6:
            progress_made = True
        if self.chests_opened > previous_chests:
            progress_made = True
        if int(self.powerup_pickups.sum()) > previous_powerups:
            progress_made = True
        if self.bosses_defeated > previous_bosses:
            progress_made = True
        if self.floors_cleared > previous_floors_cleared:
            progress_made = True
        if self.victory != previous_victory:
            progress_made = True

        gate_distance = self._closest_distance([self.current_floor.gate_position])
        chest_distance = self._closest_distance(list(self.chest_lookup))
        boss_targets = [
            enemy.position
            for enemy in self.enemy_states
            if enemy.alive and enemy.kind in {EnemyKind.MINI_BOSS, EnemyKind.FINAL_BOSS}
        ]
        boss_distance = self._closest_distance(boss_targets)

        self.closest_gate_distance = self._display_distance(gate_distance)
        self.closest_chest_distance = self._display_distance(chest_distance)
        self.closest_boss_distance = self._display_distance(boss_distance)
        if self.closest_gate_distance >= 0:
            if self.best_gate_distance_reached < 0:
                self.best_gate_distance_reached = self.closest_gate_distance
            else:
                self.best_gate_distance_reached = min(self.best_gate_distance_reached, self.closest_gate_distance)

        if self.gate_open and gate_distance is not None:
            if gate_distance < self.best_gate_distance:
                delta = self.best_gate_distance - gate_distance
                self._add_team_reward(rewards, delta * self.config.gate_distance_reward_scale)
                progress_made = True
                self.best_gate_distance = gate_distance
            elif self.best_gate_distance < 0:
                self.best_gate_distance = gate_distance
        elif gate_distance is not None and self.best_gate_distance < 0:
            self.best_gate_distance = gate_distance

        if chest_distance is not None:
            if self.best_chest_distance < 0:
                self.best_chest_distance = chest_distance
            elif chest_distance < self.best_chest_distance:
                delta = self.best_chest_distance - chest_distance
                self._add_team_reward(rewards, delta * self.config.chest_distance_reward_scale)
                progress_made = True
                self.best_chest_distance = chest_distance
        else:
            self.best_chest_distance = -1

        if not self.gate_open and boss_distance is not None and not self.locked_gate_attempt_this_step:
            if self.best_boss_distance < 0:
                self.best_boss_distance = boss_distance
            elif boss_distance < self.best_boss_distance and boss_distance > self._max_alive_attack_range():
                delta = self.best_boss_distance - boss_distance
                self._add_team_reward(rewards, delta * self.config.boss_distance_reward_scale)
                progress_made = True
                self.best_boss_distance = boss_distance
        else:
            self.best_boss_distance = boss_distance if boss_distance is not None else -1

        return progress_made

    def _move_enemy_toward(self, enemy: EnemyState, target: tuple[int, int]) -> None:
        ex, ey = enemy.position
        tx, ty = target
        dx = int(np.sign(tx - ex))
        dy = int(np.sign(ty - ey))
        candidates = [
            (ex + dx, ey + dy),
            (ex + dx, ey),
            (ex, ey + dy),
        ]
        for nx, ny in candidates:
            if self._can_enemy_enter(nx, ny):
                enemy.position = (nx, ny)
                return

    def _nearest_enemy_in_range(self, agent_id: int) -> int | None:
        origin = tuple(int(value) for value in self.positions[agent_id])
        attack_range = int(1 + self.range_bonus[agent_id])
        candidates: list[tuple[float, float, int]] = []
        for enemy_index, enemy in enumerate(self.enemy_states):
            if not enemy.alive:
                continue
            distance = self._distance(origin, enemy.position)
            if distance <= attack_range:
                candidates.append((distance, enemy.health, enemy_index))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[0][2]

    def _nearest_alive_agent(self, origin: tuple[int, int], max_distance: int | None = None) -> int | None:
        candidates: list[tuple[float, int]] = []
        for agent_id in range(self.population_size):
            if not self.alive[agent_id]:
                continue
            position = tuple(int(value) for value in self.positions[agent_id])
            distance = self._distance(origin, position)
            if max_distance is not None and distance > max_distance:
                continue
            candidates.append((distance, agent_id))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

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
        max_health = self._effective_max_health(agent_id)
        observation[cursor] = self.health[agent_id] / max_health if max_health > 0 else 0.0
        observation[cursor + 1] = self.energy[agent_id] / self.config.max_energy
        observation[cursor + 2] = (self.current_floor_index + 1) / self.config.num_floors
        observation[cursor + 3] = self.floors_cleared / self.config.num_floors
        observation[cursor + 4] = float(self.alive.sum()) / self.population_size
        observation[cursor + 5] = min(1.0, float(self.attack_bonus[agent_id]) / 12.0)
        observation[cursor + 6] = min(1.0, float(self.range_bonus[agent_id]) / 4.0)
        observation[cursor + 7] = min(1.0, float(self.speed_bonus[agent_id]) / 4.0)
        observation[cursor + 8] = float(self.diagonal_unlocked[agent_id])
        observation[cursor + 9] = self.chests_opened / max(1.0, float(self.config.num_floors * 2))
        observation[cursor + 10] = self.bosses_defeated / 2.0
        observation[cursor + 11] = float(self.gate_open)
        gate_dx, gate_dy, gate_distance = self._objective_vector(agent_id, [self.current_floor.gate_position])
        chest_dx, chest_dy, chest_distance = self._objective_vector(agent_id, list(self.chest_lookup))
        boss_targets = [
            enemy.position
            for enemy in self.enemy_states
            if enemy.alive and enemy.kind in {EnemyKind.MINI_BOSS, EnemyKind.FINAL_BOSS}
        ]
        boss_dx, boss_dy, boss_distance = self._objective_vector(agent_id, boss_targets)
        observation[cursor + 12] = gate_dx
        observation[cursor + 13] = gate_dy
        observation[cursor + 14] = gate_distance
        observation[cursor + 15] = chest_dx
        observation[cursor + 16] = chest_dy
        observation[cursor + 17] = chest_distance
        observation[cursor + 18] = boss_dx
        observation[cursor + 19] = boss_dy
        observation[cursor + 20] = boss_distance
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
        channels = np.zeros(self.config.vision_channels, dtype=np.float32)
        if not self._in_bounds(x, y) or self.current_floor.terrain[x, y]:
            channels[0] = 1.0
            return channels
        if accuracy < 1.0 and self.rng.random() > accuracy and (x, y) != tuple(self.positions[agent_id]):
            sample = int(self.rng.integers(0, self.config.vision_channels))
            channels[sample] = 1.0
            return channels
        if (x, y) == tuple(self.positions[agent_id]):
            channels[1] = 1.0
        elif any(self.alive[idx] and tuple(self.positions[idx]) == (x, y) for idx in range(self.population_size)):
            channels[2] = 1.0
        for enemy in self.enemy_states:
            if not enemy.alive or enemy.position != (x, y):
                continue
            if enemy.kind in {EnemyKind.MINI_BOSS, EnemyKind.FINAL_BOSS}:
                channels[4] = 1.0
            else:
                channels[3] = 1.0
            break
        if (x, y) in self.chest_lookup:
            channels[5] = 1.0
        if (x, y) == self.current_floor.gate_position:
            channels[6 if self.gate_open else 7] = 1.0
        if self.current_floor.slow_tiles[x, y]:
            channels[8] = 1.0
        if self.current_floor.hazard_tiles[x, y]:
            channels[9] = 1.0
        if self.current_floor.heal_tiles[x, y]:
            channels[10] = 1.0
        return channels

    def _can_agent_enter(self, agent_id: int, x: int, y: int) -> bool:
        if not self._in_bounds(x, y) or self.current_floor.terrain[x, y]:
            return False
        for other_id in range(self.population_size):
            if other_id != agent_id and self.alive[other_id] and tuple(self.positions[other_id]) == (x, y):
                return False
        for enemy in self.enemy_states:
            if enemy.alive and enemy.position == (x, y):
                return False
        return True

    def _can_enemy_enter(self, x: int, y: int) -> bool:
        if not self._in_bounds(x, y) or self.current_floor.terrain[x, y]:
            return False
        if (x, y) == self.current_floor.gate_position:
            return False
        if any(self.alive[idx] and tuple(self.positions[idx]) == (x, y) for idx in range(self.population_size)):
            return False
        if any(enemy.alive and enemy.position == (x, y) for enemy in self.enemy_states):
            return False
        return True

    def _effective_max_health(self, agent_id: int | None = None) -> float | np.ndarray:
        values = self.config.max_health + self.vitality_bonus
        if agent_id is None:
            return values
        return float(values[agent_id])

    def _closest_distance(self, targets: list[tuple[int, int]]) -> int | None:
        if not targets:
            return None
        live_positions = [tuple(int(value) for value in self.positions[idx]) for idx in range(self.population_size) if self.alive[idx]]
        if not live_positions:
            return None
        return min(self._manhattan_distance(position, target) for position in live_positions for target in targets)

    def _display_distance(self, distance: int | None) -> int:
        return int(distance) if distance is not None else -1

    def _reset_objective_tracking(self, reset_episode_tracking: bool = False) -> None:
        gate_distance = self._closest_distance([self.current_floor.gate_position])
        chest_distance = self._closest_distance(list(self.chest_lookup))
        boss_targets = [
            enemy.position
            for enemy in self.enemy_states
            if enemy.alive and enemy.kind in {EnemyKind.MINI_BOSS, EnemyKind.FINAL_BOSS}
        ]
        boss_distance = self._closest_distance(boss_targets)
        self.closest_gate_distance = self._display_distance(gate_distance)
        self.closest_chest_distance = self._display_distance(chest_distance)
        self.closest_boss_distance = self._display_distance(boss_distance)
        self.best_gate_distance = self.closest_gate_distance
        self.best_chest_distance = self.closest_chest_distance
        self.best_boss_distance = self.closest_boss_distance
        if reset_episode_tracking or self.best_gate_distance_reached < 0:
            self.best_gate_distance_reached = self.closest_gate_distance
        elif self.closest_gate_distance >= 0:
            self.best_gate_distance_reached = min(self.best_gate_distance_reached, self.closest_gate_distance)

    def _objective_vector(self, agent_id: int, targets: list[tuple[int, int]]) -> tuple[float, float, float]:
        if not targets:
            return 0.0, 0.0, 0.0
        origin = tuple(int(value) for value in self.positions[agent_id])
        target = min(targets, key=lambda candidate: self._manhattan_distance(origin, candidate))
        scale = max(1.0, float(self.grid_size - 1))
        distance_scale = max(1.0, float((self.grid_size - 1) * 2))
        dx = float(target[0] - origin[0]) / scale
        dy = float(target[1] - origin[1]) / scale
        distance = float(self._manhattan_distance(origin, target)) / distance_scale
        return dx, dy, distance

    def _count_gate_occupants(self) -> int:
        return sum(1 for agent_id in range(self.population_size) if self._agent_can_use_gate(agent_id))

    def _agent_can_use_gate(self, agent_id: int) -> bool:
        return self.gate_open and self._agent_in_gate_radius(agent_id)

    def _agent_in_gate_radius(self, agent_id: int) -> bool:
        if not self.alive[agent_id]:
            return False
        position = tuple(int(value) for value in self.positions[agent_id])
        return self._manhattan_distance(position, self.current_floor.gate_position) <= self.config.gate_interaction_radius

    def _agent_power_score(self, agent_id: int) -> float:
        return float(
            self.attack_bonus[agent_id]
            + self.range_bonus[agent_id] * 2.0
            + self.speed_bonus[agent_id] * 2.0
            + float(self.diagonal_unlocked[agent_id]) * 2.0
            + self.vitality_bonus[agent_id] / 12.0
        )

    def _alive_power_score(self) -> float:
        return float(sum(self._agent_power_score(int(agent_id)) for agent_id in np.flatnonzero(self.alive)))

    def _alive_mean_bonus(self, values: np.ndarray) -> float:
        if not self.alive.any():
            return 0.0
        return float(np.mean(values[self.alive]))

    def _boss_health_remaining(self) -> float:
        boss_health = [
            enemy.health
            for enemy in self.enemy_states
            if enemy.alive and enemy.kind in {EnemyKind.MINI_BOSS, EnemyKind.FINAL_BOSS}
        ]
        return float(sum(boss_health))

    def _max_alive_attack_range(self) -> int:
        if not self.alive.any():
            return 1
        return int(1 + np.max(self.range_bonus[self.alive]))

    def _distance(self, origin: tuple[int, int], target: tuple[int, int]) -> int:
        return max(abs(origin[0] - target[0]), abs(origin[1] - target[1]))

    def _manhattan_distance(self, origin: tuple[int, int], target: tuple[int, int]) -> int:
        return abs(origin[0] - target[0]) + abs(origin[1] - target[1])

    def _add_team_reward(self, rewards: np.ndarray, value: float) -> None:
        rewards += value
        self.total_reward += value

    def _add_agent_reward(self, agent_id: int, rewards: np.ndarray, value: float) -> None:
        rewards[agent_id] += value
        self.total_reward[agent_id] += value

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size
