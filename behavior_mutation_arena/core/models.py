from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class AgentMetrics:
    agent_id: int
    reward: float
    survival_steps: float
    floor_reached: float
    bosses_defeated: float
    chests_opened: float
    damage_dealt: float
    gate_distance: float
    fitness: float
    victory: float = 0.0
    powerups_picked: float = 0.0
    damage_powerups: float = 0.0
    range_powerups: float = 0.0
    speed_powerups: float = 0.0
    diagonal_powerups: float = 0.0
    vitality_powerups: float = 0.0
    best_gate_distance: float = 0.0
    gate_tile_visits: float = 0.0
    use_gate_attempts: float = 0.0
    invalid_use_gate_attempts: float = 0.0
    boss_damage_dealt: float = 0.0
    boss_hits: float = 0.0
    boss_health_remaining: float = 0.0
    floor5_entry_alive: float = 0.0
    floor5_entry_power_score: float = 0.0
    alive_attack_bonus: float = 0.0
    alive_range_bonus: float = 0.0
    powered_agent_deaths: float = 0.0
    miniboss_defeated: float = 0.0


@dataclass
class StepBatch:
    observations: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    info: dict[str, Any]


@dataclass
class GenerationSummary:
    generation: int
    best_fitness: float
    mean_fitness: float
    mean_reward: float
    mean_survival: float
    mean_floor_reached: float
    mean_bosses_defeated: float
    mean_chests_opened: float
    mean_damage: float
    mean_gate_distance: float
    mean_best_gate_distance: float
    mean_gate_tile_visits: float
    mean_use_gate_attempts: float
    mean_invalid_use_gate_attempts: float
    success_rate: float
    champion_id: int
    elite_ids: list[int]
    mean_powerups_picked: float = 0.0
    mean_damage_powerups: float = 0.0
    mean_range_powerups: float = 0.0
    mean_speed_powerups: float = 0.0
    mean_diagonal_powerups: float = 0.0
    mean_vitality_powerups: float = 0.0
    mean_boss_damage: float = 0.0
    mean_boss_hits: float = 0.0
    mean_boss_health_remaining: float = 0.0
    mean_floor5_entry_alive: float = 0.0
    mean_floor5_entry_power_score: float = 0.0
    mean_alive_attack_bonus: float = 0.0
    mean_alive_range_bonus: float = 0.0
    mean_powered_agent_deaths: float = 0.0
    mean_miniboss_defeated: float = 0.0
