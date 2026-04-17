from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class AgentMetrics:
    agent_id: int
    reward: float
    survival_steps: int
    floor_reached: int
    bosses_defeated: int
    chests_opened: int
    damage_dealt: float
    gate_distance: float
    fitness: float
    victory: int = 0


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
    success_rate: float
    champion_id: int
    elite_ids: list[int]
