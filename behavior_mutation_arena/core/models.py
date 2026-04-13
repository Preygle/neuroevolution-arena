from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class AgentMetrics:
    agent_id: int
    reward: float
    survival_steps: int
    kills: int
    fitness: float
    explored_cells: int = 0
    damage_dealt: float = 0.0
    camping_steps: int = 0


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
    mean_kills: float
    champion_id: int
    elite_ids: list[int]
    mean_exploration: float = 0.0
    mean_damage: float = 0.0
    mean_camping: float = 0.0
