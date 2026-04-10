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

