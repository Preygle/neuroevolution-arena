from __future__ import annotations

import numpy as np
import torch

from behavior_mutation_arena.config import ArenaConfig
from behavior_mutation_arena.core.models import AgentMetrics
from behavior_mutation_arena.rl.ppo import PPOPolicyBank


class EvolutionEngine:
    def __init__(self, config: ArenaConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def evolve(self, policy_bank: PPOPolicyBank, metrics: list[AgentMetrics]) -> list[int]:
        fitness = np.asarray([metric.fitness for metric in metrics], dtype=np.float32)
        elite_count = max(1, int(np.ceil(self.config.population_size * self.config.elite_fraction)))
        elite_indices = np.argsort(fitness)[-elite_count:][::-1].tolist()
        next_population: list[dict[str, torch.Tensor]] = []

        for elite_id in elite_indices:
            next_population.append(policy_bank.export_policy_state(elite_id))

        elite_fitness = fitness[elite_indices]
        elite_scores = np.exp(elite_fitness - elite_fitness.max())
        parent_distribution = elite_scores / elite_scores.sum()

        while len(next_population) < self.config.population_size:
            first_parent_id = int(self.rng.choice(elite_indices, p=parent_distribution))
            second_parent_id = int(self.rng.choice(elite_indices, p=parent_distribution))
            first_parent = policy_bank.export_policy_state(first_parent_id)
            second_parent = policy_bank.export_policy_state(second_parent_id)
            if self.rng.random() < self.config.crossover_rate and first_parent_id != second_parent_id:
                child_state = self._crossover_states(first_parent, second_parent)
            else:
                child_state = first_parent
            next_population.append(self._mutate_state(child_state))

        policy_bank.load_population(next_population[: self.config.population_size])
        return elite_indices

    def _mutate_state(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mutated: dict[str, torch.Tensor] = {}
        for name, tensor in state.items():
            clone = tensor.clone()
            if clone.is_floating_point():
                clone += torch.randn_like(clone) * self.config.mutation_std
            mutated[name] = clone
        return mutated

    def _crossover_states(
        self,
        first_state: dict[str, torch.Tensor],
        second_state: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        crossed: dict[str, torch.Tensor] = {}
        for name, first_tensor in first_state.items():
            second_tensor = second_state[name]
            if not first_tensor.is_floating_point():
                crossed[name] = first_tensor.clone()
                continue
            selector = torch.rand_like(first_tensor, dtype=torch.float32)
            mask = selector < self.config.crossover_swap_probability
            crossed[name] = torch.where(mask, first_tensor, second_tensor)
        return crossed
