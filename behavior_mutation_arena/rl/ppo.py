from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from behavior_mutation_arena.config import Action, ArenaConfig
from behavior_mutation_arena.rl.buffer import RolloutBuffer
from behavior_mutation_arena.rl.policy import ActorCriticPolicy


class PPOPolicyBank:
    def __init__(self, config: ArenaConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.policies = nn.ModuleList(
            [
                ActorCriticPolicy(config.observation_dim, config.action_size, config.hidden_size).to(self.device)
                for _ in range(config.population_size)
            ]
        )
        self.optimizers = [
            torch.optim.Adam(policy.parameters(), lr=config.ppo_learning_rate) for policy in self.policies
        ]

    def act(self, observations: np.ndarray, active_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        actions = np.full(self.config.population_size, Action.STAY, dtype=np.int64)
        log_probs = np.zeros(self.config.population_size, dtype=np.float32)
        values = np.zeros(self.config.population_size, dtype=np.float32)
        for agent_id in np.flatnonzero(active_mask):
            policy = self.policies[agent_id]
            obs_tensor = torch.from_numpy(observations[agent_id]).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits, value = policy(obs_tensor)
                distribution = Categorical(logits=logits)
                action = distribution.sample()
                actions[agent_id] = int(action.item())
                log_probs[agent_id] = float(distribution.log_prob(action).item())
                values[agent_id] = float(value.item())
        return actions, log_probs, values

    def evaluate_values(self, observations: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
        values = np.zeros(self.config.population_size, dtype=np.float32)
        for agent_id in np.flatnonzero(active_mask):
            obs_tensor = torch.from_numpy(observations[agent_id]).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, value = self.policies[agent_id](obs_tensor)
                values[agent_id] = float(value.item())
        return values

    def update(self, buffers: list[RolloutBuffer], last_values: np.ndarray) -> None:
        for agent_id, buffer in enumerate(buffers):
            batch = buffer.finish(
                last_value=float(last_values[agent_id]),
                gamma=self.config.ppo_gamma,
                gae_lambda=self.config.ppo_lambda,
            )
            if batch is None:
                continue
            observations = torch.from_numpy(batch.observations).float().to(self.device)
            actions = torch.from_numpy(batch.actions).long().to(self.device)
            old_log_probs = torch.from_numpy(batch.log_probs).float().to(self.device)
            returns = torch.from_numpy(batch.returns).float().to(self.device)
            advantages = torch.from_numpy(batch.advantages).float().to(self.device)
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

            policy = self.policies[agent_id]
            optimizer = self.optimizers[agent_id]
            batch_size = actions.shape[0]
            minibatch_size = min(self.config.ppo_minibatch_size, batch_size)

            for _ in range(self.config.ppo_epochs):
                permutation = torch.randperm(batch_size, device=self.device)
                for start in range(0, batch_size, minibatch_size):
                    indices = permutation[start : start + minibatch_size]
                    logits, values = policy(observations[indices])
                    distribution = Categorical(logits=logits)
                    entropy = distribution.entropy().mean()
                    new_log_probs = distribution.log_prob(actions[indices])
                    ratio = torch.exp(new_log_probs - old_log_probs[indices])
                    unclipped = ratio * advantages[indices]
                    clipped = torch.clamp(
                        ratio,
                        1.0 - self.config.ppo_clip,
                        1.0 + self.config.ppo_clip,
                    ) * advantages[indices]
                    actor_loss = -torch.min(unclipped, clipped).mean()
                    critic_loss = torch.nn.functional.mse_loss(values, returns[indices])
                    loss = (
                        actor_loss
                        + self.config.ppo_value_coef * critic_loss
                        - self.config.ppo_entropy_coef * entropy
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), self.config.ppo_max_grad_norm)
                    optimizer.step()

    def export_policy_state(self, agent_id: int) -> dict[str, torch.Tensor]:
        state_dict = self.policies[agent_id].state_dict()
        return {name: tensor.detach().cpu().clone() for name, tensor in state_dict.items()}

    def export_population_state(self) -> list[dict[str, torch.Tensor]]:
        return [self.export_policy_state(agent_id) for agent_id in range(self.config.population_size)]

    def export_optimizer_states(self) -> list[dict[str, object]]:
        return [optimizer.state_dict() for optimizer in self.optimizers]

    def load_population(self, population_states: list[dict[str, torch.Tensor]]) -> None:
        for agent_id, state in enumerate(population_states):
            self.policies[agent_id].load_state_dict(state)
            self.optimizers[agent_id] = torch.optim.Adam(
                self.policies[agent_id].parameters(),
                lr=self.config.ppo_learning_rate,
            )

    def load_optimizer_states(self, optimizer_states: list[dict[str, object]]) -> None:
        for agent_id, state in enumerate(optimizer_states):
            if agent_id >= len(self.optimizers):
                break
            self.optimizers[agent_id].load_state_dict(state)

    def checkpoint_payload(self, champion_id: int, generation: int) -> dict[str, object]:
        return {
            "generation": generation,
            "champion_id": champion_id,
            "policy_state": self.export_policy_state(champion_id),
            "config": self.config.to_dict(),
        }
