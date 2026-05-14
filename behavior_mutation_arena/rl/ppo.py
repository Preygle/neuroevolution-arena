from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from behavior_mutation_arena.config import Action, ArenaConfig
from behavior_mutation_arena.rl.buffer import RolloutBatch, RolloutBuffer
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

    def act_many(
        self,
        observation_batches: list[np.ndarray],
        active_masks: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        instance_count = len(observation_batches)
        actions = [np.full(self.config.population_size, Action.STAY, dtype=np.int64) for _ in range(instance_count)]
        log_probs = [np.zeros(self.config.population_size, dtype=np.float32) for _ in range(instance_count)]
        values = [np.zeros(self.config.population_size, dtype=np.float32) for _ in range(instance_count)]
        for agent_id in range(self.config.population_size):
            active_instances = [index for index in range(instance_count) if active_masks[index][agent_id]]
            if not active_instances:
                continue
            policy = self.policies[agent_id]
            stacked = np.stack([observation_batches[index][agent_id] for index in active_instances], axis=0)
            obs_tensor = torch.from_numpy(stacked).float().to(self.device)
            with torch.no_grad():
                logits, value = policy(obs_tensor)
                distribution = Categorical(logits=logits)
                sampled_actions = distribution.sample()
                sampled_log_probs = distribution.log_prob(sampled_actions)
            sampled_actions_np = sampled_actions.detach().cpu().numpy()
            sampled_log_probs_np = sampled_log_probs.detach().cpu().numpy()
            sampled_values_np = value.detach().cpu().numpy()
            for slot, instance_index in enumerate(active_instances):
                actions[instance_index][agent_id] = int(sampled_actions_np[slot])
                log_probs[instance_index][agent_id] = float(sampled_log_probs_np[slot])
                values[instance_index][agent_id] = float(sampled_values_np[slot])
        return actions, log_probs, values

    def evaluate_values(self, observations: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
        values = np.zeros(self.config.population_size, dtype=np.float32)
        for agent_id in np.flatnonzero(active_mask):
            obs_tensor = torch.from_numpy(observations[agent_id]).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, value = self.policies[agent_id](obs_tensor)
                values[agent_id] = float(value.item())
        return values

    def evaluate_values_many(
        self,
        observation_batches: list[np.ndarray],
        active_masks: list[np.ndarray],
    ) -> list[np.ndarray]:
        values = [np.zeros(self.config.population_size, dtype=np.float32) for _ in range(len(observation_batches))]
        for agent_id in range(self.config.population_size):
            active_instances = [index for index in range(len(observation_batches)) if active_masks[index][agent_id]]
            if not active_instances:
                continue
            stacked = np.stack([observation_batches[index][agent_id] for index in active_instances], axis=0)
            obs_tensor = torch.from_numpy(stacked).float().to(self.device)
            with torch.no_grad():
                _, value = self.policies[agent_id](obs_tensor)
            sampled_values_np = value.detach().cpu().numpy()
            for slot, instance_index in enumerate(active_instances):
                values[instance_index][agent_id] = float(sampled_values_np[slot])
        return values

    def update(self, buffers: list[RolloutBuffer], last_values: np.ndarray) -> None:
        batches: list[RolloutBatch | None] = []
        for agent_id, buffer in enumerate(buffers):
            batch = buffer.finish(
                last_value=float(last_values[agent_id]),
                gamma=self.config.ppo_gamma,
                gae_lambda=self.config.ppo_lambda,
            )
            batches.append(batch)
        self.update_from_batches(batches)

    def update_from_batches(self, batches: list[RolloutBatch | None]) -> None:
        for agent_id, batch in enumerate(batches):
            if batch is None:
                continue
            self._update_agent(agent_id, batch)

    def _update_agent(self, agent_id: int, batch: RolloutBatch) -> None:
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
        target_kl = float(getattr(self.config, "ppo_target_kl", 0.0) or 0.0)

        for _ in range(self.config.ppo_epochs):
            permutation = torch.randperm(batch_size, device=self.device)
            epoch_kl_values: list[float] = []
            for start in range(0, batch_size, minibatch_size):
                indices = permutation[start : start + minibatch_size]
                logits, values = policy(observations[indices])
                distribution = Categorical(logits=logits)
                entropy = distribution.entropy().mean()
                new_log_probs = distribution.log_prob(actions[indices])
                log_ratio = new_log_probs - old_log_probs[indices]
                ratio = torch.exp(log_ratio)
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
                with torch.no_grad():
                    approx_kl = (ratio - 1.0 - log_ratio).mean().item()
                epoch_kl_values.append(approx_kl)
            if target_kl > 0.0 and epoch_kl_values:
                mean_epoch_kl = sum(epoch_kl_values) / len(epoch_kl_values)
                if mean_epoch_kl > 1.5 * target_kl:
                    break

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
