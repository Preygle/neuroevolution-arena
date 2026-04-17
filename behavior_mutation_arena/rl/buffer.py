from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RolloutBatch:
    observations: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray

    @staticmethod
    def concatenate(batches: list["RolloutBatch"]) -> "RolloutBatch" | None:
        if not batches:
            return None
        if len(batches) == 1:
            return batches[0]
        return RolloutBatch(
            observations=np.concatenate([batch.observations for batch in batches], axis=0),
            actions=np.concatenate([batch.actions for batch in batches], axis=0),
            log_probs=np.concatenate([batch.log_probs for batch in batches], axis=0),
            returns=np.concatenate([batch.returns for batch in batches], axis=0),
            advantages=np.concatenate([batch.advantages for batch in batches], axis=0),
        )


class RolloutBuffer:
    def __init__(self) -> None:
        self.observations: list[np.ndarray] = []
        self.actions: list[int] = []
        self.log_probs: list[float] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.dones: list[bool] = []

    def add(
        self,
        observation: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        self.observations.append(observation.astype(np.float32, copy=True))
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.dones.append(bool(done))

    def finish(self, last_value: float, gamma: float, gae_lambda: float) -> RolloutBatch | None:
        if not self.observations:
            return None
        rewards = np.asarray(self.rewards, dtype=np.float32)
        values = np.asarray(self.values + [last_value], dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for index in range(len(rewards) - 1, -1, -1):
            delta = rewards[index] + gamma * values[index + 1] * (1.0 - dones[index]) - values[index]
            gae = delta + gamma * gae_lambda * (1.0 - dones[index]) * gae
            advantages[index] = gae
        returns = advantages + values[:-1]
        batch = RolloutBatch(
            observations=np.asarray(self.observations, dtype=np.float32),
            actions=np.asarray(self.actions, dtype=np.int64),
            log_probs=np.asarray(self.log_probs, dtype=np.float32),
            returns=returns.astype(np.float32),
            advantages=advantages.astype(np.float32),
        )
        self.clear()
        return batch

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self) -> int:
        return len(self.actions)
