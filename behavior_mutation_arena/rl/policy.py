from __future__ import annotations

import torch
from torch import nn


class ActorCriticPolicy(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor(observations)
        value = self.critic(observations).squeeze(-1)
        return logits, value

