from __future__ import annotations

import argparse
import statistics as stats
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from behavior_mutation_arena.config import ArenaConfig
from behavior_mutation_arena.core.environment import ArenaEnvironment
from behavior_mutation_arena.rl.policy import ActorCriticPolicy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a saved champion checkpoint across the map pool.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="artifacts/checkpoints/best_policy.pt",
    )
    parser.add_argument("--episodes-per-map", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)
    return parser


def greedy_actions(policy: ActorCriticPolicy, observations: np.ndarray, alive: np.ndarray, device: torch.device) -> np.ndarray:
    actions = np.full(observations.shape[0], 4, dtype=np.int64)
    for agent_id in np.flatnonzero(alive):
        obs_tensor = torch.from_numpy(observations[agent_id]).float().unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = policy(obs_tensor)
        actions[agent_id] = int(torch.argmax(logits, dim=-1).item())
    return actions


def main() -> None:
    args = build_parser().parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ArenaConfig(**checkpoint.get("config", {}))
    device = torch.device(args.device or config.device)
    policy = ActorCriticPolicy(config.observation_dim, config.action_size, config.hidden_size).to(device)
    policy.load_state_dict(checkpoint["policy_state"])
    policy.eval()
    env = ArenaEnvironment(config)

    for map_index, arena_map in enumerate(env.map_pool):
        map_rewards: list[float] = []
        map_survival: list[float] = []
        map_kills: list[float] = []
        map_exploration: list[float] = []
        map_camping: list[float] = []

        for episode in range(args.episodes_per_map):
            env.set_forced_map_index(map_index)
            observations = env.reset(seed=config.seed + map_index * 100 + episode)
            done = False
            while not done:
                actions = greedy_actions(policy, observations, env.alive.copy(), device)
                step_batch = env.step(actions)
                observations = step_batch.observations
                done = bool(step_batch.info["episode_done"])
            metrics = env.get_agent_metrics()
            map_rewards.append(stats.mean(metric.reward for metric in metrics))
            map_survival.append(stats.mean(metric.survival_steps for metric in metrics))
            map_kills.append(stats.mean(metric.kills for metric in metrics))
            map_exploration.append(stats.mean(metric.explored_cells for metric in metrics))
            map_camping.append(stats.mean(metric.camping_steps for metric in metrics))

        print(
            f"{arena_map.name:14s} "
            f"reward={stats.mean(map_rewards):7.2f} "
            f"survival={stats.mean(map_survival):6.2f} "
            f"kills={stats.mean(map_kills):5.2f} "
            f"explore={stats.mean(map_exploration):6.2f} "
            f"camp={stats.mean(map_camping):6.2f}"
        )


if __name__ == "__main__":
    main()
