from __future__ import annotations

import argparse
import statistics as stats
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from behavior_mutation_arena.config import ArenaConfig
from behavior_mutation_arena.core.environment import ArenaEnvironment
from behavior_mutation_arena.rl.policy import ActorCriticPolicy


def config_from_checkpoint(payload: dict[str, object]) -> ArenaConfig:
    checkpoint_config = payload.get("config", {})
    allowed_fields = {field.name for field in fields(ArenaConfig)}
    filtered_config = {
        key: value for key, value in checkpoint_config.items() if key in allowed_fields
    }
    return ArenaConfig(**filtered_config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint on the fixed dungeon campaign.")
    parser.add_argument("--checkpoint", type=str, default="artifacts/checkpoints/best_policy.pt")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--device", type=str, default=None)
    return parser


def greedy_actions(policy: ActorCriticPolicy, observations: np.ndarray, alive: np.ndarray, device: torch.device) -> np.ndarray:
    actions = np.full(observations.shape[0], 11, dtype=np.int64)
    for agent_id in np.flatnonzero(alive):
        obs_tensor = torch.from_numpy(observations[agent_id]).float().unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = policy(obs_tensor)
        actions[agent_id] = int(torch.argmax(logits, dim=-1).item())
    return actions


def main() -> None:
    args = build_parser().parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = config_from_checkpoint(checkpoint)
    device = torch.device(args.device or config.device)
    if config.environment_name != "dungeon_crawler_training_v2":
        raise SystemExit(
            "checkpoint environment does not match the dungeon crawler branch; "
            "train a new checkpoint or pass a dungeon checkpoint explicitly"
        )
    policy = ActorCriticPolicy(config.observation_dim, config.action_size, config.hidden_size).to(device)
    try:
        policy.load_state_dict(checkpoint["policy_state"])
    except RuntimeError as exc:
        raise SystemExit(
            "checkpoint policy shape does not match the current dungeon config; "
            "train a fresh checkpoint for this branch"
        ) from exc
    policy.eval()
    env = ArenaEnvironment(config)

    rewards: list[float] = []
    floor_reached: list[float] = []
    bosses: list[float] = []
    chests: list[float] = []
    powerups: list[float] = []
    damage_powerups: list[float] = []
    range_powerups: list[float] = []
    speed_powerups: list[float] = []
    diagonal_powerups: list[float] = []
    vitality_powerups: list[float] = []
    gate_distance: list[float] = []
    best_gate_distance: list[float] = []
    gate_tile_visits: list[float] = []
    use_gate_attempts: list[float] = []
    invalid_use_gate_attempts: list[float] = []
    victories: list[float] = []

    for episode in range(args.episodes):
        observations = env.reset(seed=config.seed + episode)
        done = False
        while not done:
            actions = greedy_actions(policy, observations, env.alive.copy(), device)
            step_batch = env.step(actions)
            observations = step_batch.observations
            done = bool(step_batch.info["episode_done"])
        metrics = env.get_agent_metrics()
        rewards.append(stats.mean(metric.reward for metric in metrics))
        floor_reached.append(stats.mean(metric.floor_reached for metric in metrics))
        bosses.append(stats.mean(metric.bosses_defeated for metric in metrics))
        chests.append(stats.mean(metric.chests_opened for metric in metrics))
        powerups.append(stats.mean(metric.powerups_picked for metric in metrics))
        damage_powerups.append(stats.mean(metric.damage_powerups for metric in metrics))
        range_powerups.append(stats.mean(metric.range_powerups for metric in metrics))
        speed_powerups.append(stats.mean(metric.speed_powerups for metric in metrics))
        diagonal_powerups.append(stats.mean(metric.diagonal_powerups for metric in metrics))
        vitality_powerups.append(stats.mean(metric.vitality_powerups for metric in metrics))
        gate_distance.append(stats.mean(metric.gate_distance for metric in metrics))
        best_gate_distance.append(stats.mean(metric.best_gate_distance for metric in metrics))
        gate_tile_visits.append(stats.mean(metric.gate_tile_visits for metric in metrics))
        use_gate_attempts.append(stats.mean(metric.use_gate_attempts for metric in metrics))
        invalid_use_gate_attempts.append(stats.mean(metric.invalid_use_gate_attempts for metric in metrics))
        victories.append(stats.mean(metric.victory for metric in metrics))

    print(f"episodes={args.episodes}")
    print(f"reward={stats.mean(rewards):.2f}")
    print(f"floor_reached={stats.mean(floor_reached):.2f}")
    print(f"bosses_defeated={stats.mean(bosses):.2f}")
    print(f"chests_opened={stats.mean(chests):.2f}")
    print(f"powerups_picked={stats.mean(powerups):.2f}")
    print(
        "powerup_breakdown="
        f"D:{stats.mean(damage_powerups):.2f} "
        f"R:{stats.mean(range_powerups):.2f} "
        f"S:{stats.mean(speed_powerups):.2f} "
        f"X:{stats.mean(diagonal_powerups):.2f} "
        f"V:{stats.mean(vitality_powerups):.2f}"
    )
    print(f"gate_distance={stats.mean(gate_distance):.2f}")
    print(f"best_gate_distance={stats.mean(best_gate_distance):.2f}")
    print(f"gate_tile_visits={stats.mean(gate_tile_visits):.2f}")
    print(f"use_gate_attempts={stats.mean(use_gate_attempts):.2f}")
    print(f"invalid_use_gate_attempts={stats.mean(invalid_use_gate_attempts):.2f}")
    print(f"victory_rate={stats.mean(victories):.2f}")


if __name__ == "__main__":
    main()
