from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from behavior_mutation_arena.config import ArenaConfig
from behavior_mutation_arena.core.models import GenerationSummary
from behavior_mutation_arena.core.replay import ReplayRecord, save_replay
from behavior_mutation_arena.evolution.engine import EvolutionEngine
from behavior_mutation_arena.interface.backend import build_backend
from behavior_mutation_arena.rl.buffer import RolloutBuffer
from behavior_mutation_arena.rl.ppo import PPOPolicyBank
from behavior_mutation_arena.visual.plots import plot_training_metrics

if TYPE_CHECKING:
    from behavior_mutation_arena.visual.renderer import ArenaRenderer


class ArenaSimulation:
    def __init__(
        self,
        config: ArenaConfig,
        backend: str = "python",
        seed: int | None = None,
        render: bool = False,
        artifact_dir: str | Path = "artifacts",
    ) -> None:
        self.config = config
        self.seed = config.seed if seed is None else seed
        self.backend_name = backend
        self.artifact_dir = Path(artifact_dir)
        self.checkpoint_dir = self.artifact_dir / "checkpoints"
        self.plot_dir = self.artifact_dir / "plots"
        self.replay_dir = self.artifact_dir / "replays"
        self.metrics_path = self.artifact_dir / "metrics.csv"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.replay_dir.mkdir(parents=True, exist_ok=True)
        self.environment = build_backend(config, backend=backend, seed=self.seed)
        self.policy_bank = PPOPolicyBank(config)
        self.evolution = EvolutionEngine(config)
        self.renderer: ArenaRenderer | None = None
        if render:
            from behavior_mutation_arena.visual.renderer import ArenaRenderer

            self.renderer = ArenaRenderer(config)
        self.history: list[GenerationSummary] = []
        self.best_fitness = float("-inf")

    def train(self, generations: int) -> list[GenerationSummary]:
        for generation in range(generations):
            observations = self.environment.reset(seed=self.seed + generation)
            buffers = [RolloutBuffer() for _ in range(self.config.population_size)]
            frames: list[dict[str, np.ndarray | int]] = [self.environment.snapshot()]
            episode_done = False

            while not episode_done:
                active_mask = self.environment.alive.copy()
                actions, log_probs, values = self.policy_bank.act(observations, active_mask)
                step_batch = self.environment.step(actions)
                dones = step_batch.terminated.copy()

                for agent_id in np.flatnonzero(active_mask):
                    buffers[agent_id].add(
                        observation=observations[agent_id],
                        action=int(actions[agent_id]),
                        log_prob=float(log_probs[agent_id]),
                        reward=float(step_batch.rewards[agent_id]),
                        value=float(values[agent_id]),
                        done=bool(dones[agent_id]),
                    )

                observations = step_batch.observations
                frame = self.environment.snapshot()
                frames.append(frame)
                episode_done = bool(step_batch.info["episode_done"])

                if self.renderer is not None:
                    overlay = [
                        f"Generation {generation + 1}",
                        f"Step {step_batch.info['current_step']}/{step_batch.info['episode_step_limit']}",
                        f"Alive {step_batch.info['alive_count']}",
                    ]
                    self.renderer.draw(frame, overlay)

            last_values = self.policy_bank.evaluate_values(observations, self.environment.alive.copy())
            self.policy_bank.update(buffers, last_values)

            metrics = self.environment.get_agent_metrics()
            elite_ids = self.evolution.evolve(self.policy_bank, metrics)
            fitness = np.asarray([metric.fitness for metric in metrics], dtype=np.float32)
            rewards = np.asarray([metric.reward for metric in metrics], dtype=np.float32)
            survival = np.asarray([metric.survival_steps for metric in metrics], dtype=np.float32)
            kills = np.asarray([metric.kills for metric in metrics], dtype=np.float32)
            champion_id = int(np.argmax(fitness))
            summary = GenerationSummary(
                generation=generation,
                best_fitness=float(fitness.max()),
                mean_fitness=float(fitness.mean()),
                mean_reward=float(rewards.mean()),
                mean_survival=float(survival.mean()),
                mean_kills=float(kills.mean()),
                champion_id=champion_id,
                elite_ids=elite_ids,
            )
            self.history.append(summary)

            if summary.best_fitness > self.best_fitness:
                self.best_fitness = summary.best_fitness
                checkpoint = self.policy_bank.checkpoint_payload(champion_id=champion_id, generation=generation)
                torch.save(checkpoint, self.checkpoint_dir / "best_policy.pt")
                replay = ReplayRecord(
                    generation=generation,
                    champion_id=champion_id,
                    fitness=summary.best_fitness,
                    frames=frames,
                )
                save_replay(self.replay_dir / "best_replay.pkl", replay)

        self._write_metrics()
        plot_training_metrics(self.history, self.plot_dir / "training_metrics.png")
        if self.renderer is not None:
            self.renderer.close()
        return self.history

    def _write_metrics(self) -> None:
        with self.metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "generation",
                    "best_fitness",
                    "mean_fitness",
                    "mean_reward",
                    "mean_survival",
                    "mean_kills",
                    "champion_id",
                    "elite_ids",
                ]
            )
            for row in self.history:
                writer.writerow(
                    [
                        row.generation,
                        row.best_fitness,
                        row.mean_fitness,
                        row.mean_reward,
                        row.mean_survival,
                        row.mean_kills,
                        row.champion_id,
                        " ".join(str(elite_id) for elite_id in row.elite_ids),
                    ]
                )
