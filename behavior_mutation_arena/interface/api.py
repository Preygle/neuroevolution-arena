from __future__ import annotations

import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from behavior_mutation_arena.config import ArenaConfig
from behavior_mutation_arena.core.models import AgentMetrics, GenerationSummary, StepBatch
from behavior_mutation_arena.core.replay import ReplayRecord, save_replay
from behavior_mutation_arena.evolution.engine import EvolutionEngine
from behavior_mutation_arena.interface.backend import build_backend
from behavior_mutation_arena.rl.buffer import RolloutBatch, RolloutBuffer
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
        checkpoint_interval: int = 10,
        scratch: bool = False,
    ) -> None:
        self.config = config
        self.seed = config.seed if seed is None else seed
        self.backend_name = backend
        self.checkpoint_interval = checkpoint_interval
        self.scratch = scratch
        self.instance_count = max(1, min(config.instance_count, config.max_instance_count))
        self.render_instance_count = min(self.instance_count, config.max_render_instance_count)
        self.worker_count = max(
            1,
            min(
                self.instance_count,
                config.env_worker_count if config.env_worker_count > 0 else (os.cpu_count() or 1),
            ),
        )
        self.executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=self.worker_count) if self.worker_count > 1 else None
        )
        self.artifact_dir = Path(artifact_dir)
        self.checkpoint_dir = self.artifact_dir / "checkpoints"
        self.plot_dir = self.artifact_dir / "plots"
        self.replay_dir = self.artifact_dir / "replays"
        self.metrics_path = self.artifact_dir / "metrics.csv"
        self.progress_checkpoint_path = self.checkpoint_dir / "training_state.pt"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.replay_dir.mkdir(parents=True, exist_ok=True)
        self.environments = [
            build_backend(config, backend=backend, seed=self.seed + instance_id * 1000)
            for instance_id in range(self.instance_count)
        ]
        self.environment = self.environments[0]
        self.policy_bank = PPOPolicyBank(config)
        self.evolution = EvolutionEngine(config)
        self.renderer: ArenaRenderer | None = None
        if render:
            from behavior_mutation_arena.visual.renderer import ArenaRenderer

            self.renderer = ArenaRenderer(config, instance_count=self.render_instance_count)
        self.history: list[GenerationSummary] = []
        self.best_fitness = float("-inf")
        self.start_generation = 0
        if not scratch:
            self._load_progress_checkpoint()

    def train(self, generations: int) -> list[GenerationSummary]:
        if self.start_generation >= generations:
            self._write_metrics()
            plot_training_metrics(self.history, self.plot_dir / "training_metrics.png")
            self._shutdown()
            return self.history

        completed_generations = self.start_generation
        self._print_training_banner(generations)
        try:
            for generation in range(self.start_generation, generations):
                _gen_start = time.perf_counter()
                for environment in self.environments:
                    if hasattr(environment, "set_generation"):
                        environment.set_generation(generation)

                reset_seeds = [
                    self.seed + generation * 1009 + instance_id * 97 for instance_id in range(self.instance_count)
                ]
                observations = self._reset_environments(reset_seeds)
                buffers = [
                    [RolloutBuffer() for _ in range(self.config.population_size)]
                    for _ in range(self.instance_count)
                ]
                done = [False] * self.instance_count
                final_step_batches: list[StepBatch | None] = [None] * self.instance_count

                while not all(done):
                    active_instance_ids = [instance_id for instance_id, is_done in enumerate(done) if not is_done]
                    active_observations = [observations[instance_id] for instance_id in active_instance_ids]
                    active_masks = [self.environments[instance_id].alive.copy() for instance_id in active_instance_ids]
                    actions_by_instance, log_probs_by_instance, values_by_instance = self.policy_bank.act_many(
                        active_observations,
                        active_masks,
                    )
                    step_batches = self._step_environments(active_instance_ids, actions_by_instance)

                    for slot, instance_id in enumerate(active_instance_ids):
                        step_batch = step_batches[slot]
                        dones = step_batch.terminated | step_batch.truncated
                        active_mask = active_masks[slot]
                        for agent_id in np.flatnonzero(active_mask):
                            buffers[instance_id][agent_id].add(
                                observation=observations[instance_id][agent_id],
                                action=int(actions_by_instance[slot][agent_id]),
                                log_prob=float(log_probs_by_instance[slot][agent_id]),
                                reward=float(step_batch.rewards[agent_id]),
                                value=float(values_by_instance[slot][agent_id]),
                                done=bool(dones[agent_id]),
                            )
                        observations[instance_id] = step_batch.observations
                        done[instance_id] = bool(step_batch.info["episode_done"])
                        final_step_batches[instance_id] = step_batch

                    if self.renderer is not None:
                        visible_instance_ids = self.renderer.visible_instance_indices(self.render_instance_count)
                        visible_snapshots = self._snapshot_environments(visible_instance_ids)
                        overlay_groups = [
                            self._overlay_lines(generation, instance_id, final_step_batches[instance_id])
                            for instance_id in visible_instance_ids
                        ]
                        page, total_pages = self.renderer.page_status(self.render_instance_count)
                        self.renderer.draw_many(
                            visible_snapshots,
                            overlay_groups,
                            header_lines=[
                                f"Generation {generation + 1}",
                                f"Simulated {self.instance_count} | Workers {self.worker_count}",
                                f"Render page {page}/{total_pages}",
                            ],
                        )

                merged_batches = self._merge_instance_batches(buffers, observations, final_step_batches)
                self.policy_bank.update_from_batches(merged_batches)

                metrics_by_instance = self._collect_metrics()
                metrics = self._aggregate_metrics(metrics_by_instance)
                fitness = np.asarray([metric.fitness for metric in metrics], dtype=np.float32)
                rewards = np.asarray([metric.reward for metric in metrics], dtype=np.float32)
                survival = np.asarray([metric.survival_steps for metric in metrics], dtype=np.float32)
                floor_reached = np.asarray([metric.floor_reached for metric in metrics], dtype=np.float32)
                bosses = np.asarray([metric.bosses_defeated for metric in metrics], dtype=np.float32)
                chests = np.asarray([metric.chests_opened for metric in metrics], dtype=np.float32)
                damage = np.asarray([metric.damage_dealt for metric in metrics], dtype=np.float32)
                gate_distance = np.asarray([metric.gate_distance for metric in metrics], dtype=np.float32)
                victories = np.asarray([metric.victory for metric in metrics], dtype=np.float32)
                champion_id = int(np.argmax(fitness))

                if float(fitness.max()) > self.best_fitness:
                    self.best_fitness = float(fitness.max())
                    checkpoint = self.policy_bank.checkpoint_payload(champion_id=champion_id, generation=generation)
                    torch.save(checkpoint, self.checkpoint_dir / "best_policy.pt")
                    replay = self._record_replay(generation, champion_id, float(fitness.max()))
                    save_replay(self.replay_dir / "best_replay.pkl", replay)

                elite_ids = self.evolution.evolve(self.policy_bank, metrics)
                summary = GenerationSummary(
                    generation=generation,
                    best_fitness=float(fitness.max()),
                    mean_fitness=float(fitness.mean()),
                    mean_reward=float(rewards.mean()),
                    mean_survival=float(survival.mean()),
                    mean_floor_reached=float(floor_reached.mean()),
                    mean_bosses_defeated=float(bosses.mean()),
                    mean_chests_opened=float(chests.mean()),
                    mean_damage=float(damage.mean()),
                    mean_gate_distance=float(gate_distance.mean()),
                    success_rate=float(victories.mean()),
                    champion_id=champion_id,
                    elite_ids=elite_ids,
                )
                self.history.append(summary)
                _gen_elapsed = time.perf_counter() - _gen_start
                self._print_generation_progress(summary, generations, _gen_elapsed)

                completed_generations = generation + 1
                if completed_generations % self.checkpoint_interval == 0:
                    self._save_progress_checkpoint(completed_generations)
        except KeyboardInterrupt:
            self._save_progress_checkpoint(completed_generations)
            self._write_metrics()
            plot_training_metrics(self.history, self.plot_dir / "training_metrics.png")
            self._shutdown()
            raise

        self._save_progress_checkpoint(generations)
        self._write_metrics()
        plot_training_metrics(self.history, self.plot_dir / "training_metrics.png")
        self._shutdown()
        return self.history

    def _overlay_lines(self, generation: int, instance_id: int, step_batch: StepBatch | None) -> list[str]:
        if step_batch is None:
            return [f"Instance {instance_id + 1}", "Waiting for first step"]
        return [
            f"Instance {instance_id + 1}",
            f"Step {step_batch.info['current_step']}/{step_batch.info['episode_step_limit']}",
            f"Floor {step_batch.info['floor_index']}: {step_batch.info['floor_name']}",
            f"Alive {step_batch.info['alive_count']}",
            f"Gate {step_batch.info['closest_gate_distance']}",
            f"Idle {step_batch.info['steps_since_progress']}",
        ]

    def _reset_environments(self, seeds: list[int]) -> list[np.ndarray]:
        if self.executor is None:
            return [environment.reset(seed=seed) for environment, seed in zip(self.environments, seeds)]
        futures = [
            self.executor.submit(environment.reset, seed=seed)
            for environment, seed in zip(self.environments, seeds)
        ]
        return [future.result() for future in futures]

    def _step_environments(
        self,
        instance_ids: list[int],
        actions_by_instance: list[np.ndarray],
    ) -> list[StepBatch]:
        if self.executor is None:
            return [
                self.environments[instance_id].step(actions_by_instance[slot])
                for slot, instance_id in enumerate(instance_ids)
            ]
        futures = [
            self.executor.submit(self.environments[instance_id].step, actions_by_instance[slot])
            for slot, instance_id in enumerate(instance_ids)
        ]
        return [future.result() for future in futures]

    def _snapshot_environments(self, instance_ids: list[int]) -> list[dict[str, object]]:
        if self.executor is None:
            return [self.environments[instance_id].snapshot() for instance_id in instance_ids]
        futures = [self.executor.submit(self.environments[instance_id].snapshot) for instance_id in instance_ids]
        return [future.result() for future in futures]

    def _collect_metrics(self) -> list[list[AgentMetrics]]:
        if self.executor is None:
            return [environment.get_agent_metrics() for environment in self.environments]
        futures = [self.executor.submit(environment.get_agent_metrics) for environment in self.environments]
        return [future.result() for future in futures]

    def _merge_instance_batches(
        self,
        buffers: list[list[RolloutBuffer]],
        observations: list[np.ndarray],
        final_step_batches: list[StepBatch | None],
    ) -> list[RolloutBatch | None]:
        active_masks = [environment.alive.copy() for environment in self.environments]
        last_values_by_instance = self.policy_bank.evaluate_values_many(observations, active_masks)
        merged_batches: list[RolloutBatch | None] = []
        for agent_id in range(self.config.population_size):
            agent_batches: list[RolloutBatch] = []
            for instance_id in range(self.instance_count):
                final_step_batch = final_step_batches[instance_id]
                if final_step_batch is None:
                    continue
                done_mask = final_step_batch.terminated | final_step_batch.truncated
                batch = buffers[instance_id][agent_id].finish(
                    last_value=0.0 if bool(done_mask[agent_id]) else float(last_values_by_instance[instance_id][agent_id]),
                    gamma=self.config.ppo_gamma,
                    gae_lambda=self.config.ppo_lambda,
                )
                if batch is not None:
                    agent_batches.append(batch)
            merged_batches.append(RolloutBatch.concatenate(agent_batches))
        return merged_batches

    def _aggregate_metrics(self, metrics_by_instance: list[list[AgentMetrics]]) -> list[AgentMetrics]:
        aggregated: list[AgentMetrics] = []
        for agent_id in range(self.config.population_size):
            agent_metrics = [metrics[agent_id] for metrics in metrics_by_instance]
            aggregated.append(
                AgentMetrics(
                    agent_id=agent_id,
                    reward=float(np.mean([metric.reward for metric in agent_metrics])),
                    survival_steps=int(round(np.mean([metric.survival_steps for metric in agent_metrics]))),
                    floor_reached=int(round(np.mean([metric.floor_reached for metric in agent_metrics]))),
                    bosses_defeated=int(round(np.mean([metric.bosses_defeated for metric in agent_metrics]))),
                    chests_opened=int(round(np.mean([metric.chests_opened for metric in agent_metrics]))),
                    damage_dealt=float(np.mean([metric.damage_dealt for metric in agent_metrics])),
                    gate_distance=float(np.mean([metric.gate_distance for metric in agent_metrics])),
                    fitness=float(np.mean([metric.fitness for metric in agent_metrics])),
                    victory=int(round(np.mean([metric.victory for metric in agent_metrics]))),
                )
            )
        return aggregated

    def _record_replay(self, generation: int, champion_id: int, fitness: float) -> ReplayRecord:
        replay_environment = build_backend(
            self.config,
            backend=self.backend_name,
            seed=self.seed + generation * 2003 + 17,
        )
        if hasattr(replay_environment, "set_generation"):
            replay_environment.set_generation(generation)
        observations = replay_environment.reset(seed=self.seed + generation * 2003 + 17)
        frames: list[dict[str, object]] = [replay_environment.snapshot()]
        episode_done = False
        while not episode_done:
            active_mask = replay_environment.alive.copy()
            actions, _, _ = self.policy_bank.act(observations, active_mask)
            step_batch = replay_environment.step(actions)
            observations = step_batch.observations
            frames.append(replay_environment.snapshot())
            episode_done = bool(step_batch.info["episode_done"])
        return ReplayRecord(
            generation=generation,
            champion_id=champion_id,
            fitness=fitness,
            frames=frames,
        )

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
                    "mean_floor_reached",
                    "mean_bosses_defeated",
                    "mean_chests_opened",
                    "mean_damage",
                    "mean_gate_distance",
                    "success_rate",
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
                        row.mean_floor_reached,
                        row.mean_bosses_defeated,
                        row.mean_chests_opened,
                        row.mean_damage,
                        row.mean_gate_distance,
                        row.success_rate,
                        row.champion_id,
                        " ".join(str(elite_id) for elite_id in row.elite_ids),
                    ]
                )

    def _save_progress_checkpoint(self, completed_generations: int) -> None:
        checkpoint = {
            "completed_generations": completed_generations,
            "history": [summary.__dict__.copy() for summary in self.history],
            "best_fitness": self.best_fitness,
            "population_states": self.policy_bank.export_population_state(),
            "optimizer_states": self.policy_bank.export_optimizer_states(),
            "seed": self.seed,
            "backend": self.backend_name,
            "config": self.config.to_dict(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "evolution_rng_state": self.evolution.rng.bit_generator.state,
        }
        if torch.cuda.is_available():
            checkpoint["torch_cuda_rng_state"] = torch.cuda.get_rng_state_all()
        torch.save(checkpoint, self.progress_checkpoint_path)

    def _load_progress_checkpoint(self) -> None:
        if not self.progress_checkpoint_path.exists():
            return
        checkpoint = torch.load(self.progress_checkpoint_path, map_location="cpu", weights_only=False)
        completed_generations = int(checkpoint.get("completed_generations", 0))
        population_states = checkpoint.get("population_states")
        optimizer_states = checkpoint.get("optimizer_states")
        history_rows = checkpoint.get("history", [])
        checkpoint_config = checkpoint.get("config", {})

        if population_states is None or optimizer_states is None:
            return
        if checkpoint_config.get("environment_name") != self.config.environment_name:
            return
        checkpoint_population = int(checkpoint_config.get("population_size", -1))
        checkpoint_observation_dim = int(checkpoint_config.get("observation_dim", self.config.observation_dim))
        checkpoint_action_size = int(checkpoint_config.get("action_size", self.config.action_size))
        if checkpoint_population != self.config.population_size:
            return
        if checkpoint_observation_dim != self.config.observation_dim:
            return
        if checkpoint_action_size != self.config.action_size:
            return

        self.policy_bank.load_population(population_states)
        self.policy_bank.load_optimizer_states(optimizer_states)
        self.history = [
            GenerationSummary(
                generation=row["generation"],
                best_fitness=row["best_fitness"],
                mean_fitness=row["mean_fitness"],
                mean_reward=row["mean_reward"],
                mean_survival=row["mean_survival"],
                mean_floor_reached=row["mean_floor_reached"],
                mean_bosses_defeated=row["mean_bosses_defeated"],
                mean_chests_opened=row["mean_chests_opened"],
                mean_damage=row.get("mean_damage", 0.0),
                mean_gate_distance=row.get("mean_gate_distance", 0.0),
                success_rate=row.get("success_rate", 0.0),
                champion_id=row["champion_id"],
                elite_ids=row["elite_ids"],
            )
            for row in history_rows
        ]
        self.best_fitness = float(checkpoint.get("best_fitness", float("-inf")))
        self.start_generation = completed_generations

        numpy_rng_state = checkpoint.get("numpy_rng_state")
        if numpy_rng_state is not None:
            np.random.set_state(numpy_rng_state)
        torch_rng_state = checkpoint.get("torch_rng_state")
        if torch_rng_state is not None:
            torch.set_rng_state(torch_rng_state)
        cuda_rng_state = checkpoint.get("torch_cuda_rng_state")
        if cuda_rng_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_rng_state)
        evolution_rng_state = checkpoint.get("evolution_rng_state")
        if evolution_rng_state is not None:
            self.evolution.rng.bit_generator.state = evolution_rng_state

    def _shutdown(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None

    # ------------------------------------------------------------------
    # CLI output helpers
    # ------------------------------------------------------------------

    def _print_training_banner(self, total_generations: int) -> None:
        device = self.config.device
        sep = "=" * 64
        print(sep)
        print("  Neuroevolution Arena — Training")
        print(sep)
        print(f"  Generations     : {self.start_generation} → {total_generations}  "
              f"(remaining: {total_generations - self.start_generation})")
        print(f"  Instances       : {self.instance_count}  |  Workers: {self.worker_count}")
        print(f"  Population size : {self.config.population_size}")
        print(f"  Device          : {device}")
        print(f"  Seed            : {self.seed}")
        print(sep)
        # Column header — keep widths in sync with _print_generation_progress
        print(
            f"  {'Gen':>7}  {'BestFit':>9}  {'MeanFit':>9}  {'Floor':>5}  "
            f"{'Surv':>5}  {'Boss':>4}  {'Win%':>5}  {'Champ':>5}  {'s/gen':>6}"
        )
        print("  " + "-" * 62)

    def _print_generation_progress(
        self,
        summary: GenerationSummary,
        total_generations: int,
        elapsed: float,
    ) -> None:
        gen_str = f"{summary.generation + 1}/{total_generations}"
        win_pct = summary.success_rate * 100.0
        elite_str = ",".join(str(e) for e in summary.elite_ids)
        new_best = "★" if summary.best_fitness >= self.best_fitness else " "
        print(
            f"  {gen_str:>7}  "
            f"{summary.best_fitness:>9.1f}  "
            f"{summary.mean_fitness:>9.1f}  "
            f"{summary.mean_floor_reached:>5.2f}  "
            f"{summary.mean_survival:>5.0f}  "
            f"{summary.mean_bosses_defeated:>4.2f}  "
            f"{win_pct:>4.1f}%  "
            f"P{summary.champion_id:>3}  "
            f"{elapsed:>5.1f}s"
            f"  {new_best}"
        )
