from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from behavior_mutation_arena.core.models import GenerationSummary


def plot_training_metrics(history: list[GenerationSummary], output_path: str | Path) -> None:
    if not history:
        return
    generations = [entry.generation for entry in history]
    best_fitness = [entry.best_fitness for entry in history]
    mean_fitness = [entry.mean_fitness for entry in history]
    floor_reached = [entry.mean_floor_reached for entry in history]
    bosses = [entry.mean_bosses_defeated for entry in history]
    chests = [entry.mean_chests_opened for entry in history]
    gate_distance = [entry.mean_gate_distance for entry in history]
    best_gate_distance = [entry.mean_best_gate_distance for entry in history]
    gate_tile_visits = [entry.mean_gate_tile_visits for entry in history]
    success_rate = [entry.success_rate for entry in history]

    figure, axes = plt.subplots(5, 1, figsize=(10, 18), sharex=True)
    axes[0].plot(generations, best_fitness, label="best fitness", linewidth=2.2)
    axes[0].plot(generations, mean_fitness, label="mean fitness", linewidth=1.8)
    axes[0].set_ylabel("fitness")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(generations, floor_reached, color="#1f77b4", linewidth=2.0)
    axes[1].set_ylabel("floor")
    axes[1].grid(alpha=0.3)

    axes[2].plot(generations, bosses, color="#d62728", linewidth=2.0, label="bosses")
    axes[2].plot(generations, chests, color="#2ca02c", linewidth=1.8, label="chests")
    axes[2].set_ylabel("campaign")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    axes[3].plot(generations, gate_distance, color="#8c564b", linewidth=2.0, label="end gate")
    axes[3].plot(generations, best_gate_distance, color="#bcbd22", linewidth=1.8, label="best gate")
    axes[3].set_ylabel("gate dist")
    axes[3].legend()
    axes[3].grid(alpha=0.3)

    axes[4].plot(generations, success_rate, color="#9467bd", linewidth=2.0, label="success")
    axes[4].plot(generations, gate_tile_visits, color="#17becf", linewidth=1.8, label="gate tiles")
    axes[4].set_ylabel("success / gate")
    axes[4].set_xlabel("generation")
    axes[4].legend()
    axes[4].grid(alpha=0.3)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
