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
    mean_survival = [entry.mean_survival for entry in history]
    mean_kills = [entry.mean_kills for entry in history]
    mean_exploration = [entry.mean_exploration for entry in history]
    mean_camping = [entry.mean_camping for entry in history]

    figure, axes = plt.subplots(4, 1, figsize=(10, 15), sharex=True)
    axes[0].plot(generations, best_fitness, label="best fitness", linewidth=2.2)
    axes[0].plot(generations, mean_fitness, label="mean fitness", linewidth=1.8)
    axes[0].set_ylabel("fitness")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(generations, mean_survival, color="#1f77b4", linewidth=2.0)
    axes[1].set_ylabel("survival")
    axes[1].grid(alpha=0.3)

    axes[2].plot(generations, mean_kills, color="#d62728", linewidth=2.0)
    axes[2].set_ylabel("kills")
    axes[2].grid(alpha=0.3)

    axes[3].plot(generations, mean_exploration, color="#2ca02c", linewidth=2.0, label="exploration")
    axes[3].plot(generations, mean_camping, color="#9467bd", linewidth=1.8, label="camping")
    axes[3].set_ylabel("behavior")
    axes[3].set_xlabel("generation")
    axes[3].legend()
    axes[3].grid(alpha=0.3)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
