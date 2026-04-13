from __future__ import annotations

import csv
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from behavior_mutation_arena.core.models import GenerationSummary
from behavior_mutation_arena.visual.plots import plot_training_metrics


def main() -> None:
    metrics_path = Path("artifacts/metrics.csv")
    history: list[GenerationSummary] = []
    with metrics_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            history.append(
                GenerationSummary(
                    generation=int(row["generation"]),
                    best_fitness=float(row["best_fitness"]),
                    mean_fitness=float(row["mean_fitness"]),
                    mean_reward=float(row["mean_reward"]),
                    mean_survival=float(row["mean_survival"]),
                    mean_kills=float(row["mean_kills"]),
                    champion_id=int(row["champion_id"]),
                    elite_ids=[int(value) for value in row["elite_ids"].split() if value],
                    mean_exploration=float(row.get("mean_exploration", 0.0)),
                    mean_damage=float(row.get("mean_damage", 0.0)),
                    mean_camping=float(row.get("mean_camping", 0.0)),
                )
            )
    plot_training_metrics(history, "artifacts/plots/training_metrics.png")


if __name__ == "__main__":
    main()
