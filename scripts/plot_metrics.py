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
            row = {(key.lstrip("\ufeff") if key else key): value for key, value in row.items()}
            history.append(
                GenerationSummary(
                    generation=int(row["generation"]),
                    best_fitness=float(row["best_fitness"]),
                    mean_fitness=float(row["mean_fitness"]),
                    mean_reward=float(row["mean_reward"]),
                    mean_survival=float(row["mean_survival"]),
                    mean_floor_reached=float(row["mean_floor_reached"]),
                    mean_bosses_defeated=float(row["mean_bosses_defeated"]),
                    mean_chests_opened=float(row["mean_chests_opened"]),
                    mean_powerups_picked=float(row.get("mean_powerups_picked", row.get("mean_chests_opened", 0.0))),
                    mean_damage_powerups=float(row.get("mean_damage_powerups", 0.0)),
                    mean_range_powerups=float(row.get("mean_range_powerups", 0.0)),
                    mean_speed_powerups=float(row.get("mean_speed_powerups", 0.0)),
                    mean_diagonal_powerups=float(row.get("mean_diagonal_powerups", 0.0)),
                    mean_vitality_powerups=float(row.get("mean_vitality_powerups", 0.0)),
                    mean_damage=float(row["mean_damage"]),
                    mean_gate_distance=float(row.get("mean_gate_distance", 0.0)),
                    mean_best_gate_distance=float(row.get("mean_best_gate_distance", row.get("mean_gate_distance", 0.0))),
                    mean_gate_tile_visits=float(row.get("mean_gate_tile_visits", 0.0)),
                    mean_use_gate_attempts=float(row.get("mean_use_gate_attempts", 0.0)),
                    mean_invalid_use_gate_attempts=float(row.get("mean_invalid_use_gate_attempts", 0.0)),
                    mean_boss_damage=float(row.get("mean_boss_damage", 0.0)),
                    mean_boss_hits=float(row.get("mean_boss_hits", 0.0)),
                    mean_boss_health_remaining=float(row.get("mean_boss_health_remaining", 0.0)),
                    mean_floor5_entry_alive=float(row.get("mean_floor5_entry_alive", 0.0)),
                    mean_floor5_entry_power_score=float(row.get("mean_floor5_entry_power_score", 0.0)),
                    mean_alive_attack_bonus=float(row.get("mean_alive_attack_bonus", 0.0)),
                    mean_alive_range_bonus=float(row.get("mean_alive_range_bonus", 0.0)),
                    mean_powered_agent_deaths=float(row.get("mean_powered_agent_deaths", 0.0)),
                    mean_miniboss_defeated=float(row.get("mean_miniboss_defeated", 0.0)),
                    success_rate=float(row["success_rate"]),
                    champion_id=int(row["champion_id"]),
                    elite_ids=[int(value) for value in row["elite_ids"].split() if value],
                )
            )
    plot_training_metrics(history, "artifacts/plots/training_metrics.png")


if __name__ == "__main__":
    main()
