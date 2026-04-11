from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from behavior_mutation_arena.config import ArenaConfig
from behavior_mutation_arena.interface.api import ArenaSimulation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Behavior Mutation Arena agents.")
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--backend", choices=["python", "native"], default="python")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = ArenaConfig(seed=args.seed, device=device)
    simulation = ArenaSimulation(
        config=config,
        backend=args.backend,
        seed=args.seed,
        render=args.render,
        checkpoint_interval=args.checkpoint_interval,
        scratch=args.scratch,
    )
    history = simulation.train(args.generations)
    if not history:
        print(f"nothing to run: checkpoint already covers generations={args.generations}")
        return
    final = history[-1]
    print(
        f"start_generation={simulation.start_generation} "
        f"target_generations={args.generations} "
        f"scratch={args.scratch}"
    )
    print(
        f"finished generations={args.generations} "
        f"best_fitness={max(entry.best_fitness for entry in history):.2f} "
        f"last_mean_fitness={final.mean_fitness:.2f}"
    )


if __name__ == "__main__":
    main()
