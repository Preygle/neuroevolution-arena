from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from behavior_mutation_arena.config import ArenaConfig
from behavior_mutation_arena.core.replay import load_replay
from behavior_mutation_arena.visual.renderer import ArenaRenderer


def main() -> None:
    replay_path = Path("artifacts/replays/best_replay.pkl")
    record = load_replay(replay_path)
    renderer = ArenaRenderer(ArenaConfig())
    try:
        renderer.play_replay(record)
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
