from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from behavior_mutation_arena.config import ArenaConfig
from behavior_mutation_arena.core.replay import load_replay


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="View the best replay from a training run.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--name",
        type=str,
        default=None,
        help="Training run name under artifacts/, for example: --name 32v5_seed7",
    )
    source.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Training artifact directory containing replays/best_replay.pkl.",
    )
    source.add_argument(
        "--replay",
        type=Path,
        default=None,
        help="Direct path to a best_replay.pkl file.",
    )
    parser.add_argument("--delay", type=float, default=0.08, help="Seconds between replay frames.")
    parser.add_argument(
        "--single-floor",
        action="store_true",
        help="Render only the active floor instead of the stitched 10-floor campaign map.",
    )
    parser.add_argument(
        "--overview",
        action="store_true",
        help="Show the full 10-floor stitched map instead of zooming around the active floor.",
    )
    parser.add_argument(
        "--context-floors",
        type=int,
        default=1,
        help="How many previous/next floors to include in zoomed stitched mode.",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List artifact folders that contain replays/best_replay.pkl.",
    )
    return parser


def resolve_replay_path(args: argparse.Namespace) -> Path:
    if args.replay is not None:
        return args.replay
    if args.artifact_dir is not None:
        return args.artifact_dir / "replays" / "best_replay.pkl"
    if args.name is not None:
        return ROOT / "artifacts" / args.name / "replays" / "best_replay.pkl"
    return ROOT / "artifacts" / "replays" / "best_replay.pkl"


def list_runs() -> None:
    artifacts_dir = ROOT / "artifacts"
    if not artifacts_dir.exists():
        print("No artifacts directory found.")
        return
    found = []
    for path in sorted(artifacts_dir.iterdir()):
        if not path.is_dir():
            continue
        replay_path = path / "replays" / "best_replay.pkl"
        if replay_path.exists():
            found.append(path.name)
    if not found:
        print("No training runs with replays/best_replay.pkl found.")
        return
    print("Training runs with best replays:")
    for name in found:
        print(f"  {name}")


def main() -> None:
    args = build_parser().parse_args()
    if args.list_runs:
        list_runs()
        return

    replay_path = resolve_replay_path(args)
    if not replay_path.exists():
        raise SystemExit(
            f"best replay not found: {replay_path}\n"
            "Use --list-runs to see runs that have a saved best replay."
        )

    record = load_replay(replay_path)
    config = ArenaConfig(
        render_stitched_dungeon=not args.single_floor,
        render_stitched_focus_current=(not args.single_floor and not args.overview),
        render_stitched_context_floors=max(0, args.context_floors),
    )

    from behavior_mutation_arena.visual.renderer import ArenaRenderer

    renderer = ArenaRenderer(config)
    try:
        renderer.play_replay(record, delay=max(0.0, args.delay))
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
