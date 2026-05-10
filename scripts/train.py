from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from behavior_mutation_arena.config import ArenaConfig
from behavior_mutation_arena.interface.api import ArenaSimulation


def build_parser() -> argparse.ArgumentParser:
    defaults = ArenaConfig()
    parser = argparse.ArgumentParser(description="Train the cooperative dungeon crawler team.")
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--backend", choices=["python", "native"], default="python")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--instances", type=int, default=defaults.instance_count)
    parser.add_argument("--render-instances", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument(
        "--discord-webhook-url",
        type=str,
        default=None,
        help="Discord webhook URL. Prefer DISCORD_WEBHOOK_URL so the secret is not in shell history.",
    )
    parser.add_argument(
        "--discord-interval",
        type=int,
        default=0,
        help="Post metric embeds every N generations. 0 disables Discord metrics.",
    )
    parser.add_argument(
        "--discord-video-checkpoints",
        type=int,
        default=0,
        help="Legacy option: post the current best replay once every N checkpoints. Ignored if --discord-video-interval is set.",
    )
    parser.add_argument(
        "--discord-video-interval",
        type=int,
        default=0,
        help="Post the current best replay every N generations. 0 disables generation-based replay uploads.",
    )
    parser.add_argument("--discord-video-fps", type=int, default=12)
    parser.add_argument("--discord-video-max-frames", type=int, default=360)
    parser.add_argument("--discord-video-cell-size", type=int, default=10)
    parser.add_argument(
        "--discord-video-keep",
        choices=["none", "latest", "all"],
        default="latest",
        help="Local replay GIF retention policy after Discord upload.",
    )
    parser.add_argument(
        "--discord-max-upload-mb",
        type=float,
        default=24.0,
        help="Skip Discord replay upload if the GIF is larger than this size.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Root directory for checkpoints, metrics, plots, replays. "
             "Defaults to artifacts/ (or artifacts/<name>/ if --name is given).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Short run name; creates artifacts/<name>/ so parallel runs don't clash.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = ArenaConfig(seed=args.seed, device=device)
    requested_instances = max(1, args.instances)
    if requested_instances > config.max_instance_count:
        print(
            f"requested instances={requested_instances} exceeds safe cap={config.max_instance_count}; "
            f"using {config.max_instance_count}"
        )
    config.instance_count = min(requested_instances, config.max_instance_count)
    requested_render_instances = (
        args.render_instances
        if args.render_instances is not None
        else min(config.instance_count, config.default_render_instance_count)
    )
    if requested_render_instances > config.max_render_instance_count:
        print(
            f"requested render instances={requested_render_instances} exceeds safe cap={config.max_render_instance_count}; "
            f"using {config.max_render_instance_count}"
        )
    config.max_render_instance_count = max(1, min(requested_render_instances, config.max_render_instance_count))
    # Resolve artifact directory
    if args.artifact_dir is not None:
        artifact_dir = args.artifact_dir
    elif args.name is not None:
        artifact_dir = Path("artifacts") / args.name
    else:
        artifact_dir = Path("artifacts")

    simulation = ArenaSimulation(
        config=config,
        backend=args.backend,
        seed=args.seed,
        render=args.render,
        artifact_dir=artifact_dir,
        checkpoint_interval=args.checkpoint_interval,
        scratch=args.scratch,
        discord_webhook_url=args.discord_webhook_url or os.getenv("DISCORD_WEBHOOK_URL"),
        discord_interval=args.discord_interval,
        discord_video_interval=args.discord_video_interval,
        discord_video_checkpoint_interval=args.discord_video_checkpoints,
        discord_video_fps=args.discord_video_fps,
        discord_video_max_frames=args.discord_video_max_frames,
        discord_video_cell_size=args.discord_video_cell_size,
        discord_video_keep=args.discord_video_keep,
        discord_max_upload_mb=args.discord_max_upload_mb,
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
