from __future__ import annotations

import json
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from behavior_mutation_arena.config import ArenaConfig
from behavior_mutation_arena.core.models import GenerationSummary
from behavior_mutation_arena.core.replay import load_replay


class DiscordNotifier:
    def __init__(
        self,
        webhook_url: str | None,
        metric_interval: int = 0,
        video_interval: int = 0,
        video_checkpoint_interval: int = 0,
        video_fps: int = 12,
        video_max_frames: int = 360,
        video_cell_size: int = 10,
        video_keep: str = "latest",
        max_upload_mb: float = 24.0,
        timeout_seconds: float = 20.0,
    ) -> None:
        self.webhook_url = (webhook_url or "").strip()
        self.metric_interval = max(0, int(metric_interval))
        self.video_interval = max(0, int(video_interval))
        self.video_checkpoint_interval = max(0, int(video_checkpoint_interval))
        self.video_fps = max(1, int(video_fps))
        self.video_max_frames = max(1, int(video_max_frames))
        self.video_cell_size = max(4, int(video_cell_size))
        self.video_keep = video_keep
        self.max_upload_bytes = int(max(1.0, max_upload_mb) * 1024 * 1024)
        self.timeout_seconds = timeout_seconds
        self.last_video_generation: int | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.webhook_url)

    def maybe_send_generation(self, summary: GenerationSummary, total_generations: int, run_name: str) -> None:
        if not self.enabled or self.metric_interval <= 0:
            return
        completed = summary.generation + 1
        if completed % self.metric_interval != 0:
            return
        self._safe_call(self._send_generation, summary, total_generations, run_name)

    def maybe_send_generation_replay(
        self,
        completed_generations: int,
        total_generations: int,
        replay_path: Path,
        artifact_dir: Path,
        config: ArenaConfig,
        run_name: str,
    ) -> None:
        if not self.enabled or self.video_interval <= 0:
            return
        if completed_generations <= 0 or completed_generations % self.video_interval != 0:
            return
        if self.last_video_generation == completed_generations:
            return
        self.last_video_generation = completed_generations
        self._safe_call(
            self._send_checkpoint_replay,
            completed_generations,
            total_generations,
            replay_path,
            artifact_dir,
            config,
            run_name,
        )

    def maybe_send_checkpoint_replay(
        self,
        completed_generations: int,
        total_generations: int,
        checkpoint_interval: int,
        replay_path: Path,
        artifact_dir: Path,
        config: ArenaConfig,
        run_name: str,
    ) -> None:
        if self.video_interval > 0:
            return
        if not self.enabled or self.video_checkpoint_interval <= 0:
            return
        if checkpoint_interval <= 0:
            return
        cadence = checkpoint_interval * self.video_checkpoint_interval
        if completed_generations <= 0 or completed_generations % cadence != 0:
            return
        if self.last_video_generation == completed_generations:
            return
        self.last_video_generation = completed_generations
        self._safe_call(
            self._send_checkpoint_replay,
            completed_generations,
            total_generations,
            replay_path,
            artifact_dir,
            config,
            run_name,
        )

    def _send_generation(self, summary: GenerationSummary, total_generations: int, run_name: str) -> None:
        progress = 100.0 * (summary.generation + 1) / max(1, total_generations)
        embed = {
            "title": f"{run_name} training update",
            "description": f"Generation {summary.generation + 1}/{total_generations} ({progress:.1f}%)",
            "color": 0x2F80ED,
            "fields": [
                _field("Best fitness", f"{summary.best_fitness:.1f}", True),
                _field("Mean reward", f"{summary.mean_reward:.1f}", True),
                _field("Floor", f"{summary.mean_floor_reached:.2f}", True),
                _field("Boss damage", f"{summary.mean_boss_damage:.1f}", True),
                _field("Boss hits", f"{summary.mean_boss_hits:.2f}", True),
                _field("Miniboss", f"{summary.mean_miniboss_defeated:.3f}", True),
                _field("Chests", f"{summary.mean_chests_opened:.2f}", True),
                _field("Powerups", f"{summary.mean_powerups_picked:.2f}", True),
                _field("Bad gate", f"{summary.mean_invalid_use_gate_attempts:.2f}", True),
                _field("Survival", f"{summary.mean_survival:.1f}", True),
                _field("Success", f"{summary.success_rate * 100.0:.1f}%", True),
                _field("Champion", f"P{summary.champion_id}", True),
            ],
        }
        self._post_json({"embeds": [embed]})

    def _send_checkpoint_replay(
        self,
        completed_generations: int,
        total_generations: int,
        replay_path: Path,
        artifact_dir: Path,
        config: ArenaConfig,
        run_name: str,
    ) -> None:
        if not replay_path.exists():
            self._post_json({"content": f"`{run_name}` checkpoint {completed_generations}: no best replay exists yet."})
            return
        replay = load_replay(replay_path)
        from behavior_mutation_arena.visual.replay_video import render_replay_gif

        output_path, delete_after_send = self._video_output_path(artifact_dir, completed_generations)
        render_replay_gif(
            replay,
            output_path,
            config,
            fps=self.video_fps,
            max_frames=self.video_max_frames,
            cell_size=self.video_cell_size,
        )
        size = output_path.stat().st_size
        if size > self.max_upload_bytes:
            self._post_json(
                {
                    "content": (
                        f"`{run_name}` replay GIF at checkpoint {completed_generations}/{total_generations} "
                        f"is {size / 1024 / 1024:.1f} MB, above upload cap "
                        f"{self.max_upload_bytes / 1024 / 1024:.1f} MB. "
                        "Lower --discord-video-max-frames or --discord-video-cell-size."
                    )
                }
            )
        else:
            payload = {
                "content": (
                    f"`{run_name}` best replay at checkpoint {completed_generations}/{total_generations} | "
                    f"best gen {replay.generation} | champion P{replay.champion_id} | fitness {replay.fitness:.1f}"
                )
            }
            self._post_file(payload, output_path, "image/gif")
        if delete_after_send:
            output_path.unlink(missing_ok=True)

    def _video_output_path(self, artifact_dir: Path, completed_generations: int) -> tuple[Path, bool]:
        media_dir = artifact_dir / "discord"
        media_dir.mkdir(parents=True, exist_ok=True)
        if self.video_keep == "none":
            handle = tempfile.NamedTemporaryFile(prefix="best_replay_", suffix=".gif", dir=media_dir, delete=False)
            handle.close()
            return Path(handle.name), True
        if self.video_keep == "all":
            return media_dir / f"best_replay_checkpoint_{completed_generations:06d}.gif", False
        return media_dir / "latest_best_replay.gif", False

    def _post_json(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.webhook_url,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "BehaviorMutationArena/discord"},
            method="POST",
        )
        self._open_request(request)

    def _post_file(self, payload: dict[str, Any], file_path: Path, mime_type: str) -> None:
        boundary = f"----arena-{uuid.uuid4().hex}"
        file_bytes = file_path.read_bytes()
        body = bytearray()
        self._append_multipart_field(body, boundary, "payload_json", json.dumps(payload), "application/json")
        self._append_multipart_file(body, boundary, "files[0]", file_path.name, file_bytes, mime_type)
        body.extend(f"--{boundary}--\r\n".encode("utf-8"))
        request = urllib.request.Request(
            self.webhook_url,
            data=bytes(body),
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "User-Agent": "BehaviorMutationArena/discord",
            },
            method="POST",
        )
        self._open_request(request)

    def _open_request(self, request: urllib.request.Request) -> None:
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                retry_after = _retry_after_seconds(exc)
                time.sleep(min(10.0, retry_after))
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    response.read()
                return
            raise

    def _safe_call(self, func: Any, *args: Any) -> None:
        try:
            func(*args)
        except Exception as exc:
            print(f"discord telemetry skipped: {exc}")

    @staticmethod
    def _append_multipart_field(
        body: bytearray,
        boundary: str,
        name: str,
        value: str,
        content_type: str,
    ) -> None:
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n'.encode("utf-8"))
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        body.extend(value.encode("utf-8"))
        body.extend(b"\r\n")

    @staticmethod
    def _append_multipart_file(
        body: bytearray,
        boundary: str,
        name: str,
        filename: str,
        value: bytes,
        content_type: str,
    ) -> None:
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode("utf-8")
        )
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        body.extend(value)
        body.extend(b"\r\n")


def _field(name: str, value: str, inline: bool) -> dict[str, Any]:
    return {"name": name, "value": value, "inline": inline}


def _retry_after_seconds(exc: urllib.error.HTTPError) -> float:
    try:
        payload = json.loads(exc.read().decode("utf-8"))
        return float(payload.get("retry_after", 1.0))
    except Exception:
        return 1.0
