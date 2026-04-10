from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ReplayRecord:
    generation: int
    champion_id: int
    fitness: float
    frames: list[dict[str, Any]]


def save_replay(path: str | Path, record: ReplayRecord) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump(record, handle)


def load_replay(path: str | Path) -> ReplayRecord:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)

