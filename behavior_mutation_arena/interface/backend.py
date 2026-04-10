from __future__ import annotations

from behavior_mutation_arena.config import ArenaConfig
from behavior_mutation_arena.core.environment import ArenaEnvironment


def build_backend(config: ArenaConfig, backend: str = "python", seed: int | None = None) -> ArenaEnvironment:
    if backend == "python":
        return ArenaEnvironment(config, seed=seed)
    if backend == "native":
        try:
            import arena_native  # type: ignore
        except ImportError as error:
            raise RuntimeError("Native backend requested, but the pybind11 module is not built.") from error
        if not hasattr(arena_native, "create_backend"):
            raise RuntimeError("Native backend scaffold exists, but create_backend() is not implemented yet.")
        return arena_native.create_backend(config.to_dict(), seed)
    raise ValueError(f"Unknown backend '{backend}'")

