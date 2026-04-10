# Native backend slot

This folder is the pybind11 bridge point for moving `behavior_mutation_arena.core.environment` into C++ without rewriting PPO, mutation, rendering, or orchestration.

Target native API:

- `create_backend(config: dict, seed: int | None)`
- `reset() -> np.ndarray`
- `step(actions: np.ndarray) -> StepBatch-like payload`
- `snapshot() -> dict`

The current Python code already routes backend creation through `behavior_mutation_arena.interface.backend.build_backend()`, so a future native backend can be dropped in behind the same contract.

