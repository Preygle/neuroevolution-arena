# Behavior Mutation Arena

Behavior Mutation Arena is a modular multi-agent reinforcement learning sandbox built around a 15x15 combat-and-foraging grid. The current implementation is a runnable Python-first baseline with a clean backend seam for a future pybind11 C++ environment core, while keeping PPO and evolutionary logic in Python.

## Project structure

```text
behavior_mutation_arena/
  config.py
  core/
    environment.py
    models.py
    replay.py
  rl/
    buffer.py
    policy.py
    ppo.py
  evolution/
    engine.py
  interface/
    api.py
    backend.py
  visual/
    plots.py
    renderer.py
docs/
  architecture.md
native/
  CMakeLists.txt
  README.md
  src/
    bindings.cpp
scripts/
  plot_metrics.py
  replay_best.py
  train.py
artifacts/
  checkpoints/
  plots/
  replays/
```

## Architecture

- `core.environment.ArenaEnvironment`: high-throughput grid simulation, combat, spawning, observation encoding, reward accounting
- `rl.policy.ActorCriticPolicy`: per-agent neural network policy/value model
- `rl.ppo.PPOPolicyBank`: PPO updates for the full 30-agent population
- `evolution.engine.EvolutionEngine`: elite selection, cloning, Gaussian mutation of policy weights
- `interface.api.ArenaSimulation`: generation loop, replay capture, checkpointing, metrics persistence
- `visual.renderer.ArenaRenderer`: live grid rendering and replay playback
- `visual.plots`: training metric export and plots

The environment core is intentionally separated behind `interface.backend.build_backend()`. That keeps the Python PPO stack stable when you later swap the simulation backend to C++ through pybind11.

## Current feature set

- 30 agents per generation on a 15x15 grid
- Random episode length between 100 and 300 steps
- Food, poison, melee, ranged, and rare weapon pickups
- Health, energy, inventory, durability, kills, reward, and survival tracking
- Accurate 3x3 observation window
- Noisy 5x5 observation window with configurable accuracy
- PPO updates during each generation
- Evolutionary replacement after each generation using elite selection and Gaussian mutation
- Live Pygame visualization
- Best-agent replay capture
- Fitness, survival, and kill-count plotting

## Run

Install in editable mode if you want package resolution from anywhere:

```powershell
python -m pip install -e .
```

Run training:

```powershell
python scripts/train.py --generations 20
```

Run with live rendering:

```powershell
python scripts/train.py --generations 10 --render
```

Replay the best recorded episode:

```powershell
python scripts/replay_best.py
```

Rebuild plots from CSV:

```powershell
python scripts/plot_metrics.py
```

Artifacts are written to:

- `artifacts/checkpoints/best_policy.pt`
- `artifacts/replays/best_replay.pkl`
- `artifacts/metrics.csv`
- `artifacts/plots/training_metrics.png`

## Parallelization path

- Run many training jobs in parallel at the process level today for hyperparameter sweeps.
- Move hot loops in `core.environment` into a pybind11 backend once rollout throughput becomes the bottleneck.
- Batch policy inference by population, or shard population members across vectorized arenas.
- Keep PPO in Python and only expose `reset`, `step`, `observe`, and `snapshot` from the native backend.

