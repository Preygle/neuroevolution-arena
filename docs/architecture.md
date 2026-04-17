# Architecture

## System goal

The branch `dungeon-crawler-training` replaces the random free-for-all arena with a deterministic multi-floor dungeon campaign. The training problem is no longer "survive whatever random map appears." It is "learn the best cooperative route through a fixed sequence of tactical floors."

That changes the architecture in an important way:

- environment complexity is authored, not sampled every episode
- progression is sequential, not single-map survival
- power-ups persist across floors
- miniboss and final boss checks create long-horizon dependencies

## Core modules

### `ArenaConfig`

`behavior_mutation_arena/config.py`

Centralizes all campaign, PPO, mutation, and rendering settings:

- grid size
- team size
- number of floors
- action and observation dimensions
- reward scales
- power-up and enemy stats
- PPO hyperparameters
- elite fraction and mutation standard deviation

The package still uses the historic `ArenaConfig` name, but the active semantics are now dungeon-oriented.

### `DungeonFloor`

`behavior_mutation_arena/core/dungeon_floors.py`

Each floor is a static authored problem definition:

- traversable terrain mask
- slow, hazard, and heal layers
- fixed team spawn formation
- chest positions and power-up types
- enemy positions and enemy kinds
- gate position
- gate lock rules for boss floors

This is the anchor of learnability. The floor data is deterministic, so the policy repeatedly sees the same navigation and combat problems and can accumulate route knowledge across generations.

### `ArenaEnvironment`

`behavior_mutation_arena/core/environment.py`

Owns the campaign runtime:

- active floor state
- team positions, health, energy, and alive mask
- persistent campaign buffs
- chest state
- enemy state
- gate state
- reward accumulation
- survival steps
- floor progress
- boss clears
- victory state

Public contract:

- `reset() -> np.ndarray`
- `step(actions: np.ndarray) -> StepBatch`
- `observe_all() -> np.ndarray`
- `snapshot() -> dict`
- `get_agent_metrics() -> list[AgentMetrics]`

The public API intentionally stays backend-friendly so the environment can later move behind a pybind11 C++ core without forcing a rewrite of PPO, evolution, or rendering.

### `ActorCriticPolicy`

`behavior_mutation_arena/rl/policy.py`

MLP actor-critic network that maps flattened observations to:

- action logits over the dungeon action set
- value estimates for PPO

The current branch still uses a relatively simple policy network. That is intentional for the first dungeon pass: stabilize the task first, then scale the policy once the reward curve becomes trustworthy.

### `PPOPolicyBank`

`behavior_mutation_arena/rl/ppo.py`

Maintains the active policy population:

- action selection for living agents
- value estimation
- rollout buffering
- PPO optimization
- import and export of policy and optimizer state for checkpointing and evolution
- merged PPO updates from multiple dungeon instances in the same generation

### `EvolutionEngine`

`behavior_mutation_arena/evolution/engine.py`

Applies the outer-loop search pressure:

- select elites from the current generation
- preserve the strongest policies
- crossover two elite parents into offspring
- mutate offspring weights with Gaussian noise

This keeps the hybrid PPO plus evolution structure, but the task itself is now much more stable than the old random arena.

### `ArenaSimulation`

`behavior_mutation_arena/interface/api.py`

Runs the full training lifecycle:

- reset the dungeon campaign
- gather rollouts
- gather rollouts from multiple simultaneous dungeon instances when configured
- batch policy inference across all live instances before stepping the environments
- step environments through a shared executor instead of hard-serial stepping
- update PPO
- score agents with dungeon-aware fitness
- evolve the next generation
- save best-policy checkpoints
- persist resumable training checkpoints
- export plots and metrics
- capture replay data

## Campaign design

### Team

- 5 cooperative agents
- fixed spawn formation per floor
- shared access to team-wide permanent buffs

### Floor progression

- 10 authored floors
- gate use advances the campaign
- floor 5 contains a miniboss and locked gate
- floor 10 contains the final boss and final gate
- chest decisions on earlier floors affect later success probability

### Buff persistence

Chest rewards are not disposable pickups. They change the state of the rest of the episode:

- damage bonus
- range bonus
- extra movement
- diagonal movement unlock
- vitality increase and team healing

This is what gives the training task strategic depth. A weak route can reach floor 5 quickly and still fail. A stronger route might spend time taking a chest that unlocks later boss viability.

## Observation design

Observations are flattened vectors built from two spatial views plus scalar state.

### Spatial channels

Both the accurate local view and noisy extended view encode 11 channels:

- wall
- self
- teammate
- enemy
- boss
- chest
- open gate
- locked gate
- slow tile
- hazard tile
- heal tile

### Scalar features

- normalized health
- normalized energy
- current floor progress
- floors cleared
- alive teammate ratio
- attack bonus
- range bonus
- speed bonus
- diagonal unlock flag
- chests opened
- bosses defeated
- gate-open flag

## Action design

The action set reflects dungeon traversal and tactical progression:

- four cardinal moves
- four diagonal moves
- attack
- open chest
- use gate
- stay

Diagonal actions are present in the policy head from the start but only become useful after the diagonal-movement chest is collected.

## Reward and fitness design

### Step rewards

- small step penalty
- chest rewards
- damage-dealt rewards
- floor gate rewards
- floor clear rewards
- miniboss reward
- final boss reward
- final victory reward
- death penalty
- team wipe penalty

### Fitness terms

Per-agent fitness combines:

- accumulated reward
- floors cleared
- bosses defeated
- chests opened
- damage dealt
- survival contribution
- victory bonus

This is designed to make progress through the authored campaign matter more than passive stalling.

## Metrics

`GenerationSummary` now tracks campaign-specific signals:

- best fitness
- mean fitness
- mean reward
- mean survival
- mean floor reached
- mean bosses defeated
- mean chests opened
- mean damage
- mean gate distance
- success rate

Those metrics are written to `artifacts/metrics.csv` and plotted into `artifacts/plots/training_metrics.png`.

## Visualization

`behavior_mutation_arena/visual/renderer.py`

The renderer mirrors the authored floor state:

- biome terrain
- slow, hazard, and heal overlays
- chest types
- enemy kinds
- gate lock state
- team positions and health
- optional paged multi-instance rendering with 5 instances per row

## Parallelism notes

- The current branch uses batched inference plus a `ThreadPoolExecutor` for environment stepping.
- That avoids Windows multiprocessing spawn overhead while still giving concurrency on the numpy-heavy parts of the environment.
- A future persistent process-worker backend is the next step if thread-level scaling stops being enough.
- sidebar with buffs, floor progress, bosses, and chest counts

That makes it much easier to watch whether the team is actually learning a route or just wandering.

## Checkpointing and resumption

`ArenaSimulation` saves two kinds of state:

- best policy checkpoint for evaluation and replay
- resumable training checkpoint every 10 generations

Resumption is config-aware. If the environment signature changes enough to invalidate an old training state, the loader safely ignores it instead of restoring incompatible weights.

## Native backend migration

The clean upgrade path remains:

1. Keep `ArenaSimulation`, PPO, evolution, and plotting in Python.
2. Move the dungeon stepping core behind `interface.backend.build_backend()`.
3. Preserve the same `reset/step/observe/snapshot/metrics` API.
4. Port the hottest paths first:
   - movement resolution
   - attack targeting
   - enemy turns
   - tile effects
   - observation encoding
