# Architecture

## Core modules

### `ArenaConfig`

Holds all simulation, PPO, and mutation settings in one place: grid size, population size, reward scales, item counts, weapon stats, PPO hyperparameters, and visualization defaults.

### `ArenaEnvironment`

Owns the authoritative world state:

- occupancy grid
- item grid
- terrain grid
- agent positions
- health, energy, weapon type, durability
- cumulative reward, kill count, survival steps
- exploration and camping state

Its public contract is shaped like a native backend contract:

- `reset() -> np.ndarray`
- `step(actions: np.ndarray) -> StepBatch`
- `observe_all() -> np.ndarray`
- `snapshot() -> dict`
- `get_agent_metrics() -> list[AgentMetrics]`

### `ActorCriticPolicy`

Small MLP actor-critic network per agent. The actor outputs logits over 7 actions and the critic outputs a scalar state value.

### `PPOPolicyBank`

Manages 30 independent policies and optimizers:

- samples actions for all active agents
- stores and updates policy weights with PPO
- exports/imports policy state for the evolution engine

### `EvolutionEngine`

Selects the top 20% of agents by fitness:

`fitness = reward + survival_steps + kills`

Then rebuilds the next population by cloning elites and applying Gaussian mutation to offspring weights.

### `ArenaSimulation`

Coordinates the whole training lifecycle:

- reset arena
- collect rollouts
- apply PPO updates
- score and evolve the population
- checkpoint the best policy
- save replay frames
- export plots and CSV metrics

## Observation design

Each observation is a flat vector:

- accurate `3x3` local grid with binary channels
- noisy `5x5` extended grid with binary channels
- scalar agent state features

Cell channels:

- wall
- self
- other agent
- food
- poison
- melee weapon
- ranged weapon
- rare melee weapon
- rare ranged weapon

Scalar features:

- normalized health
- normalized energy
- weapon one-hot
- normalized weapon durability

## v1.1 learning changes

- Fixed training map pool with curriculum-based map activation instead of fully unconstrained terrain randomness
- Center-biased food and weapon placement to create repeatable conflict zones
- Edge-biased spawn placement to force traversal through terrain
- Exploration reward for new cells
- Camping penalty for staying in the same local radius too long, with stronger penalty in corners
- Fitness rebalanced away from double-counting pure survival

## Native backend migration

The clean migration path is:

1. Keep `ArenaSimulation`, `PPOPolicyBank`, `EvolutionEngine`, and visualization in Python.
2. Replace `ArenaEnvironment` internals with a pybind11-backed `NativeArenaBackend`.
3. Preserve the same Python-level API so training code does not change.
4. Move the following hot paths first:
   - movement resolution
   - attack targeting and damage application
   - observation encoding
   - bulk reset/spawn logic
