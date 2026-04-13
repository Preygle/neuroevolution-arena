from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Action(IntEnum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    STAY = 4
    ATTACK = 5
    PICK_ITEM = 6


class WeaponKind(IntEnum):
    NONE = 0
    MELEE = 1
    RANGED = 2
    RARE_MELEE = 3
    RARE_RANGED = 4


class ItemKind(IntEnum):
    EMPTY = 0
    FOOD = 1
    POISON = 2
    MELEE = 3
    RANGED = 4
    RARE_MELEE = 5
    RARE_RANGED = 6


@dataclass(frozen=True)
class WeaponStats:
    kind: WeaponKind
    range: int
    damage: float
    durability: float


WEAPON_STATS = {
    WeaponKind.NONE: WeaponStats(WeaponKind.NONE, range=1, damage=4.0, durability=-1.0),
    WeaponKind.MELEE: WeaponStats(WeaponKind.MELEE, range=1, damage=18.0, durability=6.0),
    WeaponKind.RANGED: WeaponStats(WeaponKind.RANGED, range=3, damage=11.0, durability=8.0),
    WeaponKind.RARE_MELEE: WeaponStats(WeaponKind.RARE_MELEE, range=2, damage=25.0, durability=8.0),
    WeaponKind.RARE_RANGED: WeaponStats(WeaponKind.RARE_RANGED, range=5, damage=15.0, durability=10.0),
}

ITEM_TO_WEAPON = {
    ItemKind.MELEE: WeaponKind.MELEE,
    ItemKind.RANGED: WeaponKind.RANGED,
    ItemKind.RARE_MELEE: WeaponKind.RARE_MELEE,
    ItemKind.RARE_RANGED: WeaponKind.RARE_RANGED,
}


@dataclass
class ArenaConfig:
    grid_size: int = 15
    population_size: int = 30
    episode_steps_min: int = 180
    episode_steps_max: int = 180
    local_vision: int = 3
    extended_vision: int = 5
    extended_accuracy: float = 0.7
    max_health: float = 100.0
    initial_health: float = 100.0
    max_energy: float = 60.0
    initial_energy: float = 40.0
    step_energy_cost: float = 1.0
    starvation_damage: float = 2.0
    food_energy: float = 18.0
    poison_damage: float = 18.0
    food_reward: float = 8.0
    poison_reward: float = -12.0
    kill_reward: float = 28.0
    survival_reward: float = 0.15
    exploration_reward: float = 0.45
    attack_damage_reward_scale: float = 0.45
    camp_radius: int = 1
    camp_threshold: int = 8
    camping_penalty: float = 0.6
    corner_camping_penalty: float = 0.35
    fitness_kill_weight: float = 8.0
    fitness_damage_weight: float = 0.1
    fitness_exploration_weight: float = 0.75
    fitness_camping_weight: float = 1.1
    curriculum_initial_map_count: int = 4
    curriculum_full_map_count: int = 8
    curriculum_growth_generations: int = 1200
    spawn_band_width: int = 2
    food_count: int = 26
    poison_count: int = 18
    melee_count: int = 10
    ranged_count: int = 8
    rare_melee_count: int = 4
    rare_ranged_count: int = 2
    hidden_size: int = 128
    ppo_learning_rate: float = 3e-4
    ppo_clip: float = 0.2
    ppo_gamma: float = 0.99
    ppo_lambda: float = 0.95
    ppo_epochs: int = 6
    ppo_minibatch_size: int = 64
    ppo_value_coef: float = 0.5
    ppo_entropy_coef: float = 0.01
    ppo_max_grad_norm: float = 0.5
    elite_fraction: float = 0.3
    mutation_std: float = 0.015
    render_cell_size: int = 36
    render_fps: int = 30
    seed: int = 7
    device: str = "cpu"

    @property
    def action_size(self) -> int:
        return len(Action)

    @property
    def vision_channels(self) -> int:
        return 9

    @property
    def scalar_feature_count(self) -> int:
        return 8

    @property
    def observation_dim(self) -> int:
        local_cells = self.local_vision * self.local_vision
        extended_cells = self.extended_vision * self.extended_vision
        return self.vision_channels * (local_cells + extended_cells) + self.scalar_feature_count

    def item_spawn_counts(self) -> dict[ItemKind, int]:
        return {
            ItemKind.FOOD: self.food_count,
            ItemKind.POISON: self.poison_count,
            ItemKind.MELEE: self.melee_count,
            ItemKind.RANGED: self.ranged_count,
            ItemKind.RARE_MELEE: self.rare_melee_count,
            ItemKind.RARE_RANGED: self.rare_ranged_count,
        }

    def to_dict(self) -> dict[str, float | int]:
        return self.__dict__.copy()
