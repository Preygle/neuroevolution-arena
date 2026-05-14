from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Action(IntEnum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    MOVE_UP_LEFT = 4
    MOVE_UP_RIGHT = 5
    MOVE_DOWN_LEFT = 6
    MOVE_DOWN_RIGHT = 7
    ATTACK = 8
    OPEN_CHEST = 9
    USE_GATE = 10
    STAY = 11


class PowerUpType(IntEnum):
    DAMAGE = 0
    RANGE = 1
    SPEED = 2
    DIAGONAL = 3
    VITALITY = 4


class EnemyKind(IntEnum):
    SKIRMISHER = 0
    ARCHER = 1
    MINI_BOSS = 2
    FINAL_BOSS = 3


@dataclass(frozen=True)
class PowerUpStats:
    attack_bonus: float = 0.0
    range_bonus: float = 0.0
    speed_bonus: int = 0
    vitality_bonus: float = 0.0
    diagonal_unlocked: bool = False
    reward: float = 0.0


@dataclass(frozen=True)
class EnemyStats:
    health: float
    damage: float
    attack_range: int
    armor: float
    reward: float
    aggro_range: int


POWERUP_STATS = {
    PowerUpType.DAMAGE: PowerUpStats(attack_bonus=2.5, reward=20.0),
    PowerUpType.RANGE: PowerUpStats(range_bonus=1.0, reward=18.0),
    PowerUpType.SPEED: PowerUpStats(speed_bonus=1, reward=14.0),
    PowerUpType.DIAGONAL: PowerUpStats(diagonal_unlocked=True, reward=20.0),
    PowerUpType.VITALITY: PowerUpStats(vitality_bonus=12.0, reward=16.0),
}

ENEMY_STATS = {
    EnemyKind.SKIRMISHER: EnemyStats(health=32.0, damage=8.0, attack_range=1, armor=1.0, reward=10.0, aggro_range=10),
    EnemyKind.ARCHER: EnemyStats(health=26.0, damage=6.0, attack_range=3, armor=1.0, reward=10.0, aggro_range=12),
    EnemyKind.MINI_BOSS: EnemyStats(health=120.0, damage=10.0, attack_range=2, armor=2.0, reward=60.0, aggro_range=8),
    EnemyKind.FINAL_BOSS: EnemyStats(health=340.0, damage=16.0, attack_range=3, armor=6.0, reward=180.0, aggro_range=10),
}


@dataclass
class ArenaConfig:
    environment_name: str = "dungeon_crawler_training_v5"
    grid_size: int = 36
    population_size: int = 5
    num_floors: int = 10
    episode_steps_min: int = 650
    episode_steps_max: int = 650
    local_vision: int = 5
    extended_vision: int = 9
    extended_accuracy: float = 0.85
    max_health: float = 110.0
    initial_health: float = 90.0
    max_energy: float = 140.0
    initial_energy: float = 120.0
    step_energy_cost: float = 0.45
    diagonal_energy_cost: float = 0.7
    slow_tile_extra_cost: float = 0.55
    hazard_damage: float = 4.0
    heal_tile_amount: float = 2.0
    floor_transition_heal: float = 10.0
    attack_base_damage: float = 7.0
    attack_damage_reward_scale: float = 0.75
    boss_damage_reward_scale: float = 2.5
    boss_first_hit_reward: float = 25.0
    step_penalty: float = -0.03
    hazard_reward_penalty: float = -0.12
    gate_distance_reward_scale: float = 0.6
    chest_distance_reward_scale: float = 0.2
    boss_distance_reward_scale: float = 0.35
    chest_reward: float = 18.0
    gate_reward: float = 65.0
    floor_clear_reward: float = 85.0
    mini_boss_reward: float = 180.0
    final_boss_reward: float = 180.0
    victory_reward: float = 260.0
    death_penalty: float = -12.0
    powered_agent_death_penalty_scale: float = 0.5
    powered_agent_transition_reward_scale: float = 1.5
    team_wipe_penalty: float = -80.0
    invalid_gate_action_penalty: float = 0.0
    locked_gate_action_penalty: float = -1.0
    no_progress_penalty_interval: int = 12
    no_progress_penalty: float = -4.0
    no_progress_patience: int = 28
    no_progress_termination_penalty: float = -16.0
    floor_progress_weight: float = 60.0
    boss_weight: float = 160.0
    chest_weight: float = 2.0
    victory_weight: float = 220.0
    damage_weight: float = 0.18
    survival_weight: float = 0.03
    hidden_size: int = 256
    ppo_learning_rate: float = 3e-4
    ppo_clip: float = 0.2
    ppo_gamma: float = 0.995
    ppo_lambda: float = 0.95
    ppo_epochs: int = 4
    ppo_minibatch_size: int = 256
    ppo_value_coef: float = 0.5
    ppo_entropy_coef: float = 0.025
    ppo_max_grad_norm: float = 0.5
    ppo_target_kl: float = 0.02
    elite_fraction: float = 0.3
    mutation_std: float = 0.008
    mutate_critic: bool = False
    crossover_rate: float = 0.85
    crossover_swap_probability: float = 0.5
    instance_count: int = 32
    max_instance_count: int = 128
    max_render_instance_count: int = 100
    env_worker_count: int = 16
    default_render_instance_count: int = 20
    render_columns: int = 5
    render_page_seconds: float = 1.25
    render_min_cell_size: int = 6
    render_max_window_width: int = 1840
    render_max_window_height: int = 1040
    render_cell_size: int = 24
    render_fps: int = 30
    seed: int = 7
    device: str = "cpu"
    auto_use_gate: bool = True
    gate_interaction_radius: int = 1

    @property
    def action_size(self) -> int:
        return len(Action)

    @property
    def vision_channels(self) -> int:
        return 11

    @property
    def scalar_feature_count(self) -> int:
        return 21

    @property
    def observation_dim(self) -> int:
        local_cells = self.local_vision * self.local_vision
        extended_cells = self.extended_vision * self.extended_vision
        return self.vision_channels * (local_cells + extended_cells) + self.scalar_feature_count

    def to_dict(self) -> dict[str, float | int | str]:
        return self.__dict__.copy()
