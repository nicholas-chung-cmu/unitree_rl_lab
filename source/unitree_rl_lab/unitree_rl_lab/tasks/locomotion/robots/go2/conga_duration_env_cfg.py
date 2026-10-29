"""Stock basic walking with bounded gentle duration costs; neutral permits standing."""
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg
from .conga_basic_env_cfg import CongaBasicEnvCfg
from unitree_rl_lab.tasks.locomotion.mdp import duration_walking


@configclass
class CongaDurationEnvCfg(CongaBasicEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.conga_phase.func=duration_walking.features
        self.observations.critic.conga_phase.func=duration_walking.features
        self.commands.base_velocity.class_type=duration_walking.DurationVelocityCommand
        self.commands.base_velocity.limit_ranges.lin_vel_x=(-.5,.5)
        self.commands.base_velocity.limit_ranges.lin_vel_y=(-.25,.25)
        self.rewards.feet_air_time.weight=0.
        self.rewards.air_time_variance.weight=0.
        self.rewards.duration_air=RewardTermCfg(func=duration_walking.air_cost,weight=-.05)
        self.rewards.duration_stance=RewardTermCfg(func=duration_walking.stance_cost,weight=-.05)


@configclass
class CongaDurationStrongEnvCfg(CongaDurationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.duration_air.weight=-.20
        self.rewards.duration_stance.weight=-.20


@configclass
class CongaDurationAverageEnvCfg(CongaDurationStrongEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        from unitree_rl_lab.tasks.locomotion.mdp.velocity_average import tracking
        self.rewards.track_lin_vel_xy.func=tracking
        self.rewards.track_lin_vel_xy.params={}
