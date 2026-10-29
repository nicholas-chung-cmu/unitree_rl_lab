"""Fixed final early-shaping settings with only pair eligibility changed."""
from isaaclab.utils import configclass
from unitree_rl_lab.tasks.locomotion.mdp import liftoff_history
from .conga_early_env_cfg import CongaEarlyEnvCfg


@configclass
class CongaHistoryEnvCfg(CongaEarlyEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.cadence_pair.func = liftoff_history.paired_step
        self.rewards.cadence_pair.weight = .10
        self.rewards.cadence_phase.weight = .25
        for name in ('joint_pos','feet_air_time','air_time_variance'):
            getattr(self.rewards,name).weight = 0.
        self.commands.base_velocity.ranges.lin_vel_x = (-.5,.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-.25,.25)
