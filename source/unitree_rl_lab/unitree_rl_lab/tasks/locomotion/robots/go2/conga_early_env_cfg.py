"""Fresh basic locomotion with gentle early phase shaping."""
from isaaclab.utils import configclass
from .conga_cadence_paired_env_cfg import CongaCadencePairedEnvCfg


@configclass
class CongaEarlyEnvCfg(CongaCadencePairedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.cadence_support = None
        self.rewards.cadence_pair.weight = 0.0
        self.rewards.cadence_phase.weight = 0.05
        self.commands.base_velocity.ranges.lin_vel_x = (-0.1, 0.1)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.1, 0.1)
