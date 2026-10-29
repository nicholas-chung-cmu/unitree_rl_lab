"""Identical march rewards/physics, with balanced episode starting diagonals."""
from isaaclab.utils import configclass
from unitree_rl_lab.tasks.locomotion.mdp.balanced_phase import BalancedMarchCommand
from .conga_march_env_cfg import CongaMarchEnvCfg


@configclass
class CongaMarchBalancedEnvCfg(CongaMarchEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.class_type=BalancedMarchCommand
