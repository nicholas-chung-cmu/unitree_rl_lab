"""Small explicitly requested kicks, with separate height/lateral shaping."""
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg
from .conga_kick_test_env_cfg import CongaKickTestEnvCfg
from unitree_rl_lab.tasks.locomotion.mdp import kick_training


@configclass
class CongaKickTrainEnvCfg(CongaKickTestEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.class_type=kick_training.KickTrainingCommand
        self.commands.base_velocity.reference.kick_lateral=.02
        self.commands.base_velocity.reference.kick_height=.02
        self.rewards.conga_contact=None
        self.rewards.conga_kick=None
        self.rewards.kick_height=RewardTermCfg(func=kick_training.height_tracking,weight=2.)
        self.rewards.kick_lateral=RewardTermCfg(func=kick_training.lateral_tracking,weight=2.)
        self.rewards.kick_airborne=RewardTermCfg(func=kick_training.airborne,weight=.5)
        self.rewards.kick_support=RewardTermCfg(func=kick_training.missing_support,weight=-1.)


@configclass
class CongaKickRefineEnvCfg(CongaKickTrainEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.kick_airborne.func=kick_training.airborne_refined
        self.rewards.kick_airborne.weight=2.
