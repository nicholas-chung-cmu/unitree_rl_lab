"""Reviewed cadence starting values; CadenceFadeEnv fades stock terms in 200 iterations."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion.mdp import conga, conga_walk_rewards

from .conga_env_cfg import CongaCommandsCfg, CongaObservationsCfg
from .velocity_env_cfg import RewardsCfg, RobotEnvCfg


@configclass
class CadenceStartRewardsCfg(RewardsCfg):
    conga_contact = RewTerm(func=conga.contact_tracking, weight=0.25)
    conga_clearance = RewTerm(func=conga_walk_rewards.swing_quality, weight=0.5,
                             params={"std": 0.015})


@configclass
class CongaCadenceStartEnvCfg(RobotEnvCfg):
    commands: CongaCommandsCfg = CongaCommandsCfg()
    observations: CongaObservationsCfg = CongaObservationsCfg()
    rewards: CadenceStartRewardsCfg = CadenceStartRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        command = self.commands.base_velocity
        command.reference.kick_enabled = False
        command.reference.kick_scale = 0.0
        command.reference.bounce_scale = 0.0
        # Cap the norm, including combined forward/lateral commands, not just vx.
        command.reference.max_travel_speed = 0.5
        command.ranges.lin_vel_x = (-0.5, 0.5)
        command.ranges.lin_vel_y = (-0.25, 0.25)
        self.curriculum.lin_vel_cmd_levels = None
        # Keep stock physics/randomization. CadenceFadeEnv applies the approved
        # 4,800-control-step fade to the three retiring stock rewards.
        command.reference.validate()
