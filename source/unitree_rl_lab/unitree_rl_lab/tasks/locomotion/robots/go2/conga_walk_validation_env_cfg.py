"""Walking reward v2 experiment; rejected after 200 iterations (see the v2 report).

Retained for review/reproduction; does not change the final conga curriculum.
"""

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm

from unitree_rl_lab.tasks.locomotion.mdp import conga_walk_rewards as walk

from .conga_env_cfg import CongaCadenceEnvCfg


@configclass
class CongaWalkValidationEnvCfg(CongaCadenceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 256
        # Cadence already disables both trajectories while retaining the clock,
        # continuous diagonal stepping, marching, and translation/yaw commands.
        self.rewards.conga_kick = None
        self.rewards.conga_height = None
        self.rewards.base_linear_velocity = None
        self.rewards.track_lin_vel_xy.params["std"] = 0.2
        self.rewards.track_ang_vel_z.params["std"] = 0.3
        self.rewards.conga_contact = RewTerm(func=walk.contact_error, weight=-3.0)
        self.rewards.conga_clearance = RewTerm(func=walk.swing_quality, weight=2.0)
        self.rewards.conga_cadence = RewTerm(func=walk.touchdown_events, weight=1.0)
        self.rewards.conga_marching = RewTerm(func=walk.marching, weight=0.5)
        self.rewards.walk_missed_swing = RewTerm(func=walk.missed_swing, weight=-2.0)
        self.rewards.walk_nominal_height = RewTerm(func=walk.nominal_height, weight=0.5)
