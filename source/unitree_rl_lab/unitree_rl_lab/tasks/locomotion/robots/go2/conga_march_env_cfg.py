"""Zero-command march-first candidate; no automatic command/stage advancement."""
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from unitree_rl_lab.tasks.locomotion.mdp import cadence_pair_scores, march_rewards
from .conga_pause_env_cfg import CongaPauseEnvCfg


@configclass
class CongaMarchEnvCfg(CongaPauseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        command=self.commands.base_velocity
        command.ranges.lin_vel_x=(0.,0.)
        command.ranges.lin_vel_y=(0.,0.)
        command.ranges.ang_vel_z=(0.,0.)
        command.rel_standing_envs=1.
        command.rel_turning_envs=0.
        self.curriculum.terrain_levels=None
        self.rewards.cadence_phase.weight=1.
        # Dense individual progress plus pair completion. Do not require already
        # achieving alternation to earn exploratory swing credit from scratch.
        self.rewards.cadence_pair.func=cadence_pair_scores.paired_step
        self.rewards.cadence_pair.weight=.25
        self.rewards.march_clearance=RewTerm(func=march_rewards.individual_clearance,weight=.25)
        self.rewards.march_missed_swing=RewTerm(func=march_rewards.missed_swing,weight=-.5)
