"""Short-swing, phase-balanced cadence candidate; not yet trained."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from unitree_rl_lab.tasks.locomotion.mdp import cadence_pair_scores as paired
from .conga_cadence_start_env_cfg import CongaCadenceStartEnvCfg


@configclass
class CongaCadencePairedEnvCfg(CongaCadenceStartEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.reference.cadence_swing_start = .12
        self.commands.base_velocity.reference.cadence_swing_end = .32
        self.rewards.conga_contact = None
        self.rewards.conga_clearance = None
        self.rewards.cadence_support = RewTerm(func=paired.four_foot_support, weight=.1)
        self.rewards.cadence_pair = RewTerm(func=paired.paired_step, weight=.5)
        self.rewards.cadence_phase = RewTerm(func=paired.phase_matching, weight=.5)
        self.commands.base_velocity.reference.validate()
