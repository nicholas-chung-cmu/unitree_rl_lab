"""Untrained fixed-cadence pause/repetition candidate, isolated from prior runs."""
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from unitree_rl_lab.tasks.locomotion.mdp import pause_rewards
from .conga_history_env_cfg import CongaHistoryEnvCfg


@configclass
class CongaPauseEnvCfg(CongaHistoryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.cadence_pair.func=pause_rewards.paired_step
        self.rewards.pause_missing=RewTerm(func=pause_rewards.missing_pause_contacts,weight=-.25)
        self.rewards.repeated_departures=RewTerm(func=pause_rewards.repeated_departures,weight=-.025)
