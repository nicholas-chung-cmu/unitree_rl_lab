"""Standing/walking preparation with stock dynamics and conga observations."""

from dataclasses import field

from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.mdp import conga
from unitree_rl_lab.tasks.locomotion.mdp.conga_reference import CongaReferenceCfg

from .conga_env_cfg import CongaObservationsCfg
from .velocity_env_cfg import CommandsCfg, RobotEnvCfg


@configclass
class CongaBasicVelocityCommandCfg(mdp.UniformLevelVelocityCommandCfg):
    # Metadata for observations/export only; retain the stock command class,
    # sampling, curriculum and RNG consumption. No travel-envelope modulation.
    reference: CongaReferenceCfg = field(default_factory=lambda: CongaReferenceCfg(
        kick_enabled=False, kick_scale=0.0, bounce_scale=0.0,
    ))


@configclass
class CongaBasicCommandsCfg(CommandsCfg):
    base_velocity = CongaBasicVelocityCommandCfg(**vars(CommandsCfg().base_velocity))


@configclass
class CongaBasicEnvCfg(RobotEnvCfg):
    commands: CongaBasicCommandsCfg = CongaBasicCommandsCfg()
    observations: CongaObservationsCfg = CongaObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.conga_phase.func = conga.basic_conga_features
        self.observations.critic.conga_phase.func = conga.basic_conga_features
        self.commands.base_velocity.reference.validate()
