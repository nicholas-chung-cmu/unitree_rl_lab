"""Isolated, explicitly requested alternating front kicks from a stable stance."""
from isaaclab.utils import configclass
from unitree_rl_lab.tasks.locomotion.mdp.conga import CongaVelocityCommand
from unitree_rl_lab.tasks.locomotion.mdp.conga_reference import generate_kick_only_reference
from .conga_env_cfg import CongaEnvCfg


class IsolatedKickCommand(CongaVelocityCommand):
    @property
    def reference(self):
        import torch
        steps=getattr(self._env,'episode_length_buf',torch.zeros(self.num_envs,device=self.device))
        return generate_kick_only_reference(steps*self._env.step_dt,self.cfg.reference)


@configclass
class CongaKickTestEnvCfg(CongaEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        command=self.commands.base_velocity
        command.class_type=IsolatedKickCommand
        command.ranges.lin_vel_x=(0.,0.)
        command.ranges.lin_vel_y=(0.,0.)
        command.ranges.ang_vel_z=(0.,0.)
        command.rel_standing_envs=1.
        command.rel_turning_envs=0.
        command.reference.kick_enabled=True
        command.reference.kick_scale=1.
        command.reference.bounce_scale=0.
        self.rewards.conga_cadence=None
        self.rewards.conga_clearance=None
        self.rewards.conga_marching=None
