"""Basic walking command cap and gentle contact-duration shaping."""
import torch
from isaaclab.envs.mdp import UniformVelocityCommand
from .duration_scores import DurationState
from .conga_reference import generate_reference


class DurationVelocityCommand(UniformVelocityCommand):
    def _update_command(self):
        super()._update_command()
        xy=self.vel_command_b[:,:2]
        xy *= (.5/xy.norm(dim=-1).clamp_min(.5))[:,None]


def state(env):
    tick=env.common_step_counter
    if not hasattr(env,'_duration_state'):
        env._duration_state=DurationState(env.num_envs,env.device)
        env._duration_tick=-1
        env._duration_ids=env.scene['contact_forces'].find_bodies(
            ['FL_foot','FR_foot','RL_foot','RR_foot'],preserve_order=True)[0]
    if env._duration_tick != tick:
        command=env.command_manager.get_command('base_velocity')
        active=(command[:,:2].norm(dim=-1)>.05) | (command[:,2].abs()>.05)
        contact=env.scene['contact_forces'].data.net_forces_w[:,env._duration_ids].norm(dim=-1)>1.
        env._duration_cost=env._duration_state.step(contact,active,env.episode_length_buf<=1,env.step_dt)
        env._duration_tick=tick
    return env._duration_cost


def air_cost(env):
    return state(env)[:,0]


def stance_cost(env):
    return state(env)[:,1]


def features(env):
    command=env.command_manager.get_command('base_velocity')
    active=(command[:,:2].norm(dim=-1)>.05) | (command[:,2].abs()>.05)
    if not hasattr(env,'_duration_clock'):
        env._duration_clock=torch.zeros(env.num_envs,device=env.device)
        env._duration_clock_active=torch.zeros_like(active)
        env._duration_clock_tick=-1
    if env._duration_clock_tick != env.common_step_counter:
        restart=(env.episode_length_buf<=1) | ~active | ~env._duration_clock_active
        env._duration_clock=torch.where(restart,0.,env._duration_clock+env.step_dt)
        env._duration_clock_active=active.clone()
        env._duration_clock_tick=env.common_step_counter
    ref=generate_reference(env._duration_clock,env.cfg.commands.base_velocity.reference)
    return torch.cat((ref.phase_features(),command,torch.zeros(env.num_envs,2,device=env.device)),dim=-1)
