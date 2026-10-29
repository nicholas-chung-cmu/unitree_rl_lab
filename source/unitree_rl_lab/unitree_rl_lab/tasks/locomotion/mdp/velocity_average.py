"""Two-second body-frame mean-error tracking with immediate neutral response."""
import torch


class VelocityHistory:
    def __init__(self,n,device,window_steps=100):
        self.buffer=torch.zeros(window_steps,n,2,device=device)
        self.total=torch.zeros(n,2,device=device)
        self.count=torch.zeros(n,dtype=torch.long,device=device)
        self.command=torch.zeros(n,3,device=device)
        self.index=0

    def step(self,velocity,command,reset):
        active=(command[:,:2].norm(dim=-1)>.05) | (command[:,2].abs()>.05)
        changed=(command-self.command).abs().amax(dim=-1)>1e-6
        clear=reset | changed | ~active
        self.buffer[:,clear]=0
        self.total[clear]=0;self.count[clear]=0
        error=velocity-command[:,:2]
        self.total += error-self.buffer[self.index]
        self.buffer[self.index]=error
        self.count=(self.count+1).clamp_max(len(self.buffer))
        mean=self.total/self.count[:,None]
        instant=torch.exp(-error.square().sum(-1)/.25)
        averaged=torch.exp(-mean.square().sum(-1)/.25)
        result=torch.where(active,.8*averaged+.2*instant,instant)
        self.command=command.clone()
        self.index=(self.index+1)%len(self.buffer)
        return result


def tracking(env):
    if not hasattr(env,'_velocity_history'):
        env._velocity_history=VelocityHistory(env.num_envs,env.device,round(2./env.step_dt))
        env._velocity_history_tick=-1
    if env._velocity_history_tick!=env.common_step_counter:
        env._velocity_history_score=env._velocity_history.step(
            env.scene['robot'].data.root_lin_vel_b[:,:2],
            env.command_manager.get_command('base_velocity'),env.episode_length_buf<=1)
        env._velocity_history_tick=env.common_step_counter
    return env._velocity_history_score
