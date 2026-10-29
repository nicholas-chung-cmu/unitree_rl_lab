"""Episode-local starting-diagonal balance without altering physical episode age."""
import torch
from .conga import CongaVelocityCommand
from .conga_reference import generate_reference


class BalancedMarchCommand(CongaVelocityCommand):
    def __init__(self,cfg,env):
        # A private generator avoids perturbing physical reset/policy RNG draws.
        generator=torch.Generator(device=env.device)
        generator.manual_seed(int(getattr(getattr(env,'cfg',None),'seed',0) or 0)+817)
        order=torch.randperm(env.num_envs,device=env.device,generator=generator)
        self.phase_offset=(order%2).float()*.5
        super().__init__(cfg,env)

    @property
    def reference(self):
        steps=getattr(self._env,'episode_length_buf',torch.zeros(self.num_envs,device=self.device))
        return generate_reference(steps*self._env.step_dt+self.phase_offset,self.cfg.reference)

    def reset(self,env_ids=None):
        if env_ids is None:
            env_ids=torch.arange(self.num_envs,device=self.device)
        elif isinstance(env_ids,slice):
            env_ids=torch.arange(self.num_envs,device=self.device)[env_ids]
        else:
            env_ids=torch.as_tensor(env_ids,device=self.device)
        # Alternate each environment's first pair on successive episodes. This
        # balances each environment over time, even for asynchronous resets.
        self.phase_offset[env_ids]=.5-self.phase_offset[env_ids]
        return super().reset(env_ids)
