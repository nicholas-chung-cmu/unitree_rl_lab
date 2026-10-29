"""Independent kick exploration terms and balanced first-leg requests."""
import torch
from .conga_reference import generate_kick_only_reference
from ..robots.go2.conga_kick_test_env_cfg import IsolatedKickCommand


class KickTrainingCommand(IsolatedKickCommand):
    def __init__(self,cfg,env):
        generator=torch.Generator(device=env.device).manual_seed(int(env.cfg.seed or 0)+1031)
        self.phase_offset=(torch.randperm(env.num_envs,device=env.device,generator=generator)%2).float()*2.
        super().__init__(cfg,env)

    @property
    def reference(self):
        steps=getattr(self._env,'episode_length_buf',torch.zeros(self.num_envs,device=self.device))
        return generate_kick_only_reference(steps*self._env.step_dt+self.phase_offset,self.cfg.reference)

    def reset(self,env_ids=None):
        ids=torch.arange(self.num_envs,device=self.device) if env_ids is None or isinstance(env_ids,slice) else torch.as_tensor(env_ids,device=self.device)
        if isinstance(env_ids,slice):ids=ids[env_ids]
        self.phase_offset[ids]=2.-self.phase_offset[ids]
        return super().reset(ids)


from .kick_scores import components


def terms(env):
    term=env.command_manager.get_term('base_velocity')
    body,height,contact=term.foot_state()
    return components(body[:,:,1],height,contact,term.reference)


def height_tracking(env):return terms(env)[:,0]
def lateral_tracking(env):return terms(env)[:,1]
def airborne(env):return terms(env)[:,2]
def missing_support(env):return terms(env)[:,3]


def airborne_refined(env):
    term=env.command_manager.get_term('base_velocity')
    body,height,contact=term.foot_state()
    return components(body[:,:,1],height,contact,term.reference,height_aware=True)[:,2]
