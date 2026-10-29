"""Checkpoint transfer from Strong walking, then guarded kick-only resume."""
import torch
from rsl_rl.runners import OnPolicyRunner
from .march_training import MarchRunner


class KickRunner(MarchRunner):
    def phase_mode(self):return 'kick_intro_v1'

    def load(self,path,load_optimizer=True,map_location=None):
        saved=torch.load(path,weights_only=False,map_location='cpu')
        mode=(saved.get('infos') or {}).get('march_stage',{}).get('phase_mode')
        if mode=='kick_intro_v1':return super().load(path,load_optimizer,map_location)
        if mode!='duration_strong_v1':raise ValueError('Kick transfer requires Strong walking checkpoint')
        infos=OnPolicyRunner.load(self,path,load_optimizer,map_location)
        self.current_learning_iteration+=1
        self.env.unwrapped.common_step_counter=self.current_learning_iteration*24
        print(f'KICK TRANSFER: {self.current_learning_iteration} completed walking iterations',flush=True)
        return infos


class KickRefineRunner(KickRunner):
    def phase_mode(self):return 'kick_intro_air_v2'

    def load(self,path,load_optimizer=True,map_location=None):
        saved=torch.load(path,weights_only=False,map_location='cpu')
        stage=(saved.get('infos') or {}).get('march_stage',{})
        if stage.get('phase_mode')=='kick_intro_air_v2':
            return MarchRunner.load(self,path,load_optimizer,map_location)
        if stage.get('phase_mode')!='kick_intro_v1':raise ValueError('Kick refinement requires introductory kick checkpoint')
        infos=OnPolicyRunner.load(self,path,load_optimizer,map_location)
        self.current_learning_iteration+=1
        self.env.unwrapped.common_step_counter=stage['control_steps']
        print(f'KICK REFINE: continuing at {self.current_learning_iteration} with height-aware lift bonus',flush=True)
        return infos
