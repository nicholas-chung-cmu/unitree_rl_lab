"""Resume march-first PPO without applying a walking command/reward curriculum."""
from rsl_rl.runners import OnPolicyRunner


class MarchRunner(OnPolicyRunner):
    def phase_mode(self):
        name=self.env.unwrapped.cfg.commands.base_velocity.class_type.__name__
        return 'balanced' if name=='BalancedMarchCommand' else 'fixed'

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.num_steps_per_env != 24:
            raise ValueError('March-first experiment requires 24-step rollouts')

    def save(self,path,infos=None):
        infos=dict(infos or {})
        infos['march_stage']={'control_steps':self.env.unwrapped.common_step_counter,
                              'schedule_version':1,'rollout_steps':24,'phase_mode':self.phase_mode()}
        super().save(path,infos)

    def load(self,path,load_optimizer=True,map_location=None):
        infos=super().load(path,load_optimizer,map_location)
        stage=(infos or {}).get('march_stage')
        if not stage or stage['schedule_version']!=1 or stage['rollout_steps']!=24:
            raise ValueError('March resume requires a march-first checkpoint')
        if stage.get('phase_mode','fixed') != self.phase_mode():
            raise ValueError('March checkpoint starting-phase mode differs from this task')
        self.env.unwrapped.common_step_counter=stage['control_steps']
        self.current_learning_iteration+=1
        print(f'MARCH RESUME: completed iterations={self.current_learning_iteration}; phase mode={self.phase_mode()}',flush=True)
        return infos
