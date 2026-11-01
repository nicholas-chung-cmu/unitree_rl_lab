from isaaclab.utils import configclass
from .rsl_rl_ppo_cfg import BasePPORunnerCfg

@configclass
class CPGPPORunnerCfg(BasePPORunnerCfg):
    experiment_name='go2_cpg'
    max_iterations=1000
    save_interval=100
    def __post_init__(self):
        self.policy.init_noise_std=.5
        self.policy.actor_hidden_dims=[256,128,128]
        self.policy.critic_hidden_dims=[256,128,128]
