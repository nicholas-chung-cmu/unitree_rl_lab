from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class CongaPPORunnerCfg(BasePPORunnerCfg):
    experiment_name = "go2_conga"
    # More than one beat per rollout; the reference is observable, so a recurrent
    # actor is unnecessary. All stages share this actor/critic layout.
    num_steps_per_env = 48
