"""Fresh skill policies share dimensions across A--F, not with legacy checkpoints."""
from isaaclab.utils import configclass
from .rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class HierarchicalPPORunnerCfg(BasePPORunnerCfg):
    experiment_name = "go2_hierarchical"
    max_iterations = 1000
    save_interval = 50
    num_steps_per_env = 24
