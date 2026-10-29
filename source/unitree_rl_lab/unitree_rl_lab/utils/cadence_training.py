"""Cadence-only reward fade and checkpointed stage progress."""

from isaaclab.envs import ManagerBasedRLEnv
from rsl_rl.runners import OnPolicyRunner

FADE_STEPS = 200 * 24
INITIAL_WEIGHTS = {"joint_pos": -0.7, "feet_air_time": 0.1, "air_time_variance": -1.0}


def fade_fraction(step):
    return max(0.0, 1.0 - max(0, step) / FADE_STEPS)


def apply_fade(env):
    fraction = fade_fraction(getattr(env, "cadence_fade_frozen_step", env.common_step_counter))
    for name, initial in INITIAL_WEIGHTS.items():
        cfg = env.reward_manager.get_term_cfg(name)
        cfg.weight = initial * fraction
        env.reward_manager.set_term_cfg(name, cfg)
    return fraction


class CadenceFadeEnv(ManagerBasedRLEnv):
    def step(self, action):
        apply_fade(self)
        return super().step(action)


class CadenceRunner(OnPolicyRunner):
    """Basic transfer starts stage count at zero; cadence resume restores it."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.num_steps_per_env != 24:
            raise ValueError("Reviewed cadence fade requires 24 steps per PPO iteration")

    def save(self, path, infos=None):
        infos = dict(infos or {})
        infos["cadence_stage"] = {"control_steps": self.env.unwrapped.common_step_counter,
                                  "fade_steps": FADE_STEPS, "rollout_steps": 24}
        super().save(path, infos)

    def load(self, path, load_optimizer=True, map_location=None):
        infos = super().load(path, load_optimizer, map_location)
        stage = (infos or {}).get("cadence_stage")
        if stage:
            if stage["fade_steps"] != FADE_STEPS or stage["rollout_steps"] != 24:
                raise ValueError("Checkpoint cadence schedule differs from this run")
            self.env.unwrapped.common_step_counter = stage["control_steps"]
            self.current_learning_iteration += 1  # RSL stores the last completed index
        else:
            self.env.unwrapped.common_step_counter = 0
            self.current_learning_iteration = 0
        apply_fade(self.env.unwrapped)
        print(f"CADENCE RESUME: completed iterations={self.current_learning_iteration}, "
              f"stock reward fraction={fade_fraction(self.env.unwrapped.common_step_counter):.4f}", flush=True)
        return infos

    def log(self, locs, *args, **kwargs):
        super().log(locs, *args, **kwargs)
        fraction = fade_fraction(self.env.unwrapped.common_step_counter)
        self.writer.add_scalar("Cadence/stock_reward_fraction", fraction, locs["it"])
        for name, initial in INITIAL_WEIGHTS.items():
            self.writer.add_scalar(f"Cadence/weight_{name}", initial * fraction, locs["it"])
