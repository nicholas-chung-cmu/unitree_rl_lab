"""Checkpointed 50–200 iteration early phase reward and command ramp."""
from isaaclab.envs import ManagerBasedRLEnv
from rsl_rl.runners import OnPolicyRunner


def schedule(step):
    ramp = min(1.0, max(0.0, (step - 50 * 24) / (150 * 24)))
    return {
        'cadence_phase': .05 + .20 * ramp,
        'cadence_pair': .10 * ramp,
        'joint_pos': -.7 * (1-ramp),
        'feet_air_time': .1 * (1-ramp),
        'air_time_variance': -1.0 * (1-ramp),
    }, .1 + .4*ramp, .1 + .15*ramp


def apply_schedule(env):
    weights, vx, vy = schedule(getattr(env, 'early_frozen_step', env.common_step_counter))
    for name, weight in weights.items():
        cfg = env.reward_manager.get_term_cfg(name)
        cfg.weight = weight
        env.reward_manager.set_term_cfg(name, cfg)
    command = env.command_manager.get_term('base_velocity')
    command.cfg.ranges.lin_vel_x = (-vx, vx)
    command.cfg.ranges.lin_vel_y = (-vy, vy)
    return weights, vx, vy


class EarlyEnv(ManagerBasedRLEnv):
    def step(self, action):
        apply_schedule(self)
        return super().step(action)


class EarlyRunner(OnPolicyRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.num_steps_per_env != 24:
            raise ValueError('Early shaping requires 24-step PPO rollouts')

    def save(self, path, infos=None):
        infos = dict(infos or {})
        infos['early_stage'] = {'control_steps': self.env.unwrapped.common_step_counter,
                                'schedule_version': 1, 'rollout_steps': 24}
        super().save(path, infos)

    def load(self, path, load_optimizer=True, map_location=None):
        infos = super().load(path, load_optimizer, map_location)
        stage = (infos or {}).get('early_stage')
        if not stage or stage['schedule_version'] != 1 or stage['rollout_steps'] != 24:
            raise ValueError('Early shaping resume requires a matching early-stage checkpoint')
        self.env.unwrapped.common_step_counter = stage['control_steps']
        self.current_learning_iteration += 1
        apply_schedule(self.env.unwrapped)
        print(f'EARLY RESUME: completed iterations={self.current_learning_iteration}', flush=True)
        return infos

    def log(self, locs, *args, **kwargs):
        super().log(locs, *args, **kwargs)
        weights, vx, vy = schedule(self.env.unwrapped.common_step_counter)
        for name, weight in weights.items():
            self.writer.add_scalar(f'Early/weight_{name}', weight, locs['it'])
        self.writer.add_scalar('Early/max_vx', vx, locs['it'])
        self.writer.add_scalar('Early/max_vy', vy, locs['it'])
