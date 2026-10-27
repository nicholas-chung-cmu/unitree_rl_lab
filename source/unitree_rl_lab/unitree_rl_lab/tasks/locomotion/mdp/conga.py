"""Isaac Lab command, observation, reward, and diagnostic adapters for conga."""

import torch
from dataclasses import field

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

from .conga_reference import FOOT_NAMES, CongaReferenceCfg, cap_velocity, generate_reference


class CongaVelocityCommand(UniformVelocityCommand):
    """Keep raw travel commands separate from the instantaneous enveloped target.

    Reference evaluation is pure and reads episode_length_buf, which Isaac Lab
    advances BEFORE rewards and observations. Command resampling never changes
    the clock. This also supports RSL-RL's initial randomized episode lengths.
    """

    def __init__(self, cfg, env):
        cfg.reference.validate()
        super().__init__(cfg, env)
        self.foot_ids, names = self.robot.find_bodies(list(FOOT_NAMES), preserve_order=True)
        self.sensor = env.scene["contact_forces"]
        self.sensor_ids, sensor_names = self.sensor.find_bodies(list(FOOT_NAMES), preserve_order=True)
        if tuple(names) != FOOT_NAMES or tuple(sensor_names) != FOOT_NAMES:
            raise ValueError(f"Conga requires ordered foot bodies {FOOT_NAMES}; got {names}, {sensor_names}")
        keys = (
            "velocity_xy_error_mps",
            "velocity_yaw_error_radps",
            "travel_velocity_error_mps",
            "commanded_travel_speed_mps",
            "contact_mae",
            "touchdown_error_s",
            "touchdowns_per_foot_hz",
            "reference_touchdowns_per_foot_hz",
            "kick_lateral_error_m",
            "kick_height_error_m",
            "body_height_error_m",
            "foot_slip_mps",
            "tilt_rad",
        )
        self.metrics = {key: torch.zeros(self.num_envs, device=self.device) for key in keys}
        self._sums = {key: torch.zeros_like(value) for key, value in self.metrics.items()}
        self._samples = torch.zeros(self.num_envs, device=self.device)
        self._kick_samples = torch.zeros_like(self._samples)
        self._travel_samples = torch.zeros_like(self._samples)
        self._touchdowns = torch.zeros_like(self._samples)
        self._expected_touchdowns = torch.zeros_like(self._samples)
        self._previous_contact = torch.ones(self.num_envs, 4, device=self.device)
        self._valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def reference(self):
        steps = getattr(self._env, "episode_length_buf", None)
        if steps is None:  # observation shape inference during manager setup
            steps = torch.zeros(self.num_envs, device=self.device)
        return generate_reference(steps * self._env.step_dt, self.cfg.reference)

    @property
    def raw_command(self):
        return cap_velocity(self.vel_command_b, self.cfg.reference.max_travel_speed)

    @property
    def command(self):
        return self.raw_command * self.reference.travel[:, None]

    def foot_state(self):
        pos = self.robot.data.body_pos_w[:, self.foot_ids]
        relative = pos - self.robot.data.root_pos_w[:, None, :]
        quat = self.robot.data.root_quat_w[:, None, :].expand(-1, 4, -1)
        body = quat_apply_inverse(quat.reshape(-1, 4), relative.reshape(-1, 3)).reshape(-1, 4, 3)
        clearance = pos[..., 2] - self._env.scene.env_origins[:, None, 2] - self.cfg.reference.foot_radius
        force = self.sensor.data.net_forces_w[:, self.sensor_ids].norm(dim=-1)
        contact = force > 1.0
        return body, clearance, contact

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        # Include pure yaw explicitly, instead of hoping independent XY samples
        # land exactly at zero. rel_standing_envs supplies true marching commands.
        turning = torch.rand(len(env_ids), device=self.device) < self.cfg.rel_turning_envs
        self.vel_command_b[env_ids[turning], :2] = 0.0
        self._update_command()

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device)
        result = super().reset(env_ids)
        for values in self._sums.values():
            values[env_ids] = 0
        for values in (
            self._samples,
            self._kick_samples,
            self._travel_samples,
            self._touchdowns,
            self._expected_touchdowns,
        ):
            values[env_ids] = 0
        self._previous_contact[env_ids] = 1
        self._valid[env_ids] = False
        return result

    def _update_metrics(self):
        ref = self.reference
        body, height, contact = self.foot_state()
        valid = self._env.episode_length_buf > 0
        first = self.sensor.compute_first_contact(self._env.step_dt)[:, self.sensor_ids]
        first = first & self._valid[:, None] & valid[:, None]
        expected = (self._previous_contact < 0.5) & (ref.contact >= 0.5) & self._valid[:, None]
        self._samples += valid
        self._touchdowns += first.sum(-1)
        self._expected_touchdowns += (expected & valid[:, None]).sum(-1)
        self._kick_samples += ref.kick_weight * valid
        traveling = (ref.travel > 0.99) & (self.raw_command[:, :2].norm(dim=-1) > 0.05) & valid
        self._travel_samples += traveling
        rows = torch.arange(self.num_envs, device=self.device)
        command = self.command
        velocity_error = (command[:, :2] - self.robot.data.root_lin_vel_b[:, :2]).norm(dim=-1)
        slip = self.robot.data.body_lin_vel_w[:, self.foot_ids, :2].norm(dim=-1)
        gravity_z = self.robot.data.projected_gravity_b[:, 2]
        values = {
            "velocity_xy_error_mps": velocity_error,
            "velocity_yaw_error_radps": (command[:, 2] - self.robot.data.root_ang_vel_b[:, 2]).abs(),
            "travel_velocity_error_mps": velocity_error * traveling,
            "commanded_travel_speed_mps": self.raw_command[:, :2].norm(dim=-1),
            "contact_mae": (contact.float() - ref.contact).abs().mean(-1),
            "touchdown_error_s": (ref.touchdown_distance * first).sum(-1),
            "kick_lateral_error_m": (body[rows, ref.kick_leg, 1] - ref.kick_y).abs() * ref.kick_weight,
            "kick_height_error_m": (
                (height[rows, ref.kick_leg] - ref.foot_height[rows, ref.kick_leg]).abs() * ref.kick_weight
            ),
            "body_height_error_m": (
                (self.robot.data.root_pos_w[:, 2] - self._env.scene.env_origins[:, 2] - ref.body_height).abs()
            ),
            "foot_slip_mps": (slip * contact).sum(-1) / contact.sum(-1).clamp_min(1),
            "tilt_rad": torch.acos((-gravity_z).clamp(-1, 1)),
        }
        for key, value in values.items():
            self._sums[key] += value * valid
            denominator = self._samples
            if key.startswith("kick_"):
                denominator = self._kick_samples
            elif key == "touchdown_error_s":
                denominator = self._touchdowns
            elif key == "travel_velocity_error_mps":
                denominator = self._travel_samples
            self.metrics[key][:] = self._sums[key] / denominator.clamp_min(1e-6)
        duration = (self._samples * self._env.step_dt).clamp_min(self._env.step_dt)
        self.metrics["touchdowns_per_foot_hz"][:] = self._touchdowns / (4 * duration)
        self.metrics["reference_touchdowns_per_foot_hz"][:] = self._expected_touchdowns / (4 * duration)
        self._previous_contact[:] = ref.contact
        self._valid[:] = valid


@configclass
class CongaVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = CongaVelocityCommand
    reference: CongaReferenceCfg = field(default_factory=CongaReferenceCfg)
    rel_turning_envs: float = 0.15


def conga_features(env):
    term = env.command_manager.get_term("base_velocity")
    ref = term.reference
    scales = torch.tensor(
        [term.cfg.reference.kick_scale * term.cfg.reference.kick_enabled, term.cfg.reference.bounce_scale],
        device=env.device,
    ).expand(env.num_envs, -1)
    return torch.cat((ref.phase_features(), term.raw_command, scales), dim=-1)


def contact_tracking(env):
    term = env.command_manager.get_term("base_velocity")
    ref = term.reference
    _, _, contact = term.foot_state()
    # Downweight uncertain boundaries smoothly instead of penalizing either
    # contact choice at a reference liftoff/touchdown crossing.
    confidence = (2 * ref.contact - 1).square()
    return 1 - (confidence * (contact.float() - ref.contact).square()).mean(-1)


def kick_tracking(env, lateral_std: float, height_std: float):
    term = env.command_manager.get_term("base_velocity")
    ref = term.reference
    body, height, _ = term.foot_state()
    rows = torch.arange(env.num_envs, device=env.device)
    error = ((body[rows, ref.kick_leg, 1] - ref.kick_y) / lateral_std).square()
    error += ((height[rows, ref.kick_leg] - ref.foot_height[rows, ref.kick_leg]) / height_std).square()
    return ref.kick_weight * torch.exp(-error)


def height_tracking(env, std: float):
    term = env.command_manager.get_term("base_velocity")
    height = term.robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return torch.exp(-((height - term.reference.body_height) / std).square())


def vertical_velocity_error(env):
    term = env.command_manager.get_term("base_velocity")
    return (term.robot.data.root_lin_vel_w[:, 2] - term.reference.body_velocity_z).square()


def cadence_tracking(env, timing_std: float):
    term = env.command_manager.get_term("base_velocity")
    ref = term.reference
    first = term.sensor.compute_first_contact(env.step_dt)[:, term.sensor_ids]
    score = torch.exp(-(ref.touchdown_distance / timing_std).square()) - 0.5
    # Signed event reward: timely touchdowns earn credit, extra off-phase
    # touchdowns cost reward. Missing events get no credit. Contact/clearance
    # rewards supply dense feedback and disfavor a stationary robot.
    return (first * score).sum(-1) / 4 * (env.episode_length_buf > 1) * (1 - ref.recovery)


def swing_clearance(env, std: float):
    term = env.command_manager.get_term("base_velocity")
    ref = term.reference
    _, height, _ = term.foot_state()
    # A neutral 0.5 recovery contact target must not reward lifting all four feet.
    swing = (1 - ref.contact) * (2 * ref.contact - 1).square()
    return (swing * torch.exp(-((height - ref.foot_height) / std).square())).sum(-1) / swing.sum(-1).clamp_min(1)


def marching_in_place(env, std: float):
    term = env.command_manager.get_term("base_velocity")
    stationary = term.raw_command.norm(dim=-1) < 0.05
    drift = term.robot.data.root_lin_vel_b[:, :2].square().sum(-1)
    drift += term.robot.data.root_ang_vel_b[:, 2].square()
    return stationary * torch.exp(-drift / 0.1**2) * swing_clearance(env, std)
