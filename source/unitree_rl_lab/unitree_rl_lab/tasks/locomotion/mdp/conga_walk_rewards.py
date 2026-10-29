"""Experimental walking rewards; the final conga rewards remain unchanged."""

import torch


def contact_error(env):
    term = env.command_manager.get_term("base_velocity")
    ref = term.reference
    _, _, contact = term.foot_state()
    return ((2 * ref.contact - 1).square() * (contact.float() - ref.contact).square()).mean(-1)


def swing_quality(env, std=0.015):
    term = env.command_manager.get_term("base_velocity")
    ref = term.reference
    _, height, contact = term.foot_state()
    # Concentrate on the middle of swing, and require actual lift and unloading.
    weight = (ref.foot_height / term.cfg.reference.swing_height).square()
    lift = (height / ref.foot_height.clamp_min(0.015)).clamp(0, 1)
    quality = lift * (~contact) * torch.exp(-((height - ref.foot_height) / std).square())
    return (weight * quality).sum(-1) / weight.sum(-1).clamp_min(1)


def missed_swing(env):
    term = env.command_manager.get_term("base_velocity")
    ref = term.reference
    _, height, contact = term.foot_state()
    weight = (ref.foot_height / term.cfg.reference.swing_height).square()
    shortfall = ((ref.foot_height - height) / ref.foot_height.clamp_min(0.015)).clamp(0, 1)
    # A planted foot is a missing swing even if compliant geometry reports height.
    deficit = torch.maximum(shortfall.square(), contact.float())
    return (weight * deficit).sum(-1) / weight.sum(-1).clamp_min(1)


def touchdown_events(env, timing_std=0.08, min_air_time=0.08):
    term = env.command_manager.get_term("base_velocity")
    ref = term.reference
    first = term.sensor.compute_first_contact(env.step_dt)[:, term.sensor_ids]
    score = torch.exp(-(ref.touchdown_distance / timing_std).square()) - 0.5
    valid_swing = term.sensor.data.last_air_time[:, term.sensor_ids] >= min_air_time
    # Contact chatter cannot collect positive event credit. Off-phase events
    # retain their cost. RewardManager multiplies by dt, so cancel it for events.
    score = torch.where(valid_swing, score, score.clamp_max(0))
    return (first * score).mean(-1) / env.step_dt * (env.episode_length_buf > 1)


def marching(env):
    term = env.command_manager.get_term("base_velocity")
    stationary = term.raw_command.norm(dim=-1) < 0.05
    drift = term.robot.data.root_lin_vel_b[:, :2].square().sum(-1)
    drift += term.robot.data.root_ang_vel_b[:, 2].square()
    return stationary * torch.exp(-drift / 0.1**2) * swing_quality(env)


def nominal_height(env, std=0.04):
    term = env.command_manager.get_term("base_velocity")
    height = term.robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return torch.exp(-((height - term.cfg.reference.base_height) / std).square())
