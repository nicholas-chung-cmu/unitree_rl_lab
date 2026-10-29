"""Paired cadence scores with separately normalized support and swing budgets.

Pure score functions can be loaded without Isaac Sim for offline auditing.
Normalization is fixed from the reference, never from the policy's behavior.
"""

import torch


def raw_scores(target_contact, target_height, contact, height, swing_height=.03, std=.015):
    support_gate = target_contact.min(-1).values
    swing_gate = 1 - support_gate
    match = 1 - (2 * target_contact - 1).square() * (contact - target_contact).square()
    pattern = match.prod(-1)
    emphasis = (target_height / swing_height).square().sum(-1)
    activity = emphasis / emphasis.clamp_min(1)
    lift = (height / target_height.clamp_min(.015)).clamp(0, 1)
    quality = lift * (1-contact) * torch.exp(-((height-target_height)/std).square())
    a = target_height[:, 0] >= target_height[:, 1]
    pair = torch.where(a, torch.minimum(quality[:, 0], quality[:, 3]),
                       torch.minimum(quality[:, 1], quality[:, 2]))
    support = torch.where(a, contact[:, 1]*contact[:, 2], contact[:, 0]*contact[:, 3])
    return torch.stack((support_gate*contact.prod(-1), activity*pair*support,
                        support_gate*pattern, swing_gate*pattern), -1)


def normalized_scores(raw, normalizers):
    scaled = raw / normalizers.clamp_min(1e-8)
    # Give most phase credit to completing scheduled swings, not remaining planted.
    # Each state is normalized separately, so changing its duration does not grow its budget.
    return torch.stack((scaled[:, 0], scaled[:, 1], .2*scaled[:, 2]+.8*scaled[:, 3]), -1)


def env_scores(env):
    from .conga_reference import generate_reference
    term = env.command_manager.get_term("base_velocity")
    cfg = term.cfg.reference
    if not hasattr(term, "_paired_normalizers"):
        times = torch.arange(round(4/env.step_dt), device=env.device, dtype=torch.float64) * env.step_dt
        ideal = generate_reference(times, cfg)
        c = (ideal.contact >= .5 - 1e-6).double()
        term._paired_normalizers = raw_scores(ideal.contact, ideal.foot_height, c, ideal.foot_height,
                                              cfg.swing_height).mean(0).float()
    ref = term.reference
    _, height, contact = term.foot_state()
    return normalized_scores(raw_scores(ref.contact, ref.foot_height, contact.float(), height,
                                         cfg.swing_height), term._paired_normalizers)


def four_foot_support(env):
    return env_scores(env)[:, 0]


def paired_step(env):
    return env_scores(env)[:, 1]


def phase_matching(env):
    return env_scores(env)[:, 2]
