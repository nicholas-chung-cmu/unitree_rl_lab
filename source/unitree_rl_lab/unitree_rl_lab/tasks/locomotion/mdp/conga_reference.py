"""Pure Torch conga references; independent of Isaac Sim for numerical verification."""

import math
import torch
from dataclasses import dataclass

FOOT_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")


@dataclass
class CongaReferenceCfg:
    beat_period: float = 0.5
    kick_start: float = 1.25
    kick_duration: float = 0.25
    kick_lateral: float = 0.08
    kick_height: float = 0.05
    kick_scale: float = 1.0
    kick_enabled: bool = True
    bounce_peak_to_peak: float = 0.015
    bounce_scale: float = 1.0
    base_height: float = 0.32
    swing_height: float = 0.03
    contact_ramp: float = 0.04
    travel_ramp_down: float = 0.15
    travel_ramp_up: float = 0.20
    max_travel_speed: float = 1.25
    foot_radius: float = 0.022
    # FK at hip=+/-0.1, thigh=0.8, calf=-1.5, link lengths=0.213.
    nominal_front_y: float = 0.0465 + 0.0955 * math.cos(0.1) + 0.213 * (math.cos(0.8) + math.cos(-0.7)) * math.sin(0.1)

    def validate(self):
        if self.beat_period != 0.5 or self.kick_start != 1.25 or self.kick_duration != 0.25:
            raise ValueError("The finalized conga timing is 120 BPM, kick at 1.25 s for 0.25 s.")
        if not 0.0 <= self.kick_scale <= 1.0 or not 0.0 <= self.bounce_scale <= 1.0:
            raise ValueError("Curriculum scales must lie in [0, 1].")
        if not 0.0 < self.max_travel_speed <= 1.25:
            raise ValueError("Travel speed must be positive and no greater than 1.25 m/s.")
        if not 0.0 < self.contact_ramp <= 0.1:
            raise ValueError("Contact ramp must lie in (0, 0.1] seconds.")
        if min(self.travel_ramp_down, self.travel_ramp_up) <= 0:
            raise ValueError("Travel ramps must be positive.")


@dataclass
class CongaReference:
    phase: torch.Tensor
    beat_index: torch.Tensor
    beat_phase: torch.Tensor
    kick_leg: torch.Tensor
    kick_active: torch.Tensor
    kick_weight: torch.Tensor
    contact: torch.Tensor
    foot_height: torch.Tensor
    kick_y: torch.Tensor
    body_height: torch.Tensor
    body_velocity_z: torch.Tensor
    travel: torch.Tensor
    recovery: torch.Tensor
    touchdown_distance: torch.Tensor

    def phase_features(self):
        return torch.stack(
            (
                torch.sin(2 * math.pi * self.phase),
                torch.cos(2 * math.pi * self.phase),
                torch.sin(2 * math.pi * self.beat_phase),
                torch.cos(2 * math.pi * self.beat_phase),
                self.travel,
            ),
            dim=-1,
        )


def smoothstep(x):
    x = x.clamp(0.0, 1.0)
    return x.square() * (3.0 - 2.0 * x)


def swing_window(t, start, end, ramp):
    """C1 soft swing occupancy and zero-velocity endpoint clearance pulse."""
    weight = smoothstep((t - start) / ramp) * smoothstep((end - t) / ramp)
    u = ((t - start) / (end - start)).clamp(0.0, 1.0)
    pulse = torch.sin(math.pi * u).square()
    return weight, pulse


def cap_velocity(command, limit):
    """Cap XY vector magnitude without changing yaw or direction; never mutate input."""
    xy = command[:, :2]
    xy = xy * (limit / torch.linalg.vector_norm(xy, dim=-1, keepdim=True).clamp_min(1e-8)).clamp_max(1)
    return torch.cat((xy, command[:, 2:3]), dim=-1)


def generate_reference(time_s: torch.Tensor, cfg: CongaReferenceCfg) -> CongaReference:
    """Evaluate an external per-environment clock. No hidden state or speed dependency."""
    t = time_s.remainder(4.0)
    phrase_t = t.remainder(2.0)
    phrase = (t / 2.0).long()
    beat_phase = t.remainder(cfg.beat_period) / cfg.beat_period
    kick_leg = 1 - phrase  # FR first, FL next; foot ordering is explicit.
    swing = torch.zeros((*t.shape, 4), device=t.device, dtype=t.dtype)
    height = torch.zeros_like(swing)
    distance = torch.full_like(swing, 4.0)
    diagonal_a = torch.tensor([True, False, False, True], device=t.device)

    # Compute periodic distance to all touchdown events in the full cycle. Soft
    # contact crosses 0.5 halfway through the touchdown ramp, defining event time.
    events = []
    if cfg.kick_enabled:
        windows = ((0.04, 0.40), (0.54, 0.90), (1.02, 1.23))
        for p in range(2):
            for i, (start, end) in enumerate(windows):
                pair = diagonal_a if (i + p) % 2 == 0 else ~diagonal_a
                w, pulse = swing_window(t, start + 2 * p, end + 2 * p, cfg.contact_ramp)
                swing += w[:, None] * pair
                height += cfg.swing_height * pulse[:, None] * pair
                events.append((end + 2 * p - cfg.contact_ramp / 2, pair))
            pair = torch.arange(4, device=t.device) == (1 - p)
            events.append((1.5 + 2 * p - cfg.contact_ramp / 2, pair))
    else:
        for i in range(8):
            start, end = i * 0.5 + 0.04, i * 0.5 + 0.40
            pair = diagonal_a if i % 2 == 0 else ~diagonal_a
            w, pulse = swing_window(t, start, end, cfg.contact_ramp)
            swing += w[:, None] * pair
            height += cfg.swing_height * pulse[:, None] * pair
            events.append((end - cfg.contact_ramp / 2, pair))
    for event_t, feet in events:
        d = ((t - event_t + 2.0).remainder(4.0) - 2.0).abs()
        distance = torch.where(feet, torch.minimum(distance, d[:, None]), distance)

    end = cfg.kick_start + cfg.kick_duration
    kick_weight, pulse = swing_window(phrase_t, cfg.kick_start, end, cfg.contact_ramp)
    kick_weight = kick_weight * cfg.kick_enabled
    pulse = pulse * cfg.kick_enabled
    mask = torch.arange(4, device=t.device)[None, :] == kick_leg[:, None]
    swing += kick_weight[:, None] * mask
    # During introduction, blend normal clearance into full kick clearance.
    kick_height = cfg.swing_height + cfg.kick_scale * (cfg.kick_height - cfg.swing_height)
    height += kick_height * pulse[:, None] * mask
    sign = torch.where(kick_leg == 0, 1.0, -1.0)
    kick_y = sign * (cfg.nominal_front_y + cfg.kick_lateral * cfg.kick_scale * pulse)
    travel = 1.0 - smoothstep((phrase_t - (cfg.kick_start - cfg.travel_ramp_down)) / cfg.travel_ramp_down)
    travel += smoothstep((phrase_t - end) / cfg.travel_ramp_up)
    if not cfg.kick_enabled:
        travel = torch.ones_like(t)
    # Require the kick touchdown, then let the policy choose recovery contacts
    # while travel resumes. Forcing four planted feet here conflicts with motion.
    recovery = smoothstep((phrase_t - end) / 0.10) * smoothstep((2.0 - phrase_t) / 0.08)
    recovery = recovery * cfg.kick_enabled
    contact = (1.0 - swing) * (1.0 - recovery[:, None]) + 0.5 * recovery[:, None]
    omega = 2 * math.pi / cfg.beat_period
    amplitude = cfg.bounce_peak_to_peak * cfg.bounce_scale / 2
    return CongaReference(
        phase=t / 4.0,
        beat_index=(t / cfg.beat_period).long(),
        beat_phase=beat_phase,
        kick_leg=kick_leg,
        kick_active=(phrase_t >= cfg.kick_start) & (phrase_t < end) & cfg.kick_enabled,
        kick_weight=kick_weight,
        contact=contact,
        foot_height=height,
        kick_y=kick_y,
        body_height=cfg.base_height + amplitude * torch.sin(omega * t),
        body_velocity_z=amplitude * omega * torch.cos(omega * t),
        travel=travel,
        recovery=recovery,
        touchdown_distance=distance,
    )
