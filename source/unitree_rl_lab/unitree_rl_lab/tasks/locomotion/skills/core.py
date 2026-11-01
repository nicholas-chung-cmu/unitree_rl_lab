"""Vectorized skill goals, completion feedback and deterministic musical scheduling.

Foot order is FL, FR, RL, RR. Positions are world-frame sole positions; yaw
and displacement goals are relative to the body heading at skill dispatch.
No state in this module changes physical episode ages or resets robot physics.
"""
from dataclasses import dataclass
from enum import IntEnum
import math

import torch
import torch.nn.functional as F


class Skill(IntEnum):
    STEP = 0
    KICK = 1
    STABILIZE = 2


class Status(IntEnum):
    RUNNING = 0
    SUCCESS = 1
    TIMEOUT = 2
    FAILURE = 3


@dataclass
class SkillConfig:
    bpm: float = 90.0
    max_speed: float = 0.5
    max_yaw_rate: float = 0.8
    kick_duration: float = 0.25
    kick_lateral: float = 0.08
    kick_height: float = 0.05
    step_clearance: float = 0.04
    contact_on: float = 5.0
    contact_off: float = 2.0
    confirm_time: float = 0.04
    foot_tolerance: float = 0.045
    body_tolerance: float = 0.06
    heading_tolerance: float = 0.12
    kick_tolerance: float = 0.015
    endpoint_speed: float = 0.25
    stable_tilt: float = 0.30
    stable_angular_speed: float = 1.5
    timeout_grace: float = 0.4
    # Musical beats are LANDING targets. Dispatch one beat ahead by default.
    landing_offsets: tuple = (0.0, 1.0, 2.0)
    step_lead_beats: tuple = (1.0, 1.0, 1.0)
    touchdown_tolerance: float | None = None  # relaxed in A--D; tightened in E/F
    bounce_amplitude: float = 0.0  # Stage F only, ramp manually to .0075 m.

    def __post_init__(self):
        if self.bpm <= 0 or not 0 < self.max_speed <= 0.5 or self.max_yaw_rate <= 0:
            raise ValueError("Invalid tempo or speed limits")
        if not 0 <= self.landing_offsets[0] < self.landing_offsets[1] < self.landing_offsets[2] < 2.5:
            raise ValueError("STEP landing beats must precede the 2.5-beat kick")
        if len(self.step_lead_beats) != 3 or min(self.step_lead_beats) <= 0:
            raise ValueError("STEP lead times must be positive")
        if self.kick_duration <= 0 or self.kick_duration >= 1.5 * self.beat:
            raise ValueError("Kick leaves no post-kick landing window")
        if not 0 <= self.contact_off < self.contact_on or self.confirm_time <= 0:
            raise ValueError("Invalid contact hysteresis")

    @property
    def beat(self):
        return 60.0 / self.bpm

    @property
    def phrase(self):
        return 4 * self.beat


@dataclass
class State:
    body: torch.Tensor       # (N,3), world
    yaw: torch.Tensor        # (N,)
    feet: torch.Tensor       # (N,4,3), sole positions, world
    force: torch.Tensor      # (N,4), force norms in N
    foot_velocity: torch.Tensor  # (N,4,3), world
    tilt: torch.Tensor       # (N,), radians
    angular_speed: torch.Tensor  # (N,), roll/pitch speed
    ground: torch.Tensor     # (N,), flat local ground z


def rotate_xy(x, yaw):
    c, s = yaw.cos(), yaw.sin()
    return torch.stack((c*x[..., 0]-s*x[..., 1], s*x[..., 0]+c*x[..., 1]), -1)


def wrap(x):
    return torch.atan2(x.sin(), x.cos())


def smooth(x):
    x = x.clamp(0, 1)
    return x*x*x*(10 + x*(-15 + 6*x))


class SkillExecutor:
    """One shared STEP policy contract for normal and raised-leg entries.

    ``begin`` sets goals; ``update`` consumes exactly one physics/control sample.
    A terminal status is latched until another begin. Event rewards occur once.
    """
    def __init__(self, n, device, cfg=None):
        self.cfg = cfg or SkillConfig()
        self.n, self.device = n, device
        def zeros(*shape, dtype=torch.float32):
            return torch.zeros(n, *shape, device=device, dtype=dtype)
        self.kind = zeros(dtype=torch.long) + Skill.STABILIZE
        self.status = zeros(dtype=torch.long)
        self.selected = zeros(4, dtype=torch.bool)
        self.entry_raised = zeros(4, dtype=torch.bool)
        self.contact = zeros(4, dtype=torch.bool)
        self.contact_age = zeros(4)
        self.air_age = zeros(4)
        self.lifted = zeros(4, dtype=torch.bool)
        self.elapsed = zeros()
        self.duration = zeros() + 1
        self.kick_lateral = zeros() + self.cfg.kick_lateral
        self.kick_height = zeros() + self.cfg.kick_height
        self.start_body, self.start_feet = zeros(3), zeros(4, 3)
        self.start_yaw, self.goal = zeros(), zeros(3)
        self.target_feet = zeros(4, 3)
        self.nominal_xy = zeros(4, 2)
        self.success_event, self.failure_event = zeros(), zeros()
        self.premature = zeros()
        self.swing_count = zeros(4)
        self.landing_count = zeros(4)
        self.swing_active = zeros(4, dtype=torch.bool)
        self.support_lost = zeros(dtype=torch.bool)
        self.touchdowns = zeros(4)
        self.touchdown_time = zeros(4) - 1
        self.support_departures = zeros()
        self.previous_contact = zeros(4, dtype=torch.bool)
        self.metrics = {}

    def reset(self, ids, state):
        self.contact[ids] = False
        self.previous_contact[ids] = False
        self.contact_age[ids] = 0
        self.air_age[ids] = 0
        self.nominal_xy[ids] = rotate_xy(
            state.feet[ids, :, :2]-state.body[ids, None, :2], -state.yaw[ids, None])
        self.begin(ids, Skill.STABILIZE, torch.ones(len(ids), 4, dtype=torch.bool, device=self.device),
                   torch.zeros(len(ids), 3, device=self.device), 0.3, state, validate=False)

    @property
    def confirmed(self):
        return self.contact & (self.contact_age >= self.cfg.confirm_time - 1e-6)

    def begin(self, ids, kind, selected, goal, duration, state, validate=True,
              lateral_displacement=None, foot_height=None):
        if len(ids) == 0:
            return
        selected = torch.as_tensor(selected, device=self.device, dtype=torch.bool).expand(len(ids), 4)
        raised = ~self.confirmed[ids]
        if validate:
            four = self.confirmed[ids].all(-1)
            one_front = raised.sum(-1).eq(1) & raised[:, :2].any(-1)
            contains = (~raised | selected).all(-1)
            if kind == Skill.STEP:
                diagonal = ((selected == selected.new_tensor([1, 0, 0, 1])).all(-1)
                            | (selected == selected.new_tensor([0, 1, 1, 0])).all(-1))
                valid = diagonal & (four | (one_front & contains))
            elif kind == Skill.KICK:
                valid = four & selected.sum(-1).eq(1) & selected[:, :2].any(-1)
            else:
                valid = (four | one_front) & (selected == self.confirmed[ids]).all(-1)
            if not bool(valid.all()):
                raise ValueError("Invalid skill entry contact configuration")
        duration = torch.as_tensor(duration, device=self.device).expand(len(ids))
        if bool((duration <= 0).any()):
            raise ValueError("Skill duration must be positive")
        self.kind[ids], self.status[ids] = int(kind), int(Status.RUNNING)
        self.selected[ids], self.entry_raised[ids] = selected, raised
        self.elapsed[ids], self.duration[ids] = 0, duration
        self.start_body[ids], self.start_yaw[ids] = state.body[ids], state.yaw[ids]
        self.start_feet[ids], self.goal[ids] = state.feet[ids], goal
        self.lifted[ids] = raised & selected
        self.premature[ids] = 0
        self.swing_count[ids] = (raised & selected).float()
        self.landing_count[ids] = 0
        self.swing_active[ids] = raised & selected
        self.support_lost[ids] = False
        self.touchdowns[ids] = 0
        self.touchdown_time[ids] = -1
        self.support_departures[ids] = 0
        self.success_event[ids], self.failure_event[ids] = 0, 0
        self.target_feet[ids] = state.feet[ids]
        if kind == Skill.STEP:
            endpoint = (state.body[ids, None, :2]
                        + rotate_xy(goal[:, None, :2], state.yaw[ids, None])
                        + rotate_xy(self.nominal_xy[ids], state.yaw[ids, None] + goal[:, None, 2]))
            self.target_feet[ids, :, :2] = torch.where(selected[..., None], endpoint, state.feet[ids, :, :2])
            self.target_feet[ids, :, 2] = state.ground[ids, None]
        elif kind == Skill.KICK:
            lateral_goal = self.cfg.kick_lateral if lateral_displacement is None else lateral_displacement
            height_goal = self.cfg.kick_height if foot_height is None else foot_height
            self.kick_lateral[ids] = torch.as_tensor(lateral_goal, device=self.device).expand(len(ids))
            self.kick_height[ids] = torch.as_tensor(height_goal, device=self.device).expand(len(ids))
            if bool((self.kick_lateral[ids] < 0).any() | (self.kick_height[ids] <= 0).any()):
                raise ValueError("Kick displacement must be outward and height positive")
            # Body-relative y endpoint: nominal +/- .08; x remains unconstrained.
            lateral = self.nominal_xy[ids].clone()
            lateral[:, 0, 1] += self.kick_lateral[ids]
            lateral[:, 1, 1] -= self.kick_lateral[ids]
            endpoint = state.body[ids, None, :2] + rotate_xy(lateral, state.yaw[ids, None])
            self.target_feet[ids, :, :2] = torch.where(selected[..., None], endpoint, state.feet[ids, :, :2])
            self.target_feet[ids, :, 2] = torch.where(selected, state.ground[ids, None]+self.kick_height[ids, None],
                                                   state.feet[ids, :, 2])

    def reference(self, state):
        p = (self.elapsed / self.duration).clamp(0, 1)
        blend = smooth(p)
        target = self.target_feet.clone()
        kick = self.kind == Skill.KICK
        # Refresh kick endpoint in current body heading/position; fore-aft is free.
        relative = rotate_xy(state.feet[..., :2]-state.body[:, None, :2], -state.yaw[:, None])
        kick_xy = relative.clone()
        signs = state.body.new_tensor([1, -1, 1, -1])
        kick_xy[..., 1] = self.nominal_xy[..., 1] + signs*self.kick_lateral[:, None]
        endpoint = state.body[:, None, :2] + rotate_xy(kick_xy, state.yaw[:, None])
        target[..., :2] = torch.where((kick[:, None] & self.selected)[..., None], endpoint, target[..., :2])
        ref = self.start_feet + blend[:, None, None]*(target-self.start_feet)
        step = self.kind == Skill.STEP
        ref[..., 2] += (step[:, None] & self.selected)*self.cfg.step_clearance*torch.sin(math.pi*p[:, None]).square()
        hold = self.kind == Skill.STABILIZE
        ref = torch.where(hold[:, None, None], self.start_feet, ref)
        # KICK imposes transverse/height goals only, never a fore-aft target.
        relative_ref = rotate_xy(ref[..., :2]-state.body[:, None, :2], -state.yaw[:, None])
        relative_ref[..., 0] = torch.where(kick[:, None] & self.selected, relative[..., 0], relative_ref[..., 0])
        ref[..., :2] = state.body[:, None, :2] + rotate_xy(relative_ref, state.yaw[:, None])
        return ref, target

    def update(self, state, dt):
        self.success_event.zero_()
        self.failure_event.zero_()
        old_contact = self.contact.clone()
        old_confirmed = self.confirmed.clone()
        self.contact = torch.where(self.contact, state.force > self.cfg.contact_off, state.force >= self.cfg.contact_on)
        self.contact_age = torch.where(self.contact, self.contact_age+dt, 0)
        self.air_age = torch.where(~self.contact, self.air_age+dt, 0)
        running = self.status == Status.RUNNING
        self.elapsed += dt * running
        touchdown = self.confirmed & ~old_confirmed & running[:, None]
        self.touchdowns += touchdown
        # Report the onset of a subsequently confirmed contact, not the later
        # confirmation sample, so 40 ms filtering does not shift the musical beat.
        self.touchdown_time = torch.where(touchdown, self.elapsed[:, None]-self.contact_age+dt, self.touchdown_time)
        support = torch.where((self.kind == Skill.STABILIZE)[:, None], self.selected, ~self.selected)
        self.support_departures += (old_contact & ~self.contact & support).sum(-1)*running
        p = self.elapsed/self.duration
        clearance = state.feet[..., 2]-state.ground[:, None]
        self.lifted |= (self.air_age >= self.cfg.confirm_time) & (clearance > .008) & self.selected
        # Count physical swings, not single-frame force/contact chatter. A swing
        # stays latched until a confirmed landing, preventing clearance jitter
        # from manufacturing extra departures within one airborne interval.
        verified_air = ((self.air_age >= self.cfg.confirm_time-1e-6)
                        & (clearance > .008) & self.selected & running[:, None])
        departure = verified_air & ~self.swing_active
        self.swing_count += departure
        self.swing_active |= departure
        landing = touchdown & self.swing_active & self.selected
        self.landing_count += landing
        self.swing_active &= ~landing
        self.support_lost |= ((self.air_age >= self.cfg.confirm_time-1e-6)
                              & support).any(-1) & running
        premature = (self.contact & ~old_contact & self.entry_raised & self.selected).any(-1) & (p < .6)
        self.premature += premature * running * (self.kind == Skill.STEP)
        ref, target = self.reference(state)
        error = (state.feet-target).norm(dim=-1)
        placement = torch.where(self.selected, error, 0).amax(-1)
        displacement = rotate_xy(state.body[:, :2]-self.start_body[:, :2], -self.start_yaw)
        heading = wrap(state.yaw-self.start_yaw)
        body_error = (displacement-self.goal[:, :2]).norm(dim=-1)
        heading_error = wrap(heading-self.goal[:, 2]).abs()
        stable = ((state.tilt < self.cfg.stable_tilt) & (state.angular_speed < self.cfg.stable_angular_speed))
        all_contact = self.confirmed.all(-1)
        step_ok = (all_contact & (self.lifted | ~self.selected).all(-1)
                   & (placement < self.cfg.foot_tolerance) & (body_error < self.cfg.body_tolerance)
                   & (heading_error < self.cfg.heading_tolerance) & (p >= .7))
        step_ok &= (((self.swing_count == 1) | ~self.selected).all(-1)
                    & ((self.landing_count == 1) | ~self.selected).all(-1)
                    & ~self.support_lost)
        invalid_step = ((self.kind == Skill.STEP)
                        & (((self.swing_count > 1) & self.selected).any(-1) | self.support_lost))
        touchdown_error = (self.touchdown_time-self.duration[:, None]).abs()
        if self.cfg.touchdown_tolerance is not None:
            step_ok &= ((touchdown_error <= self.cfg.touchdown_tolerance) | ~self.selected).all(-1)
        support_ok = (self.confirmed | self.selected).all(-1)
        airborne = ((~self.contact & (clearance > .02)) | ~self.selected).all(-1)
        kick_speed = (state.foot_velocity.norm(dim=-1)*self.selected).amax(-1)
        kick_ok = (support_ok & airborne & (placement < self.cfg.kick_tolerance)
                   & (kick_speed < self.cfg.endpoint_speed) & (p >= .9))
        # STABILIZE preserves the exact entry support mask and airborne pose.
        hold_error = (state.feet-self.start_feet).norm(dim=-1).amax(-1)
        hold_ok = ((self.confirmed == self.selected).all(-1) & (hold_error < self.cfg.foot_tolerance) & (p >= 1))
        ok = stable & torch.where(self.kind == Skill.STEP, step_ok,
                                 torch.where(self.kind == Skill.KICK, kick_ok, hold_ok))
        fallen = (state.tilt > .65) | ((state.body[:, 2]-state.ground) < .16)
        timeout = self.elapsed > self.duration + self.cfg.timeout_grace
        new = torch.where(fallen | invalid_step, int(Status.FAILURE), torch.where(ok, int(Status.SUCCESS),
                          torch.where(timeout, int(Status.TIMEOUT), int(Status.RUNNING))))
        self.status = torch.where(running, new, self.status)
        self.success_event = (running & (self.status == Status.SUCCESS)).float()
        self.failure_event = (running & (self.status >= Status.TIMEOUT)).float()
        slip = (state.foot_velocity[..., :2].norm(dim=-1)*self.contact).sum(-1)/self.contact.sum(-1).clamp_min(1)
        self.metrics = dict(duration=self.elapsed.clone(), foot_error=placement, body_error=body_error,
                            heading_error=heading_error, displacement=displacement, heading_change=heading,
                            contacts=self.contact.clone(), raised=(clearance > .02) & ~self.contact,
                            tilt=state.tilt, slip=slip, premature_touchdowns=self.premature.clone(),
                            timing_error=self.elapsed-self.duration, reference_error=(state.feet-ref).norm(dim=-1),
                            contact_age=self.contact_age.clone(), fall=fallen,
                            base_height=state.body[:, 2]-state.ground, foot_clearance=clearance,
                            selected=self.selected.clone(), lifted=self.lifted.clone(),
                            verified_swings=self.swing_count.clone(), verified_landings=self.landing_count.clone(),
                            support_lost=self.support_lost.clone(),
                            touchdowns=self.touchdowns.clone(), touchdown_time=self.touchdown_time.clone(),
                            touchdown_timing_error=torch.where(self.selected & (self.touchdown_time >= 0), touchdown_error, 0).amax(-1),
                            missing_touchdowns=(self.selected & (self.touchdown_time < 0)).sum(-1),
                            support_departures=self.support_departures.clone(),
                            contact_sequence_correct=step_ok & (self.premature == 0) & (self.support_departures == 0),
                            kick_height_error=((clearance-self.kick_height[:, None]).abs()*self.selected).amax(-1),
                            kick_lateral_error=((state.feet-target)[..., :2].norm(dim=-1)*self.selected).amax(-1))
        self.previous_contact.copy_(self.contact)

    def features(self, state):
        ref, target = self.reference(state)
        relative_target = target-state.body[:, None]
        relative_target[..., :2] = rotate_xy(relative_target[..., :2], -state.yaw[:, None])
        relative_ref = ref-state.feet
        relative_ref[..., :2] = rotate_xy(relative_ref[..., :2], -state.yaw[:, None])
        displacement = rotate_xy(state.body[:, :2]-self.start_body[:, :2], -self.start_yaw)
        remaining_goal = self.goal.clone()
        remaining_goal[:, :2] -= displacement
        remaining_goal[:, 2] = wrap(self.goal[:, 2] - wrap(state.yaw-self.start_yaw))
        return torch.cat((F.one_hot(self.kind, 3), self.selected, self.goal, remaining_goal,
                          relative_target.flatten(1), relative_ref.flatten(1), self.contact,
                          self.entry_raised, ((state.feet[..., 2]-state.ground[:, None] > .02) & ~self.contact),
                          self.duration[:, None], (self.duration-self.elapsed)[:, None],
                          (self.elapsed/self.duration).clamp(0, 2)[:, None]), -1).float()


class DancePlanner:
    """Hybrid event/clock planner. No unsafe scheduled interruption of a skill.

    Beats 1, 2 and 3 are touchdown targets. STEP commands precede their beats.
    After KICK, the next phrase's first STEP starts at kick completion and lands
    on that phrase's first beat, with approximately .75 s available.
    Late skills recover before resynchronizing; kicks are skipped, never caught up.
    """
    def __init__(self, n, device, cfg=None):
        self.cfg = cfg or SkillConfig()
        self.tick = torch.zeros(n, dtype=torch.long, device=device)
        self.phrase = torch.zeros_like(self.tick)
        self.slot = torch.zeros_like(self.tick)
        self.due = torch.zeros(n, dtype=torch.float64, device=device)
        self.pending_raised = torch.full_like(self.tick, -1)
        self.recovering = torch.zeros(n, dtype=torch.bool, device=device)
        self.command = torch.zeros(n, 3, device=device)
        self.dispatch_error = torch.zeros(n, device=device)
        self.due[:] = self.first_dispatch

    @property
    def first_dispatch(self):
        return (self.cfg.landing_offsets[0]-self.cfg.step_lead_beats[0])*self.cfg.beat

    def time(self, dt):
        # A startup pre-roll allows the first STEP to land on musical beat 1.
        return self.tick.double()*dt + self.first_dispatch

    def reset(self, ids):
        for value in (self.tick, self.phrase, self.slot, self.due, self.dispatch_error):
            value[ids] = 0
        self.pending_raised[ids] = -1
        self.recovering[ids] = False
        self.due[ids] = self.first_dispatch

    def set_velocity(self, velocity):
        self.command.copy_(velocity)
        self.command[:, :2] *= (self.cfg.max_speed/self.command[:, :2].norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp(max=1)
        self.command[:, 2].clamp_(-self.cfg.max_yaw_rate, self.cfg.max_yaw_rate)

    def advance(self, executor, state, dt):
        self.tick += 1
        now = self.time(dt)  # no accumulated floating-point phase increments
        done = executor.status != Status.RUNNING
        failed = executor.status >= Status.TIMEOUT
        self.recovering |= failed
        # A successful kick must hand off directly, including when it finished late.
        kick_done = done & (executor.kind == Skill.KICK)
        valid_raised = (~executor.confirmed).sum(-1).eq(1) & (~executor.confirmed[:, :2]).any(-1)
        for leg in (0, 1):  # fixed skill categories, never loops over environments
            mask = kick_done & valid_raised & executor.selected[:, leg]
            ids = mask.nonzero().flatten()
            if len(ids):
                self.phrase[ids] += 1
                self.slot[ids] = 1
                landing = self.phrase[ids].double()*self.cfg.phrase + self.cfg.landing_offsets[0]*self.cfg.beat
                duration = (landing-now[ids]).float()
                # A late kick may leave too little time. Give recovery a full
                # beat, then resynchronize; never compress the landing unsafely.
                late = duration < .5*self.cfg.beat
                duration = torch.where(late, self.cfg.beat, duration)
                self.recovering[ids] |= late
                self.due[ids] = self.phrase[ids].double()*self.cfg.phrase + (
                    self.cfg.landing_offsets[1]-self.cfg.step_lead_beats[1])*self.cfg.beat
                self.dispatch_error[ids] = (now[ids]-((self.phrase[ids].double()-1)*self.cfg.phrase
                                                        + 2.5*self.cfg.beat+self.cfg.kick_duration)).float()
                self.pending_raised[ids] = leg
                pair = executor.selected.new_tensor([1, 0, 0, 1] if leg == 0 else [0, 1, 1, 0])
                executor.begin(ids, Skill.STEP, pair, self.command[ids]*duration[:, None], duration, state)
                done[ids] = False
        # Failure recovery: only known valid support states can accept a new skill.
        four = executor.confirmed.all(-1)
        recover_four = done & self.recovering & four
        ids = recover_four.nonzero().flatten()
        if len(ids):
            self.phrase[ids] = torch.floor((now[ids]-self.first_dispatch)/self.cfg.phrase).long()+1
            self.slot[ids] = 0
            self.due[ids] = self.phrase[ids].double()*self.cfg.phrase+self.first_dispatch
            self.recovering[ids] = False
        ready = done & four & (now >= self.due) & ~self.recovering
        # Do not compress a late step to hit a kick. Skip to a future phrase.
        overdue = ready & ((now-self.due) > .10)
        # Initial stabilization is deliberately outside the musical clock.
        overdue &= self.tick > round(.5/dt)
        ids = overdue.nonzero().flatten()
        if len(ids):
            self.phrase[ids] = torch.floor((now[ids]-self.first_dispatch)/self.cfg.phrase).long()+1
            self.slot[ids] = 0
            self.due[ids] = self.phrase[ids].double()*self.cfg.phrase+self.first_dispatch
        ready &= ~overdue
        dispatch_slot = self.slot.clone()
        for slot in (0, 1, 2, 3):
            ids = (ready & (dispatch_slot == slot)).nonzero().flatten()
            if not len(ids):
                continue
            self.dispatch_error[ids] = (now[ids]-self.due[ids]).float()
            if slot == 3:
                leg = 1-self.phrase[ids].remainder(2)
                selected = F.one_hot(leg, 4).bool()
                executor.begin(ids, Skill.KICK, selected, torch.zeros_like(self.command[ids]), self.cfg.kick_duration, state)
            else:
                pair = (self.phrase[ids]+slot).remainder(2)
                selected = (torch.arange(4, device=ids.device)[None, :] == pair[:, None]) | (
                    torch.arange(4, device=ids.device)[None, :] == (3-pair)[:, None])
                landing = self.phrase[ids].double()*self.cfg.phrase+self.cfg.landing_offsets[slot]*self.cfg.beat
                duration = (landing-now[ids]).float().clamp_min(dt)
                executor.begin(ids, Skill.STEP, selected, self.command[ids]*duration[:, None], duration, state)
                self.slot[ids] += 1
                offset = (self.cfg.landing_offsets[slot+1]-self.cfg.step_lead_beats[slot+1]) if slot < 2 else 2.5
                self.due[ids] = self.phrase[ids].double()*self.cfg.phrase+offset*self.cfg.beat
        # Early STEP completion holds measured four-foot support, not the
        # unfinished swing reference. Feedback was consumed before dispatch.
        hold = (executor.status != Status.RUNNING) & four & (now < self.due)
        ids = hold.nonzero().flatten()
        if len(ids):
            executor.begin(ids, Skill.STABILIZE, executor.confirmed[ids],
                           torch.zeros_like(self.command[ids]), (self.due[ids]-now[ids]).float().clamp_min(dt), state)
