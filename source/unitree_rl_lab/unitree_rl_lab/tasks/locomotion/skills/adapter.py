"""Isaac Lab adapter for the simulator-independent hierarchical skill core."""
from dataclasses import field
import torch
import torch.nn.functional as F

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_apply_inverse

from .preparation import PreparationConfig, support_forces
from .shaping import clearance_progress, swing_placement_gate, body_height_deviation
from .core import DancePlanner, Skill, SkillConfig, SkillExecutor, State, Status, rotate_xy, smooth

FOOT_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")


class SkillCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        self.sensor = env.scene["contact_forces"]
        self.foot_ids, names = self.robot.find_bodies(list(FOOT_NAMES), preserve_order=True)
        self.sensor_ids, sensor_names = self.sensor.find_bodies(list(FOOT_NAMES), preserve_order=True)
        if tuple(names) != FOOT_NAMES or tuple(sensor_names) != FOOT_NAMES:
            raise ValueError("Hierarchical skills require FL, FR, RL, RR ordering")
        self.core = SkillExecutor(self.num_envs, self.device, cfg.skill)
        self.planner = DancePlanner(self.num_envs, self.device, cfg.skill)
        self.last_tick = -1
        self.sequence = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.preparing = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.prepare_elapsed = torch.zeros(self.num_envs, device=self.device)
        self.prepare_leg = torch.zeros_like(self.sequence)
        self.prepare_height = torch.zeros(self.num_envs, device=self.device)
        self.prepare_lateral = torch.zeros(self.num_envs, device=self.device)
        self.prepare_start = torch.zeros(self.num_envs, 4, 3, device=self.device)
        self.prepare_body = torch.zeros(self.num_envs, 3, device=self.device)
        self.prepare_yaw = torch.zeros(self.num_envs, device=self.device)
        self.prepare_joint_target = self.robot.data.default_joint_pos.clone()
        self.prepare_confirm = torch.zeros(self.num_envs, device=self.device)
        self.prepare_diagnostics = {}
        self.prepare_lift_time = torch.full_like(self.prepare_elapsed, -1.)
        self.prepare_unload_confirm = torch.zeros_like(self.prepare_elapsed)
        self.preparation_failed = torch.zeros_like(self.preparing)
        self.last_events = {}
        self.external_velocity = False
        self.metrics = {name: torch.zeros(self.num_envs, device=self.device) for name in
                        ("successes", "attempts", "timeouts", "failures", "raised_step_successes", "kick_step_successes")}
        self.from_kick = torch.zeros_like(self.preparing)
        self.pending_from_kick = torch.zeros_like(self.preparing)
        self.started = torch.zeros_like(self.preparing)
        self.settle_elapsed = torch.zeros(self.num_envs, device=self.device)
        self.settle_confirm = torch.zeros_like(self.settle_elapsed)

    def state(self):
        robot = self.robot.data
        feet = robot.body_pos_w[:, self.foot_ids].clone()
        feet[..., 2] -= .022
        _, _, yaw = euler_xyz_from_quat(robot.root_quat_w)
        return State(robot.root_pos_w, yaw, feet,
                     self.sensor.data.net_forces_w[:, self.sensor_ids].norm(dim=-1),
                     robot.body_lin_vel_w[:, self.foot_ids],
                     torch.acos((-robot.projected_gravity_b[:, 2]).clamp(-1, 1)),
                     robot.root_ang_vel_b[:, :2].norm(dim=-1), self._env.scene.env_origins[:, 2])

    @property
    def command(self):
        return torch.cat((self.core.features(self.state()), self.body_reference(), (self.preparing | ~self.started)[:, None].float()), -1)

    def body_reference(self):
        phase = (2*torch.pi*self.planner.time(self._env.step_dt)/self.cfg.skill.beat).float()
        amplitude = self.cfg.skill.bounce_amplitude
        return torch.stack((.32+amplitude*phase.sin(), amplitude*2*torch.pi/self.cfg.skill.beat*phase.cos()), -1)

    def set_velocity(self, velocity):
        """Operator commands affect future goals only; never restart active skills."""
        self.external_velocity = True
        self.planner.set_velocity(velocity)

    def reset(self, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device)
        if env_ids is not None:
            ids = ids[env_ids] if isinstance(env_ids, slice) else torch.as_tensor(env_ids, device=self.device)
        logs = {name: float(value[ids].mean()) for name, value in self.metrics.items()}
        for value in self.metrics.values():
            value[ids] = 0
        self.core.reset(ids, self.state())
        self.planner.reset(ids)
        self.sequence[ids] = 0
        self.started[ids] = False
        self.settle_elapsed[ids] = 0
        self.settle_confirm[ids] = 0
        self.from_kick[ids] = False
        self.pending_from_kick[ids] = False
        self.preparing[ids] = False
        self.preparation_failed[ids] = False
        self.prepare_elapsed[ids] = 0
        self.prepare_confirm[ids] = 0
        self.prepare_lift_time[ids] = -1
        self.prepare_unload_confirm[ids] = 0
        # Random diagonal / front leg, both normal and prepared entries in Stage B.
        self.prepare_leg[ids] = torch.randint(0, 2, (len(ids),), device=self.device)
        self.prepare_height[ids] = torch.empty(len(ids), device=self.device).uniform_(.04, .06)
        self.prepare_lateral[ids] = torch.empty(len(ids), device=self.device).uniform_(.06, .09)
        if not self.external_velocity:
            velocity = self.planner.command.clone()
            velocity[ids] = torch.empty(len(ids), 3, device=self.device).uniform_(-1, 1)
            velocity[ids, :2] *= self.cfg.travel_range
            velocity[ids, 2] *= self.cfg.yaw_range
            # Explicit marching and pure turning cases.
            mode = torch.rand(len(ids), device=self.device)
            velocity[ids[mode < .2]] = 0
            velocity[ids[(mode >= .2) & (mode < .35)], :2] = 0
            self.planner.set_velocity(velocity)
        return logs

    def _resample_command(self, env_ids):
        pass  # Skill dispatch, never a periodic command resample.

    def _update_metrics(self):
        pass

    def sync(self):
        """Called by termination/reward terms BEFORE auto-reset; one update per step."""
        tick = getattr(self._env, "common_step_counter", 0)
        if tick == self.last_tick:
            return
        self.last_tick = tick
        state = self.state()
        self.core.update(state, self._env.step_dt)
        waiting = ~self.started
        self.settle_elapsed += waiting*self._env.step_dt
        stable = (self.core.confirmed.all(-1) & (state.tilt < .15)
                  & (self.robot.data.root_lin_vel_w.norm(dim=-1) < .05)
                  & (self.robot.data.root_ang_vel_w.norm(dim=-1) < .2)
                  & (state.foot_velocity.norm(dim=-1).amax(-1) < .05))
        self.settle_confirm = torch.where(waiting, torch.where(
            stable, self.settle_confirm+self._env.step_dt, 0.), self.settle_confirm)
        c = self.core
        event = (c.success_event.bool() | c.failure_event.bool()) & ~self.preparing
        meaningful = event & (c.kind != Skill.STABILIZE)
        self.metrics["attempts"] += meaningful
        self.metrics["successes"] += meaningful & (c.status == Status.SUCCESS)
        self.metrics["timeouts"] += meaningful & (c.status == Status.TIMEOUT)
        self.metrics["failures"] += meaningful & (c.status == Status.FAILURE)
        raised_step = meaningful & (c.kind == Skill.STEP) & c.entry_raised.any(-1)
        self.metrics["raised_step_successes"] += raised_step & (c.status == Status.SUCCESS)
        self.metrics["kick_step_successes"] += raised_step & self.from_kick & (c.status == Status.SUCCESS) & (c.premature == 0)
        self.last_events = {name: value.clone() for name, value in c.metrics.items()}
        self.last_events.update(event=meaningful.clone(), kind=c.kind.clone(), status=c.status.clone(),
                                entry_raised=c.entry_raised.clone(), from_kick=self.from_kick.clone(),
                                dance_timing_error=self.planner.dispatch_error.clone(),
                                preparation_failed=self.preparation_failed.clone(),
                                base_quaternion=self.robot.data.root_quat_w.clone(),
                                base_linear_velocity=self.robot.data.root_lin_vel_w.clone())

    def _step(self, ids, raised=False):
        c, state = self.core, self.state()
        if not len(ids):
            return
        if raised:
            leg = (~c.confirmed[ids, :2]).long().argmax(-1)
        else:
            leg = (self.prepare_leg[ids]+self.sequence[ids]).remainder(2)
        selected = F.one_hot(leg, 4).bool() | F.one_hot(3-leg, 4).bool()
        duration = torch.empty(len(ids), device=self.device).uniform_(*self.cfg.step_duration_range)
        goal = self.planner.command[ids]*duration[:, None]
        c.begin(ids, Skill.STEP, selected, goal, duration, state)
        self.from_kick[ids] = self.pending_from_kick[ids]
        self.pending_from_kick[ids] = False
        self.sequence[ids] += 1

    def _update_command(self):
        c, state = self.core, self.state()
        four = c.confirmed.all(-1)
        done = c.status == Status.SUCCESS
        # Wait for measured four-foot support after a physical reset.
        init = (~self.started & four & (self.settle_elapsed >= 1.)
                & (self.settle_confirm >= .5))
        c.nominal_xy[init] = rotate_xy(state.feet[init, :, :2]-state.body[init, None, :2], -state.yaw[init, None])
        # Reset poses can settle vertically before initial support is acquired.
        c.start_feet[~self.started] = state.feet[~self.started]
        # The measured settling gate replaces the short initial STABILIZE timer.
        c.status[init] = Status.SUCCESS
        self.started |= init
        if self.cfg.stage == "B":
            prepare = init & (torch.rand(self.num_envs, device=self.device) < self.cfg.raised_fraction)
            self.preparing |= prepare
            self.prepare_start[prepare] = state.feet[prepare]
            self.prepare_body[prepare] = state.body[prepare]
            self.prepare_yaw[prepare] = state.yaw[prepare]
            self.prepare_joint_target[prepare] = self.robot.data.default_joint_pos[prepare]
            self.prepare_elapsed[prepare] = 0
            normal = init & ~prepare
            self._step(normal.nonzero().flatten())
        if self.preparing.any():
            self.prepare_elapsed += self.preparing*self._env.step_dt
            legmask = F.one_hot(self.prepare_leg, 4).bool()
            clearance = state.feet[..., 2]-state.ground[:, None]
            supported = (c.confirmed | legmask).all(-1)
            raised = ((~c.contact & (clearance > .03)) | ~legmask).all(-1)
            relative = rotate_xy(state.feet[..., :2]-state.body[:, None, :2], -state.yaw[:, None])
            outward = (relative[..., 1]-c.nominal_xy[..., 1])*relative.new_tensor([1, -1, 1, -1])
            endpoint = (((clearance-self.prepare_height[:, None]).abs() < .015)
                        & ((outward-self.prepare_lateral[:, None]).abs() < .025)) | ~legmask
            cfg = self.cfg.preparation
            rows = torch.arange(self.num_envs, device=self.device)
            unload = supported & (state.force[rows, self.prepare_leg] < cfg.unload_force) & (state.tilt < .2)
            self.prepare_unload_confirm = torch.where(unload & self.preparing, self.prepare_unload_confirm+self._env.step_dt, 0.)
            begin_lift = self.preparing & (self.prepare_lift_time < 0) & (self.prepare_elapsed >= cfg.shift_seconds) & (self.prepare_unload_confirm >= cfg.unload_seconds)
            self.prepare_lift_time[begin_lift] = self.prepare_elapsed[begin_lift]
            foot_speed = state.foot_velocity.norm(dim=-1)
            valid = (supported & raised & endpoint.all(-1) & (state.tilt < .2)
                     & (self.robot.data.root_ang_vel_w.norm(dim=-1) < cfg.max_angular_speed)
                     & (self.robot.data.root_lin_vel_w.norm(dim=-1) < cfg.max_body_speed)
                     & (foot_speed.amax(-1) < cfg.max_foot_speed))
            rows = torch.arange(self.num_envs, device=self.device)
            self.prepare_diagnostics = dict(elapsed=self.prepare_elapsed.clone(), supported=supported, raised=raised,
                height=clearance[rows, self.prepare_leg], target_height=self.prepare_height.clone(),
                outward=outward[rows, self.prepare_leg], target_outward=self.prepare_lateral.clone(),
                tilt=state.tilt, angular_speed=state.angular_speed,
                base_height=state.body[:, 2]-state.ground, base_speed=self.robot.data.root_lin_vel_w.norm(dim=-1),
                force=state.force, valid=valid, confirm=self.prepare_confirm.clone(),
                foot_speed=foot_speed, lift_time=self.prepare_lift_time.clone())
            lift_complete = (self.prepare_lift_time >= 0) & (self.prepare_elapsed-self.prepare_lift_time >= cfg.lift_seconds)
            self.prepare_confirm = torch.where(valid & self.preparing & lift_complete, self.prepare_confirm+self._env.step_dt, 0)
            ready = (self.preparing & (self.prepare_lift_time >= 0)
                     & (self.prepare_elapsed-self.prepare_lift_time >= cfg.lift_seconds)
                     & (self.prepare_confirm >= cfg.hold_seconds))
            ids = ready.nonzero().flatten()
            self.preparing[ids] = False
            self._step(ids, raised=True)
            self.preparation_failed |= self.preparing & (self.prepare_elapsed > cfg.timeout_seconds)
        if self.cfg.stage in ("E", "F"):
            # Starting support acquisition is outside the musical clock.
            self.planner.reset(init.nonzero().flatten())
            previous_kind = c.kind.clone()
            self.planner.advance(c, state, self._env.step_dt)
            new_step = (c.kind == Skill.STEP) & (c.elapsed == 0)
            self.from_kick[new_step] = previous_kind[new_step] == Skill.KICK
            return
        active = self.started & ~self.preparing
        done = (c.status == Status.SUCCESS) & active
        # Stage C keeps a successful kick raised through a brief STABILIZE; the
        # episode ends after that hold. D directly chains the same policy.
        kick_done = done & (c.kind == Skill.KICK)
        if self.cfg.stage == "C":
            ids = kick_done.nonzero().flatten()
            c.begin(ids, Skill.STABILIZE, c.confirmed[ids], torch.zeros(len(ids), 3, device=self.device), .15, state)
            self.sequence[ids] = 1
        else:
            ids = kick_done.nonzero().flatten()
            self.pending_from_kick[ids] = True
            self._step(ids, raised=True)
        done = (c.status == Status.SUCCESS) & active
        if self.cfg.stage == "C":
            kick = done & (c.kind == Skill.STABILIZE) & (self.sequence == 0) & four
        elif self.cfg.stage == "D":
            kick = done & (c.kind == Skill.STEP) & (self.sequence >= self.cfg.steps_before_kick) & four
        else:
            kick = torch.zeros_like(done)
        ids = kick.nonzero().flatten()
        if len(ids):
            leg = self.prepare_leg[ids]
            c.begin(ids, Skill.KICK, F.one_hot(leg, 4).bool(), torch.zeros(len(ids), 3, device=self.device),
                    self.cfg.skill.kick_duration, state)
            self.prepare_leg[ids] = 1-leg
            if self.cfg.stage == "D":
                self.sequence[ids] = 0
            self.from_kick[ids] = False
        if self.cfg.stage in ("A", "B", "D"):
            done = (c.status == Status.SUCCESS) & active & four
            self._step(done.nonzero().flatten())

    def preparation_actions(self, actions):
        """Support-wrench control, unloading, then a slow airborne endpoint hold."""
        if not self.preparing.any():
            return actions
        ids = self.preparing.nonzero().flatten()
        state = self.state()
        cfg = self.cfg.preparation
        leg = self.prepare_leg[ids]
        index = torch.arange(len(ids), device=self.device)
        legmask = F.one_hot(leg, 4).bool()
        elapsed = self.prepare_elapsed[ids]
        shift = smooth(elapsed/cfg.shift_seconds)
        lift = torch.where(self.prepare_lift_time[ids] >= 0,
                           smooth((elapsed-self.prepare_lift_time[ids])/cfg.lift_seconds), 0.)
        feet = state.feet[ids]
        mass = self.robot.data.default_mass.to(self.device)[ids]
        total_mass = mass.sum(-1)
        com = (self.robot.data.body_com_pos_w[ids]*mass[..., None]).sum(1)/total_mass[:, None]
        centroid = (self.prepare_start[ids, :, :2]*(~legmask)[..., None]).sum(1)/3
        # Track actual whole-body CoM over the remaining support triangle.
        desired_xy = self.prepare_body[ids, :2]*(1-shift[:, None])+centroid*shift[:, None]
        desired_height = self.prepare_body[ids, 2]*(1-shift)+(state.ground[ids]+cfg.body_height)*shift
        force = torch.zeros(len(ids), 3, device=self.device)
        force[:, :2] = 350*(desired_xy-com[:, :2])-65*self.robot.data.root_lin_vel_w[ids, :2]
        force[:, 2] = total_mass*9.81+500*(desired_height-state.body[ids, 2])-70*self.robot.data.root_lin_vel_w[ids, 2]
        roll, pitch, yaw = euler_xyz_from_quat(self.robot.data.root_quat_w[ids])
        roll = torch.atan2(torch.sin(roll), torch.cos(roll))
        pitch = torch.atan2(torch.sin(pitch), torch.cos(pitch))
        signs = torch.where(leg == 0, 1., -1.)
        torque = torch.stack((25*(signs*cfg.lean_roll*shift-roll),
                              25*(cfg.lean_pitch*shift-pitch),
                              15*torch.atan2(torch.sin(self.prepare_yaw[ids]-yaw), torch.cos(self.prepare_yaw[ids]-yaw))), -1)
        torque[:, :2] = rotate_xy(torque[:, :2], yaw)
        torque -= 3*self.robot.data.root_ang_vel_w[ids]
        weights = torch.ones(len(ids), 4, device=self.device)
        weights[index, leg] = 1-shift
        reactions = support_forces(feet, com, force, torque, weights)
        # Swing endpoint stays body-relative laterally and ground-relative in z.
        nominal = self.core.nominal_xy[ids].clone()
        nominal[index, leg, 1] += signs*self.prepare_lateral[ids]
        endpoint = self.prepare_start[ids].clone()
        endpoint[..., :2] = state.body[ids, None, :2]+rotate_xy(nominal, state.yaw[ids, None])
        endpoint[index, leg, 2] = state.ground[ids]+self.prepare_height[ids]
        desired_feet = self.prepare_start[ids]+lift[:, None, None]*(endpoint-self.prepare_start[ids])
        swing_force = 250*(desired_feet-feet)-12*state.foot_velocity[ids]
        # Fade reaction load into Cartesian foot control before liftoff.
        motor_foot_force = -reactions + legmask[..., None]*shift[:, None, None]*swing_force
        jac = self.robot.root_physx_view.get_jacobians()[ids][:, self.foot_ids, :3, 6:]
        torque_joint = torch.einsum('nfij,nfi->nj', jac, motor_foot_force)
        torque_joint += self.robot.root_physx_view.get_gravity_compensation_forces()[ids, 6:]
        torque_joint -= .3*self.robot.data.joint_vel[ids]
        stiffness = self.robot.data.default_joint_stiffness[ids]
        damping = self.robot.data.default_joint_damping[ids]
        # Convert torque demand into the stock PD actuator's position target.
        q = self.robot.data.joint_pos[ids] + (torque_joint+damping*self.robot.data.joint_vel[ids])/stiffness
        limits = self.robot.data.soft_joint_pos_limits[ids]
        q = q.clamp(limits[..., 0], limits[..., 1])
        self.prepare_joint_target[ids] = q
        result = actions.clone()
        result[ids] = (q-self.robot.data.default_joint_pos[ids])/.25
        return result


@configclass
class SkillCommandCfg(CommandTermCfg):
    class_type: type = SkillCommand
    resampling_time_range: tuple = (1.e9, 1.e9)
    debug_vis: bool = False
    skill: SkillConfig = field(default_factory=SkillConfig)
    preparation: PreparationConfig = field(default_factory=PreparationConfig)
    stage: str = "A"
    step_duration_range: tuple = (.8, 1.2)
    travel_range: float = .08
    yaw_range: float = .12
    raised_fraction: float = .5
    steps_before_kick: int = 1
    isolated_steps: bool = True


class SkillEnv(ManagerBasedRLEnv):
    def step(self, action):
        term = self.command_manager.get_term("skill")
        # Stock default-pose PD holds the robot while the reset drop settles.
        # Do not let an untrained policy disturb the measured admission window.
        action = torch.where((~term.started)[:, None], torch.zeros_like(action), action)
        return super().step(term.preparation_actions(action))


def term(env):
    return env.command_manager.get_term("skill")


def skill_failure(env):
    t = term(env)
    t.sync()
    # Training tasks terminate failed trials. Dance allows four-foot recovery;
    # invalid support cannot dispatch conflicting goals and eventually times out.
    failed = t.core.status >= Status.TIMEOUT
    if t.cfg.stage in ("E", "F"):
        failed &= ~t.core.confirmed.all(-1)
    settling_failed = ~t.started & ((t.settle_elapsed > 5.) | t.core.metrics["fall"])
    return (failed & t.started & ~t.preparing) | t.preparation_failed | settling_failed


def isolated_complete(env):
    t = term(env)
    kick_hold = (t.cfg.stage == "C") & (t.sequence > 0) & (t.core.kind == Skill.STABILIZE)
    isolated_step = (t.cfg.stage in ("A", "B")) & t.cfg.isolated_steps & (t.core.kind == Skill.STEP)
    return (kick_hold | isolated_step) & (t.core.status == Status.SUCCESS)


def physical_fall(env):
    t = term(env)
    t.sync()
    fallen = t.core.metrics["fall"] | env.termination_manager.get_term("base_contact")
    return (fallen & env.termination_manager.terminated).float()/env.step_dt


def skill_reward(env, component):
    t = term(env)
    t.sync()
    c, s = t.core, t.state()
    step, kick = c.kind == Skill.STEP, c.kind == Skill.KICK
    p = (c.elapsed/c.duration).clamp(0, 1)
    reference, _ = c.reference(s)
    if component == "placement":
        error = (s.feet-reference).square().sum(-1)
        mask = c.selected | (c.kind == Skill.STABILIZE)[:, None]
        credit = torch.exp(-error/.0025)
        gate = swing_placement_gate(s.feet[..., 2]-s.ground[:, None],
                                    reference[..., 2]-s.ground[:, None])
        credit *= torch.where(step[:, None] & c.selected, gate, 1.)
        result = credit.mul(mask).sum(-1)/mask.sum(-1).clamp_min(1)
    elif component == "clearance":
        credit = clearance_progress(s.feet[..., 2]-s.ground[:, None],
                                    reference[..., 2]-s.ground[:, None])
        result = (credit*c.selected).sum(-1)/c.selected.sum(-1).clamp_min(1)*step
    elif component == "body_height":
        result = body_height_deviation(s.body[:, 2]-s.ground, t.body_reference()[:, 0])
    elif component == "displacement":
        actual = rotate_xy(s.body[:, :2]-c.start_body[:, :2], -c.start_yaw)
        result = torch.exp(-(actual-c.goal[:, :2]*smooth(p[:, None])).square().sum(-1)/.0064)*step
    elif component == "heading":
        from .core import wrap
        error = wrap(s.yaw-c.start_yaw-c.goal[:, 2]*smooth(p))
        result = torch.exp(-error.square()/.04)
    elif component == "support":
        required = ~c.selected
        required = torch.where((c.kind == Skill.STABILIZE)[:, None], c.selected, required)
        result = (c.confirmed & required).sum(-1)/required.sum(-1).clamp_min(1)
    elif component == "contact":
        # STEP contact targets follow the actual vertical reference, including
        # raised entries. Blend near ground (0--8 mm) instead of demanding
        # touchdown at .70 progress while the reference is still 26 mm high.
        step_air = ((reference[..., 2]-s.ground[:, None])/.008).clamp(0, 1)
        desired_air = torch.where(step[:, None], step_air*c.selected,
                                  (c.selected & (kick & (p > .2))[:, None]).float())
        desired_air = torch.where((c.kind == Skill.STABILIZE)[:, None],
                                  (~c.selected).float(), desired_air)
        result = (1-((~c.contact).float()-desired_air).abs()).mean(-1)
    elif component == "success":
        result = c.success_event*(c.kind != Skill.STABILIZE)/env.step_dt
    elif component == "failure":
        result = c.failure_event/env.step_dt
    elif component == "transition":
        result = c.success_event*step*c.entry_raised.any(-1)*t.from_kick*(c.premature == 0)/env.step_dt
    elif component == "premature":
        result = step*(c.entry_raised & c.selected & c.contact).any(-1)*(p < .6)
    elif component == "kick_touchdown":
        result = kick*(c.selected & c.contact).any(-1)*(p > .2)
    elif component == "endpoint_speed":
        result = kick*(p > .8)*(s.foot_velocity.square().sum(-1)*c.selected).sum(-1)
    elif component == "touchdown_timing":
        valid = c.selected & (c.touchdown_time >= 0)
        error = ((c.touchdown_time-c.duration[:, None]).abs()/.1).clamp(max=1)
        result = step*(error*valid).sum(-1)/c.selected.sum(-1).clamp_min(1)
    elif component == "bounce":
        target, velocity = t.body_reference().unbind(-1)
        result = (s.body[:, 2]-s.ground-target).square()+.05*(t.robot.data.root_lin_vel_w[:, 2]-velocity).square()
    else:
        raise ValueError(component)
    return result * (~t.preparing & t.started)
