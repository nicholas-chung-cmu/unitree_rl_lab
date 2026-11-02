"""Measured-contact scheduler and bounded position residuals for Isaac Lab."""
from dataclasses import field
import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.utils import configclass
from .core import GaitConfig
from .controller import CPGController


class CPGCommand(CPGController, CommandTerm):
    """Isaac Lab lifecycle wrapper around the shared controller."""
    def __init__(self,cfg,env):
        CommandTerm.__init__(self,cfg,env)
        CPGController.__init__(self,cfg,env)


@configclass
class CPGCommandCfg(CommandTermCfg):
    class_type:type=CPGCommand
    resampling_time_range:tuple=(1e9,1e9)
    gait:GaitConfig=field(default_factory=GaitConfig)
    sensor_uncertainty:dict=field(default_factory=dict)


class ResidualPositionAction(JointPositionAction):
    def process_actions(self,actions):
        self._raw_actions[:]=actions
        c=self._env.command_manager.get_term('cpg')
        residual=torch.tanh(actions)*c.cfg.gait.residual_scale
        residual*=c.ready[:,None]
        limits=self._asset.data.soft_joint_pos_limits
        self._processed_actions=(c.q_ref+residual).clamp(limits[:,:,0],limits[:,:,1])


@configclass
class ResidualPositionActionCfg(JointPositionActionCfg):
    class_type:type=ResidualPositionAction


def cpg_failure(env):
    c=env.command_manager.get_term('cpg');c.sync()
    return (~c.ready)&(c.age>5.)


def reward(env,component):
    c=env.command_manager.get_term('cpg');c.sync();r=c.robot.data
    if component=='contact':out=(c.events.contact==c.expected).float().mean(-1)
    elif component=='clearance':
        sole=r.body_pos_w[:,c.foot_ids,2]-.022-env.scene.env_origins[:,2,None]
        out=(torch.exp(-((sole-c.clearance)/.025).square())-torch.exp(-(c.clearance/.025).square())).mean(-1)
    elif component=='velocity':out=torch.exp(-((r.root_lin_vel_b[:,:2]-getattr(c,'travel_velocity',c.velocity)[:,:2])/c.cfg.gait.velocity_reward_std).square().sum(-1))
    elif component=='yaw':out=torch.exp(-((r.root_ang_vel_b[:,2]-getattr(c,'travel_velocity',c.velocity)[:,2])/c.cfg.gait.yaw_reward_std).square())
    elif component=='height':
        target=.30+c.cfg.gait.bounce_amplitude*torch.sin(2*torch.pi*c.time/c.cfg.gait.beat)
        out=((r.root_pos_w[:,2]-env.scene.env_origins[:,2]-target)/.05).square().clamp_max(4)
    elif component=='residual':out=torch.tanh(env.action_manager.action).square().mean(-1)
    elif component=='timing':out=c.reward_event/env.step_dt
    else:raise ValueError(component)
    mask=c.travel_active if component in ('velocity','yaw') and c.cfg.gait.exclude_pause_tracking else c.ready
    return out*mask


def measured_policy_observation(env, extra_feedback=False):
    """One versioned actor vector from the exact same measurements as control."""
    c=env.command_manager.get_term('cpg');c.sync();r=c.control_data
    terms=[(r.root_ang_vel_b*.2).clamp(-100,100),r.projected_gravity_b.clamp(-100,100),
           (r.joint_pos-c.robot.data.default_joint_pos).clamp(-100,100),
           (r.joint_vel*.05).clamp(-100,100),env.action_manager.action.clamp(-100,100),c.command]
    if extra_feedback:terms.append(c.measured_features)
    return torch.cat(terms,-1)


def randomize_passive_actuator(env, env_ids, static_range=(.15,.25), dynamic_range=(.075,.125)):
    actuator=env.scene['robot'].actuators['GO2HV']
    ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
    for field,bounds in [('_friction_static',static_range),('_friction_dynamic',dynamic_range)]:
        value=getattr(actuator,field)
        value[ids]=torch.rand_like(value[ids])*(bounds[1]-bounds[0])+bounds[0]
