"""Later-stage discrete dance adapter. Requires separate physical qualification."""
from dataclasses import field
import torch
from isaaclab.utils import configclass
from .adapter import CPGCommand,CPGCommandCfg
from .core import GaitConfig,fk
from .controller import CPGDanceController

class CPGDanceCommand(CPGDanceController, CPGCommand):
    """Isaac Lab lifecycle wrapper; all dance behavior is shared with deployment."""
    pass

@configclass
class CPGDanceCommandCfg(CPGCommandCfg):
    class_type:type=CPGDanceCommand
    gait:GaitConfig=field(default_factory=lambda:GaitConfig(
        mode='dance',swing_seconds=.2,clearance=.06,load_compensation=1.,
        training_zero_fraction=1.,balanced_phrases=True,wrench_compensation=1.,
        support_velocity_damping=6.,support_com_x=-.001,support_com_z=-.02,
        kick_shift_back=.03,kick_shift_side=.015,freeze_hold_phase=True))


def dance_reward(env,component):
    c=env.command_manager.get_term('cpg');c.sync()
    kick=(c.dance.kind==c.dance.KICK)|(c.dance.kind==c.dance.HOLD)
    selected=c.dance.selected
    if component=='support_margin':
        # Dense *penalty* on weak required supports, never a reason to keep the
        # selected swing foot down. Truth forces are reward-only, not actor input.
        stance=~selected
        force=c.sensor.data.net_forces_w[:,c.sensor_ids,2].clamp_min(0)
        deficit=((10.-force)/10.).clamp(0,1).square()
        mask=kick|((c.dance.kind==c.dance.STEP)&(c.dance.raised>=0))
        return (deficit*stance).sum(-1)/stance.sum(-1).clamp_min(1)*mask*c.ready
    if component=='kick_tracking':
        feet=fk(c.robot.data.joint_pos[:,c.joint_ids].reshape(-1,4,3))
        error=((feet[:,:,:2]-c.feet_ref[:,:,:2]).square().sum(-1)*selected).sum(-1)
        return torch.exp(-error/c.cfg.gait.kick_tracking_std**2)*kick*c.ready
    if component=='endpoint_speed':
        speed=(c.robot.data.body_lin_vel_w[:,c.foot_ids].square().sum(-1)*selected).sum(-1)
        return speed*((c.dance.kind==c.dance.HOLD)|(kick&(c.dance.phase>=1)))
    if component=='hold_stationary':
        r=c.robot.data
        error=r.root_lin_vel_b[:,:2].square().sum(-1)/.05**2+r.root_ang_vel_b[:,2].square()/.2**2
        return torch.exp(-error)*(c.dance.kind==c.dance.HOLD)*c.ready
    if component in ('kick_stationary','neutral_stationary'):
        r=c.robot.data
        if component=='kick_stationary':
            mask=(c.dance.kind==c.dance.KICK)&c.ready
            linear_std=.06
        else:
            # No translation/yaw target is changed. Only genuinely neutral
            # travel intervals receive this supplemental precision reward.
            mask=c.travel_active&(c.velocity.norm(dim=-1)<.005)
            linear_std=.025
        error=r.root_lin_vel_b[:,:2].square().sum(-1)/linear_std**2+r.root_ang_vel_b[:,2].square()/.2**2
        return torch.exp(-error)*mask
    if component=='hold_quality':
        feet=fk(c.robot.data.joint_pos[:,c.joint_ids].reshape(-1,4,3))
        planar=((feet[:,:,:2]-c.dance.endpoint_body[:,:,:2]).square().sum(-1)*selected).sum(-1)
        sole=c.robot.data.body_pos_w[:,c.foot_ids,2]-.022-env.scene.env_origins[:,2,None]
        height=((sole-c.cfg.gait.kick_height).square()*selected).sum(-1)
        speed=(c.robot.data.body_lin_vel_w[:,c.foot_ids].square().sum(-1)*selected).sum(-1)
        error=planar/.02**2+height/.015**2+speed/.08**2
        if c.cfg.gait.hold_angular_std:
            error+=c.robot.data.root_ang_vel_b.square().sum(-1)/c.cfg.gait.hold_angular_std**2
        return torch.exp(-.5*error)*(c.dance.kind==c.dance.HOLD)*c.ready
    raise ValueError(component)
