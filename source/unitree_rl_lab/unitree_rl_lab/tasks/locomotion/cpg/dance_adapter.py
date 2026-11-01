"""Later-stage discrete dance adapter. Requires separate physical qualification."""
from dataclasses import field
import torch
from isaaclab.utils import configclass
from .adapter import CPGCommand,CPGCommandCfg
from .core import GaitConfig,ik,fk,gravity_position_offset,foot_velocity
from .dance import DanceReference
from unitree_rl_lab.tasks.locomotion.skills.preparation import support_forces

class CPGDanceCommand(CPGCommand):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.dance=DanceReference(self.num_envs,self.device,cfg.gait)
        self.kick_success=torch.zeros_like(self.ready)
        self.kick_failure=torch.zeros_like(self.ready)
        self.kick_active=torch.zeros_like(self.ready)
        self.kick_ever_air=torch.zeros_like(self.ready)
        self.kick_hold_age=torch.zeros_like(self.time)
        self.metrics.update({k:torch.zeros_like(self.time) for k in ('kicks','kick_successes','kick_failures','postkick_successes')})
        self.postkick=torch.zeros_like(self.ready)

    def reset(self,env_ids=None):
        ids=torch.arange(self.num_envs,device=self.device) if env_ids is None else env_ids
        logs=super().reset(ids)
        if hasattr(self,'dance'):
            self.dance.reset(ids);self.time[ids]=self.dance.time[ids]
            self.target_time[ids]=self.dance.target_time[ids]
            if not self.external_velocity:
                self.velocity[ids[self.dance.extended_hold[ids]]]=0.
                self.requested_velocity[ids]=self.velocity[ids]
            for x in (self.kick_active,self.kick_ever_air,self.kick_success,self.kick_failure,self.postkick,self.kick_hold_age):x[ids]=0
        return logs

    @property
    def travel_velocity(self):
        pause=(self.dance.kind==self.dance.KICK)|(self.dance.kind==self.dance.HOLD)|(self.dance.kind==self.dance.RECOVERY)
        return self.velocity*(~pause[:,None])

    @property
    def command(self):
        features=super().command
        if hasattr(self,'dance'):
            front=torch.where(self.dance.selected.any(-1),self.dance.selected[:,:2].long().argmax(-1),self.dance.initial_front)
            gait_time=(front+1)*self.cfg.gait.beat+self.time-self.dance.target_time
            if self.cfg.gait.freeze_hold_phase:
                gait_time=torch.where(self.dance.kind==self.dance.HOLD,(front+1)*self.cfg.gait.beat,gait_time)
            phase=2*torch.pi*gait_time/self.cfg.gait.cycle
            features[:,12:14]=torch.stack((phase.sin(),phase.cos()),-1)
            # Replace the four reserved kick features without changing policy shape.
            features[:,-5:-1]=torch.stack(((self.dance.kind==2).float(),(self.dance.kind==3).float(),
                                           (self.dance.raised==0).float(),(self.dance.raised==1).float()),-1)
        return features

    def sync(self):
        previous_tick=self.last_tick
        super().sync()
        if previous_tick==self.last_tick or not hasattr(self,'dance'):return
        g=self.cfg.gait;r=self.robot.data;dt=self._env.step_dt
        kick=self.dance.kind==self.dance.KICK
        hold=self.dance.kind==self.dance.HOLD
        selected=self.dance.selected
        sole=self.last_records['sole']
        air=((self.events.air>=.04)&selected).any(-1)
        self.kick_ever_air|=kick&air
        body_feet=fk(r.joint_pos[:,self.joint_ids].reshape(-1,4,3))
        # Joint encoders supply lateral endpoint; ground-relative clearance and
        # contact/speed are privileged evaluation measurements, not actor inputs.
        lateral_error=((body_feet[:,:,1]-self.dance.endpoint_body[:,:,1]).abs()*selected).sum(-1)
        planar_error=((body_feet[:,:,:2]-self.dance.endpoint_body[:,:,:2]).norm(dim=-1)*selected).sum(-1)
        height_error=((sole-g.kick_height).abs()*selected).sum(-1)
        speed=(r.body_lin_vel_w[:,self.foot_ids].norm(dim=-1)*selected).sum(-1)
        support=(self.events.confirmed|selected).all(-1)
        stable=(r.projected_gravity_b[:,:2].norm(dim=-1)<.2)&(r.root_ang_vel_b.norm(dim=-1)<.4)
        endpoint=(planar_error<.02)&(height_error<.015)&(speed<.10)&support&stable&air
        self.kick_success=kick&(self.time>=self.dance.target_time)&endpoint
        self.kick_hold_age=torch.where(hold,self.kick_hold_age+dt,0.)
        # Early touchdown after verified kick lift is failure. Hold must retain air.
        grounded=(self.events.confirmed&selected).any(-1)
        unsupported=((self.events.air>=.04)&~selected).any(-1)
        self.kick_failure=(kick&((self.time>self.dance.target_time+.2)|(self.kick_ever_air&grounded)))|(hold&(grounded|unsupported))
        self.metrics['kick_successes']+=self.kick_success
        self.metrics['kick_failures']+=self.kick_failure
        self.metrics['postkick_successes']+=self.postkick&self.last_records['correct']
        self.reward_event+=self.kick_success.float()*.5-self.kick_failure.float()*.5
        self.last_records['kick_success']=self.kick_success.clone()
        self.last_records['kick_failure']=self.kick_failure.clone()
        self.last_records['kick_leg']=selected[:,:2].long().argmax(-1)
        self.last_records['kick_active']=kick.clone()
        self.last_records['hold_active']=hold.clone()
        self.last_records['hold_valid']=hold&endpoint
        self.last_records['kick_lateral_error']=lateral_error.clone()
        self.last_records['kick_planar_error']=planar_error.clone()
        self.last_records['kick_height_error']=height_error.clone()
        self.last_records['kick_speed']=speed.clone()
        self.last_records['kick_endpoint_delay']=self.time-self.dance.target_time
        self.last_records['postkick_success']=self.postkick&self.last_records['correct']

    def _update_command(self):
        self.sync();g=self.cfg.gait
        self._advance_velocity()
        step_success=self.last_records.get('correct',torch.zeros_like(self.ready))
        step_failure=self.last_records.get('finish',torch.zeros_like(self.ready))&~step_success
        previous_missed=self.dance.missed.clone()
        feet,clear=self.dance.update(self._env.step_dt,self.ready,self.events.confirmed,
                                    step_success|self.kick_success,step_failure|self.kick_failure,self.velocity)
        self.time=self.dance.time.clone();self.pair=self.dance.selected.clone()
        self.target_time=self.dance.target_time.clone()
        dispatched=self.dance.dispatched
        step=dispatched&(self.dance.kind==self.dance.STEP)
        kick=dispatched&(self.dance.kind==self.dance.KICK)
        self.active|=step
        self.postkick[step]=self.dance.raised[step]>=0
        self.lifted[step]=False;self.landed[step]=False;self.swing_latch[step]=False
        self.lift_counts[step]=0;self.timed_landing[step]=True;self.support_lost[step]=False
        # An airborne front leg enters the STEP as its first verified swing.
        raised=torch.arange(4,device=self.device)[None,:]==self.dance.raised[:,None]
        entry=step[:,None]&raised
        self.lifted|=entry;self.swing_latch|=entry;self.lift_counts+=entry
        self.kick_ever_air[kick]=False
        self.metrics['attempts']+=step;self.metrics['kicks']+=kick
        self.pending_skip=self.dance.missed>previous_missed;self.metrics['skipped']+=self.pending_skip
        self.recovery=self.dance.kind==self.dance.RECOVERY
        self.feet_ref=feet;self.clearance=clear;self.expected=clear<.008
        q=ik(feet).flatten(1);limits=self.robot.data.soft_joint_pos_limits[:,self.joint_ids]
        clipped=q.clamp(limits[:,:,0],limits[:,:,1])
        self.metrics['limit_clamps']+=(q!=clipped).any(-1)
        self.ik_error=(fk(clipped.reshape(-1,4,3))-feet).norm(dim=-1).amax(-1)
        if g.load_compensation:
            weights=1-(clear/.008).clamp(0,1)
            load=self.mass[:,None]*9.81*weights/weights.sum(-1,keepdim=True).clamp_min(1)
            force=-self.robot.data.projected_gravity_b[:,None,:]*load[:,:,None]
            if g.wrench_compensation:
                gravity=self.robot.data.projected_gravity_b
                roll=torch.atan2(-gravity[:,1],-gravity[:,2])
                pitch=torch.asin(gravity[:,0].clamp(-1,1))
                torque=-g.support_angular_damping*self.robot.data.root_ang_vel_b.clone()
                torque[:,:2]-=20*torch.stack((roll,pitch),-1)
                com=torch.zeros(self.num_envs,3,device=self.device)
                com[:,0]=g.support_com_x;com[:,2]=g.support_com_z
                desired_force=-gravity*self.mass[:,None]*9.81
                if g.support_velocity_damping:
                    q_measured=self.robot.data.joint_pos[:,self.joint_ids].reshape(-1,4,3)
                    qd_measured=self.robot.data.joint_vel[:,self.joint_ids].reshape(-1,4,3)
                    rotational=torch.cross(self.robot.data.root_ang_vel_b[:,None,:].expand(-1,4,-1),fk(q_measured),dim=-1)
                    support=(self.events.confirmed&~self.dance.selected).float()
                    estimated=-((foot_velocity(q_measured,qd_measured)+rotational)*support[:,:,None]).sum(1)/support.sum(-1,keepdim=True).clamp_min(1)
                    tangent=estimated-gravity*(estimated*gravity).sum(-1,keepdim=True)
                    desired_force-=g.support_velocity_damping*self.mass[:,None]*tangent
                allocated=support_forces(feet,com,desired_force,torque,weights)
                three=(self.dance.kind==self.dance.KICK)|(self.dance.kind==self.dance.HOLD)
                force=torch.where(three[:,None,None],force+g.wrench_compensation*(allocated-force),force)
            stiffness=self.robot.data.default_joint_stiffness[:,self.joint_ids].reshape(-1,4,3)
            offset=gravity_position_offset(clipped.reshape(-1,4,3),force,stiffness).clamp(-.4,.4)
            clipped=(clipped+g.load_compensation*offset.flatten(1)).clamp(limits[:,:,0],limits[:,:,1])
        self.q_ref[:,self.joint_ids]=clipped

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
    if component=='kick_tracking':
        feet=fk(c.robot.data.joint_pos[:,c.joint_ids].reshape(-1,4,3))
        error=((feet[:,:,:2]-c.feet_ref[:,:,:2]).square().sum(-1)*selected).sum(-1)
        return torch.exp(-error/.03**2)*kick*c.ready
    if component=='endpoint_speed':
        speed=(c.robot.data.body_lin_vel_w[:,c.foot_ids].square().sum(-1)*selected).sum(-1)
        return speed*((c.dance.kind==c.dance.HOLD)|(kick&(c.dance.phase>=1)))
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
