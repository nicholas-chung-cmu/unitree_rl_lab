"""Measured-contact scheduler and bounded position residuals for Isaac Lab."""
from dataclasses import field
import math
import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_apply_inverse
from .core import GaitConfig, ContactEvents, nominal_feet, walking_reference, WalkingReference, advance_velocity, schedule, ik, fk, gravity_position_offset, foot_velocity, timing_valid, FOOTS


class CPGCommand(CommandTerm):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.robot=env.scene['robot'];self.sensor=env.scene['contact_forces']
        self.foot_ids,_=self.robot.find_bodies([f'{x}_foot' for x in FOOTS],preserve_order=True)
        self.sensor_ids,_=self.sensor.find_bodies([f'{x}_foot' for x in FOOTS],preserve_order=True)
        self.joint_ids,_=self.robot.find_joints([f'{x}_{j}_joint' for x in FOOTS for j in ('hip','thigh','calf')],preserve_order=True)
        n=self.num_envs;d=self.device
        self.time=torch.zeros(n,device=d);self.age=torch.zeros_like(self.time)
        self.ready=torch.zeros(n,dtype=torch.bool,device=d)
        self.stable_age=torch.zeros_like(self.time)
        self.velocity=torch.zeros(n,3,device=d);self.external_velocity=False
        self.requested_velocity=torch.zeros_like(self.velocity)
        self.events=ContactEvents(n,d)
        self.active=torch.zeros_like(self.ready);self.recovery=torch.zeros_like(self.ready)
        self.pair=torch.zeros(n,4,dtype=torch.bool,device=d)
        self.lifted=torch.zeros_like(self.pair);self.landed=torch.zeros_like(self.pair)
        self.swing_latch=torch.zeros_like(self.pair)
        self.timed_landing=torch.ones_like(self.pair)
        self.lift_counts=torch.zeros(n,4,device=d)
        self.support_lost=torch.zeros_like(self.ready)
        self.target_time=torch.zeros_like(self.time)
        self.last_beat=torch.full((n,),-1,device=d,dtype=torch.long)
        self.last_swing=torch.zeros_like(self.ready)
        self.last_tick=-1
        self.q_ref=self.robot.data.default_joint_pos.clone()
        self.feet_ref=nominal_feet(n,d,cfg.gait)
        self.clearance=torch.zeros(n,4,device=d);self.expected=torch.ones_like(self.pair)
        self.reward_event=torch.zeros_like(self.time)
        self.pending_skip=torch.zeros_like(self.ready)
        self.touch_error=torch.zeros(n,4,device=d)
        self.touch_valid=torch.zeros_like(self.pair)
        self.lift_valid=torch.zeros_like(self.pair)
        self.last_records={}
        self.metrics={k:torch.zeros(n,device=d) for k in ('attempts','correct','missed','skipped','FL_RR_correct','FR_RL_correct','limit_clamps')}
        self.ik_error=torch.zeros_like(self.time)
        self.mass=self.robot.root_physx_view.get_masses().sum(-1).to(self.device)
        self.walking_reference=WalkingReference(n,d,cfg.gait)

    @property
    def command(self):
        # Actor inputs are encoders, IMU, operator/reference state only. No truth pose/velocity/contact.
        phase=2*math.pi*self.time/self.cfg.gait.cycle
        phrase=2*math.pi*self.time/(4*self.cfg.gait.beat)
        kick=torch.zeros(self.num_envs,4,device=self.device)
        return torch.cat((self.q_ref,torch.stack((phase.sin(),phase.cos(),phrase.sin(),phrase.cos()),-1),
                          self.expected.float(),self.velocity,self.pair.float(),
                          (self.target_time-self.time).clamp(-1,1)[:,None],kick,
                          (~self.ready|self.recovery).float()[:,None]),-1)

    def set_velocity(self,v):
        self.external_velocity=True
        self.requested_velocity[:]=torch.as_tensor(v,device=self.device)
        speed=self.requested_velocity[:,:2].norm(dim=-1,keepdim=True)
        self.requested_velocity[:,:2]*=(self.cfg.gait.max_speed/speed.clamp_min(1e-9)).clamp_max(1)
        self.requested_velocity[:,2].clamp_(-self.cfg.gait.max_yaw,self.cfg.gait.max_yaw)
        g=self.cfg.gait
        if not (g.linear_acceleration_limit or g.yaw_acceleration_limit):
            self.velocity[:]=self.requested_velocity
        else:
            self.velocity[~self.ready]=self.requested_velocity[~self.ready]

    def _advance_velocity(self):
        g=self.cfg.gait
        if g.linear_acceleration_limit or g.yaw_acceleration_limit:
            updated=advance_velocity(self.velocity,self.requested_velocity,self._env.step_dt,g)
            self.velocity[:]=torch.where(self.ready[:,None],updated,self.requested_velocity)

    def reset(self,env_ids=None):
        ids=torch.arange(self.num_envs,device=self.device) if env_ids is None else env_ids
        logs={k:float(v[ids].mean()) for k,v in self.metrics.items()}
        for v in self.metrics.values():v[ids]=0
        for v in (self.time,self.age,self.stable_age,self.ready,self.active,self.recovery,self.pair,self.lifted,
                  self.landed,self.swing_latch,self.lift_counts,self.support_lost,self.last_swing,self.reward_event,self.pending_skip):v[ids]=0
        self.timed_landing[ids]=True
        self.last_beat[ids]=-1;self.events.reset(ids)
        self.walking_reference.reset(ids)
        self.q_ref[ids]=self.robot.data.default_joint_pos[ids]
        self.expected[ids]=True;self.clearance[ids]=0
        if not self.external_velocity:
            self.velocity[ids]=torch.rand(len(ids),3,device=self.device)*2-1
            self.velocity[ids,:2]*=self.cfg.gait.max_speed
            self.velocity[ids,2]*=self.cfg.gait.max_yaw
            norm=self.velocity[ids,:2].norm(dim=-1,keepdim=True)
            self.velocity[ids,:2]*=(self.cfg.gait.max_speed/norm.clamp_min(1e-8)).clamp_max(1)
            self.velocity[ids[torch.rand(len(ids),device=self.device)<self.cfg.gait.training_zero_fraction]]=0
            self.requested_velocity[ids]=self.velocity[ids]
        return logs

    def _resample_command(self,env_ids):pass
    def _update_metrics(self):pass

    def sync(self):
        tick=self._env.common_step_counter
        if tick==self.last_tick:return
        self.last_tick=tick
        dt=self._env.step_dt;g=self.cfg.gait;r=self.robot.data
        self.age+=dt
        td,lo,onset=self.events.update(self.sensor.data.net_forces_w[:,self.sensor_ids].norm(dim=-1),self.time,dt)
        self.touch_valid=td & self.active[:,None] & self.pair & self.lifted & ~self.landed
        self.touch_error=onset-self.target_time[:,None]
        sole=r.body_pos_w[:,self.foot_ids,2]-.022-self._env.scene.env_origins[:,2,None]
        verified=(self.events.air>=.04-1e-6)&(sole>.008)
        self.swing_latch &= ~td
        new_lift=verified & ~self.swing_latch & self.pair & self.active[:,None]
        self.swing_latch |= new_lift
        self.lift_valid=new_lift
        self.lift_counts+=new_lift
        self.lifted|=new_lift
        self.landed|=self.touch_valid
        self.support_lost|=((self.events.air>=.04-1e-6)&~self.pair).any(-1)&self.active
        self.reward_event.zero_()
        self.reward_event-=.5*self.pending_skip
        self.reward_event+=((torch.exp(-(self.touch_error/g.timing_tolerance).square())-.5)*self.touch_valid).sum(-1)
        # Close each attempted STEP after the physical contact window, never by reference alone.
        finish=self.active & (self.time>self.target_time+g.timing_tolerance)
        correct=finish & ((self.landed&self.pair).sum(-1)==2) & ~self.support_lost
        # Every selected foot must touch down in the allowed window. Latch errors separately.
        self.timed_landing &= ~(self.touch_valid & ~timing_valid(self.touch_error,g.timing_tolerance))
        correct &= self.timed_landing.all(-1) & ((self.lift_counts*self.pair).sum(-1)==2)
        self.metrics['correct']+=correct;self.metrics['missed']+=finish&~correct
        self.metrics['FL_RR_correct']+=correct&self.pair[:,0]
        self.metrics['FR_RL_correct']+=correct&self.pair[:,1]
        self.reward_event+=correct.float()*.5
        self.reward_event-=.5*(finish&~correct)
        self.active[finish]=False
        self.recovery |= finish & ~self.events.confirmed.all(-1)
        q=r.joint_pos[:,self.joint_ids].reshape(-1,4,3)
        qd=r.joint_vel[:,self.joint_ids].reshape(-1,4,3)
        foot_v=foot_velocity(q,qd)+torch.cross(r.root_ang_vel_b[:,None,:].expand(-1,4,-1),fk(q),dim=-1)
        estimated_body_velocity=-foot_v.mean(1)
        stable=(self.events.confirmed.all(-1)&(r.projected_gravity_b[:,:2].norm(dim=-1)<.15)
                &(estimated_body_velocity.norm(dim=-1)<.08)&(r.root_ang_vel_b.norm(dim=-1)<.3))
        self.stable_age=torch.where(stable,self.stable_age+dt,0.)
        admitted=~self.ready&(self.age>=1.)&(self.stable_age>=.25)
        self.ready|=admitted
        self.recovery&=~(stable&(self.stable_age>=.12))
        self.last_records={'finish':finish.clone(),'correct':correct.clone(),'pair':self.pair.clone(),
                           'touchdown':self.touch_valid.clone(),'touch_error':self.touch_error.clone(),
                           'liftoff':new_lift.clone(),'sole':sole.clone(),'contact':self.events.contact.clone(),
                           'expected':self.expected.clone(),'time':self.time.clone(),'ready':self.ready.clone(),
                           'raw_touchdown':td.clone(),'raw_liftoff':lo.clone(),'target_time':self.target_time.clone()}
        self.last_records['paired_lift']=finish&((self.lifted&self.pair).sum(-1)==2)

    def _update_command(self):
        self.sync()
        self._advance_velocity()
        g=self.cfg.gait;dt=self._env.step_dt
        self.time+=self.ready*dt
        beat,within,pair,swing=schedule(self.time,g)
        start=swing&~self.last_swing&self.ready
        # Scheduler dispatches at the lead time; bad support skips this opportunity without slowing music.
        admit=start&~self.active&~self.recovery&self.events.confirmed.all(-1)
        self.metrics['skipped']+=start&~admit
        self.pending_skip=start&~admit
        self.recovery|=self.pending_skip
        self.active|=admit;self.pair[admit]=pair[admit]
        self.lifted[admit]=False;self.landed[admit]=False;self.swing_latch[admit]=False;self.support_lost[admit]=False
        self.lift_counts[admit]=0;self.timed_landing[admit]=True
        self.target_time[admit]=(beat[admit]+1)*g.beat
        self.metrics['attempts']+=admit
        self.last_swing=swing
        if g.integrated_walk_reference:
            feet,clear,swingfeet=self.walking_reference.update(self.time,self.velocity,self.ready)
        else:
            feet,clear,swingfeet=walking_reference(self.time,self.velocity,g)
        intent_clear=clear.clone()
        enabled=self.ready&~self.recovery
        # Block unadmitted scheduled swings but retain stance references.
        swing_allowed=self.active[:,None]&self.pair
        feet[:,:,2]=torch.where(swingfeet&~swing_allowed,-g.leg_height,feet[:,:,2])
        clear*=swing_allowed
        nominal=nominal_feet(self.num_envs,self.device,g)
        fallback=self.feet_ref+(nominal-self.feet_ref).clamp(-.005,.005)
        feet=torch.where(enabled[:,None,None],feet,fallback)
        clear*=enabled[:,None]
        self.feet_ref=feet
        # Suppressed swings still have musical goals: standing must not earn a
        # perfect contact score merely because admission rejected the swing.
        self.clearance=intent_clear*self.ready[:,None]
        self.expected=~(self.clearance>.008)
        if g.attitude_compensation:
            gravity=self.robot.data.projected_gravity_b
            roll=torch.atan2(-gravity[:,1],-gravity[:,2]).clamp(-.3,.3)*g.attitude_compensation
            pitch=torch.asin(gravity[:,0].clamp(-1,1)).clamp(-.3,.3)*g.attitude_compensation
            tilt=quat_from_euler_xyz(roll,pitch,torch.zeros_like(roll))
            feet=quat_apply_inverse(tilt[:,None,:].expand(-1,4,-1).reshape(-1,4),feet.reshape(-1,3)).reshape(-1,4,3)
        q=ik(feet).flatten(1)
        limits=self.robot.data.soft_joint_pos_limits[:,self.joint_ids]
        clipped=q.clamp(limits[:,:,0],limits[:,:,1])
        self.metrics['limit_clamps']+=(q!=clipped).any(-1)
        self.ik_error=(fk(clipped.reshape(-1,4,3))-feet).norm(dim=-1).amax(-1)
        if g.load_compensation:
            weights=1-(clear/.008).clamp(0,1)
            load=self.mass[:,None]*9.81*weights/weights.sum(-1,keepdim=True).clamp_min(1)
            force=-self.robot.data.projected_gravity_b[:,None,:]*load[:,:,None]
            stiffness=self.robot.data.default_joint_stiffness[:,self.joint_ids].reshape(-1,4,3)
            offset=gravity_position_offset(clipped.reshape(-1,4,3),force,stiffness).clamp(-.4,.4)
            clipped=(clipped+g.load_compensation*offset.flatten(1)).clamp(limits[:,:,0],limits[:,:,1])
        self.q_ref[:,self.joint_ids]=clipped


@configclass
class CPGCommandCfg(CommandTermCfg):
    class_type:type=CPGCommand
    resampling_time_range:tuple=(1e9,1e9)
    gait:GaitConfig=field(default_factory=GaitConfig)


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
    return out*c.ready
