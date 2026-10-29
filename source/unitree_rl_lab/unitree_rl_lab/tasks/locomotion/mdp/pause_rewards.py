"""Scheduled support cost and per-foot repetition accounting.

Costs see every sampled contact departure. Only a real opposite-pair swing
clears repetition history; noise, one foot or flight cannot clear it.
"""
import torch


def pause_weight(time):
    progress=(time-.32).remainder(.5)
    return (progress/.02).clamp(0,1)*((.30-progress)/.02).clamp(0,1)


class StepHistory:
    def __init__(self,n,device):
        self.previous_step=torch.full((n,),-2,dtype=torch.long,device=device)
        self.previous_contact=torch.zeros((n,4),dtype=torch.bool,device=device)
        self.used=torch.zeros_like(self.previous_contact)
        self.last=torch.full((n,),-1,dtype=torch.long,device=device)
        self.ground=torch.zeros((n,2),dtype=torch.long,device=device)
        self.clear=torch.zeros_like(self.ground)
        self.armed=torch.zeros((n,2),dtype=torch.bool,device=device)
        self.eligible=torch.zeros_like(self.armed)

    def update(self,contact,height,step):
        c=contact.bool()
        reset=step!=self.previous_step+1
        self.used[reset]=False;self.last[reset]=-1
        self.ground[reset]=0;self.clear[reset]=0
        self.armed[reset]=False;self.eligible[reset]=False
        departure=self.previous_contact & ~c & ~reset[:,None]
        repeats=(departure & self.used).sum(-1)
        self.used |= departure
        cp=torch.stack((c[:,[0,3]],c[:,[1,2]]),1)
        hp=torch.stack((height[:,[0,3]],height[:,[1,2]]),1)
        ground=cp.all(-1)
        qualified=(~cp).all(-1)&(hp>=.005).all(-1)&ground.flip(-1)
        self.ground=torch.where(ground,self.ground+1,0)
        self.clear=torch.where(qualified,self.clear+1,0)
        landed=self.ground>=2
        self.armed |= landed
        self.eligible &= ~landed
        confirmed=self.armed & (self.clear>=2)
        self.armed &= ~confirmed
        # A simultaneous flight cannot satisfy opposite-pair ground support.
        for p,other in ((0,[1,2]),(1,[0,3])):
            event=confirmed[:,p]
            self.eligible[:,p]=torch.where(event,(self.last>=0)&(self.last!=p),self.eligible[:,p])
            self.last=torch.where(event,p,self.last)
            for foot in other:
                self.used[:,foot] &= ~event
        self.previous_contact=c.clone();self.previous_step=step.clone()
        return self.eligible & (~cp).all(-1), repeats


def state(env):
    term=env.command_manager.get_term('base_velocity')
    if not hasattr(term,'_pause_history'):
        if abs(env.step_dt-.02)>1e-8:
            raise ValueError('Pause reward timing requires 50 Hz control')
        term._pause_history=StepHistory(env.num_envs,env.device)
        term._pause_tick=None
        times=torch.arange(200,device=env.device,dtype=torch.float64)*env.step_dt
        term._pause_norm=pause_weight(times).mean().float()
    if term._pause_tick!=env.common_step_counter:
        _,height,contact=term.foot_state()
        gate,repeats=term._pause_history.update(contact,height,env.episode_length_buf)
        term._pause_values=(gate,repeats,contact)
        term._pause_tick=env.common_step_counter
    return term,term._pause_values


def paired_step(env):
    from .cadence_pair_scores import paired_step as existing
    term,(gate,_,_)=state(env)
    ref=term.reference
    pair=torch.where(ref.foot_height[:,0]>=ref.foot_height[:,1],0,1)
    return existing(env)*gate.gather(1,pair[:,None]).squeeze(1)


def missing_pause_contacts(env):
    term,(_,_,contact)=state(env)
    time=env.episode_length_buf*env.step_dt+getattr(term,'phase_offset',0.)
    return pause_weight(time)/term._pause_norm*(~contact.bool()).float().mean(-1)


def repeated_departures(env):
    _,(_,repeats,_)=state(env)
    # Manager multiplies by dt: weight -.025 becomes -.025 per repeated foot,
    # i.e. -.05 for a repeated diagonal, even when its takeoffs are staggered.
    return repeats.float()/env.step_dt
