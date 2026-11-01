"""Discrete musical choreography, independent of the walking reference oscillator.

The clock owns opportunities, contact feedback owns completion/admission. This
module never stretches beats or adds catch-up travel after a skipped opportunity.
"""
from dataclasses import dataclass
import math
from .core import GaitConfig

@dataclass(frozen=True)
class Opportunity:
    kind:str
    start:float
    target:float
    pair:tuple[int,...]
    phrase:int
    from_kick:bool=False

class DanceScheduler:
    def __init__(self,cfg:GaitConfig):
        self.cfg=cfg
        self.last_id=None
        self.active=None
        self.recovery=False
        self.missed=0
        self.raised_front=None

    def phrase(self,index):
        """Times relative to first musical touchdown, with negative-time pre-roll."""
        g=self.cfg;base=index*4*g.beat
        result=[]
        for step in range(3):
            front=(index*3+step)%2
            target=base+step*g.beat
            result.append(Opportunity('STEP',target-g.swing_seconds,target,(front,3-front),index,
                                      from_kick=(index>0 and step==0)))
        leg=(index+1)%2
        start=base+2.5*g.beat
        result.append(Opportunity('KICK',start,start+g.kick_duration,(leg,),index))
        return result

    def dispatch(self,time,confirmed_contacts,previous_success=False,failed=False):
        """Single-controller scalar contract, also used by deterministic integration tests.

        The simulator's batched walk adapter uses the equivalent tensor admission
        logic. No KICK is dispatched by the walking curriculum.
        """
        if failed:
            self.recovery=True;self.active=None
        if previous_success and self.active is not None:
            if self.active.kind=='KICK':self.raised_front=self.active.pair[0]
            else:self.raised_front=None
            self.active=None
        index=max(0,math.floor(time/(4*self.cfg.beat)))
        candidates=self.phrase(index)+self.phrase(index+1)
        for event in candidates:
            identity=(event.phrase,event.kind,event.target)
            if event.start<=time<event.start+.021 and identity!=self.last_id:
                self.last_id=identity
                valid=all(confirmed_contacts[j] for j in range(4) if j!=self.raised_front)
                if event.from_kick:valid &= self.raised_front in event.pair
                if self.active is not None or self.recovery or not valid:
                    self.missed+=1;return None
                self.active=event;return event
        return None

    def recover(self,confirmed_contacts,strategy='land_and_rejoin'):
        """Explicit recovery request; never silently count recovery as a musical STEP.

        land_and_rejoin requests a slow controlled landing then awaits four-foot
        support. hold_and_stop preserves the three-foot reference for an external
        supervisor. Neither strategy resets the musical clock.
        """
        if strategy not in ('land_and_rejoin','land_and_stop','hold_and_stop'):raise ValueError(strategy)
        if strategy=='land_and_rejoin' and all(confirmed_contacts):
            self.recovery=False;self.raised_front=None
        return strategy

    @property
    def static_hold_before_next_step(self):
        return 1.5*self.cfg.beat-self.cfg.kick_duration-self.cfg.swing_seconds
