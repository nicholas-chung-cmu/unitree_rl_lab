"""Bounded time-averaged duration costs; usable without Isaac for replay scoring."""
import torch


def error(duration, low, high, scale):
    distance=(low-duration).clamp_min(0)+(duration-high).clamp_min(0)
    return 1-torch.exp(-(distance/scale).square())


class DurationState:
    def __init__(self,n,device):
        self.contact=torch.ones(n,4,dtype=torch.bool,device=device)
        self.age=torch.zeros(n,4,device=device)
        self.valid=torch.zeros_like(self.contact)
        self.active=torch.zeros(n,dtype=torch.bool,device=device)
        self.air=torch.zeros_like(self.age)
        self.stance=torch.zeros_like(self.age)

    def step(self,contact,active,reset,dt):
        fresh=reset | ~active | ~self.active
        self.age[fresh]=0; self.valid[fresh]=False
        self.air[fresh]=0; self.stance[fresh]=0
        self.contact[fresh]=contact[fresh]
        change=(contact != self.contact) & ~fresh[:,None]
        # Hold completed-duration costs in time, not per event. This avoids
        # making the cost simply proportional to number of transitions.
        landing=change & contact & self.valid
        leaving=change & ~contact & self.valid
        self.air=torch.where(landing,error(self.age,.16,.24,.20),self.air)
        self.stance=torch.where(leaving,error(self.age,.70,.90,.80),self.stance)
        self.valid |= change
        self.age=torch.where(change,torch.zeros_like(self.age),self.age)+dt
        # Partial initial intervals cannot be scored for being too short; an
        # ongoing overrun still costs, including indefinitely planted feet.
        air_over=1-torch.exp(-((self.age-.24).clamp_min(0)/.20).square())
        stance_over=1-torch.exp(-((self.age-.90).clamp_min(0)/.80).square())
        a=torch.maximum(self.air,torch.where(~contact,air_over,0.))
        s=torch.maximum(self.stance,torch.where(contact,stance_over,0.))
        self.contact=contact.clone(); self.active=active.clone()
        return torch.stack((a.mean(-1),s.mean(-1)),dim=-1)*active[:,None]
