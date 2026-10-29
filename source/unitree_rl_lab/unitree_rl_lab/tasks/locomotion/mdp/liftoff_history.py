"""Vectorized causal last-diagonal gate, matching the offline 50 Hz audit."""
import torch


class LiftoffHistory:
    def __init__(self, n, device):
        self.last = torch.full((n,), -1, device=device, dtype=torch.long)
        self.previous_step = torch.full((n,), -2, device=device, dtype=torch.long)
        self.previous_contact = torch.zeros((n,4), device=device, dtype=torch.bool)
        self.off = torch.full((n,4), -100000, device=device, dtype=torch.long)
        self.air = torch.zeros((n,2), device=device, dtype=torch.long)
        self.ground = torch.zeros_like(self.air)
        self.armed = torch.zeros((n,2), device=device, dtype=torch.bool)
        self.eligible = torch.zeros_like(self.armed)

    def update(self, contact, step):
        c = contact.bool()
        reset = step != self.previous_step + 1
        self.last[reset] = -1
        self.off[reset] = -100000
        self.air[reset] = 0
        self.ground[reset] = 0
        self.armed[reset] = False
        self.eligible[reset] = False
        crossed = self.previous_contact & ~c & ~reset[:,None]
        self.off = torch.where(crossed, step[:,None], self.off)
        self.off[c] = -100000
        pairs = torch.stack((c[:,[0,3]], c[:,[1,2]]),1)
        both_air = (~pairs).all(-1)
        both_ground = pairs.all(-1)
        self.air = torch.where(both_air, self.air+1, 0)
        self.ground = torch.where(both_ground, self.ground+1, 0)
        landed = self.ground >= 2
        self.armed |= landed
        self.eligible &= ~landed
        ready = self.armed & (self.air >= 2)
        self.armed &= ~ready
        times = torch.stack((self.off[:,[0,3]], self.off[:,[1,2]]),1)
        candidates = ready & (times.min(-1).values >= 0) & ((times.max(-1).values-times.min(-1).values) <= 2)
        unique = (candidates.sum(-1)==1) & c.any(-1)
        pair = candidates.long().argmax(-1)
        for p in (0,1):
            event = unique & (pair==p)
            value = (self.last>=0) & (self.last!=p)
            self.eligible[:,p] = torch.where(event,value,self.eligible[:,p])
        self.eligible &= ~(candidates & ~unique[:,None])
        self.last = torch.where(unique,pair,self.last)
        self.previous_contact = c.clone()
        self.previous_step = step.clone()
        return self.eligible & both_air


def paired_step(env):
    from . import cadence_pair_scores
    term = env.command_manager.get_term('base_velocity')
    if not hasattr(term, '_liftoff_history'):
        if abs(env.step_dt-.02)>1e-8:
            raise ValueError('Reviewed history gate requires 50 Hz control')
        term._liftoff_history = LiftoffHistory(env.num_envs, env.device)
        term._history_tick = None
    if term._history_tick != env.common_step_counter:
        _, _, contact = term.foot_state()
        term._history_gate = term._liftoff_history.update(contact, env.episode_length_buf)
        term._history_tick = env.common_step_counter
    ref = term.reference
    scheduled = torch.where(ref.foot_height[:,0]>=ref.foot_height[:,1],0,1)
    gate = term._history_gate.gather(1,scheduled[:,None]).squeeze(1)
    return cadence_pair_scores.paired_step(env)*gate
