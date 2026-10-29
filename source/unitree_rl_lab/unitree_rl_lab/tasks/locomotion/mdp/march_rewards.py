"""Dense march-first swing discovery and missing-swing costs; no Isaac imports."""
import torch


def raw_terms(target_contact,target_height,contact,height,swing_height=.03,std=.015):
    emphasis=(target_height/swing_height).square()
    quality=(height/target_height.clamp_min(.015)).clamp(0,1)*(1-contact)
    quality*=torch.exp(-((height-target_height)/std).square())
    clearance=(emphasis*quality).sum(-1)/emphasis.sum(-1).clamp_min(1)
    # Zero at neutral/boundary targets; full cost only for explicitly scheduled
    # swing feet remaining in contact. Never penalizes the supporting diagonal.
    expected_air=((.5-target_contact)/.5).clamp(0,1)
    missed=(expected_air*contact).sum(-1)/2
    opportunity=expected_air.sum(-1)/2
    return torch.stack((clearance,missed,opportunity),-1)


def terms(env):
    from .conga_reference import generate_reference
    term=env.command_manager.get_term('base_velocity')
    cfg=term.cfg.reference
    if not hasattr(term,'_march_norm'):
        time=torch.arange(round(4/env.step_dt),device=env.device,dtype=torch.float64)*env.step_dt
        ref=generate_reference(time,cfg)
        ideal=(ref.contact>=.5-1e-6).double()
        raw=raw_terms(ref.contact,ref.foot_height,ideal,ref.foot_height,cfg.swing_height)
        term._march_norm=raw.mean(0)[[0,2]].float().clamp_min(1e-8)
    _,height,contact=term.foot_state()
    ref=term.reference
    raw=raw_terms(ref.contact,ref.foot_height,contact.float(),height,cfg.swing_height)
    return raw[:,:2]/term._march_norm


def individual_clearance(env):return terms(env)[:,0]


def missed_swing(env):return terms(env)[:,1]
