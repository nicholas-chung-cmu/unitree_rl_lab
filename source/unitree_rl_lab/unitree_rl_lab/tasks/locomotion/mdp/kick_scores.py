"""Pure Torch kick reward components, for shared runtime and offline scoring."""
import torch


def components(body_y,height,contact,reference,lateral_std=.02,height_std=.015,height_aware=False):
    rows=torch.arange(len(height),device=height.device)
    leg=reference.kick_leg
    lateral=reference.kick_weight*torch.exp(-((body_y[rows,leg]-reference.kick_y)/lateral_std).square())
    clearance=reference.kick_weight*torch.exp(-((height[rows,leg]-reference.foot_height[rows,leg])/height_std).square())
    peak=reference.foot_height.amax(-1)
    # Pulse is inferred from the target path at the fixed introductory .02 m peak.
    middle=(peak/.02).clamp(0,1).square()
    airborne=reference.kick_weight*middle*(~contact[rows,leg])
    if height_aware:
        airborne=airborne*(height[rows,leg]/.01).clamp(0,1)
    # Smoothly remove the requested foot from support during its kick, while
    # always requiring the other three. All four are supported outside requests.
    mask=torch.arange(4,device=height.device)[None,:]==leg[:,None]
    support=1-reference.kick_weight[:,None]*mask
    missing=(support*(~contact)).sum(-1)/support.sum(-1)
    return torch.stack((clearance,lateral,airborne,missing),dim=-1)

