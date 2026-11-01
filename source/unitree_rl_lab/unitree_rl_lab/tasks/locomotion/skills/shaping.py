"""Dense STEP lift shaping and mild body-height deviation cost, in metres."""
import torch


def clearance_progress(height, target_height, scale=.04):
    """Signed improvement over a grounded foot's absolute height error.

    Ground earns zero; lifting toward the reference improves continuously.
    Overshoot loses credit, so remaining aloft beyond touchdown is not rewarded.
    """
    return ((target_height.abs()-(height-target_height).abs())/scale).clamp(-1, 1)


def swing_placement_gate(height, target_height, transition=.008):
    """Full placement credit needs the requested clearance during swing.

    Ground transition is soft and placement is ungated at the landing endpoint.
    Only clearance is used: no discontinuous force threshold blocks exploration.
    """
    air_fraction = (target_height/transition).clamp(0, 1)
    achieved = (height/target_height.clamp_min(1e-6)).clamp(0, 1)
    return 1-air_fraction+air_fraction*achieved


def body_height_deviation(height, target_height, tolerance=.01, scale=.05):
    """Bounded quadratic cost: 1 cm tolerance, 5 cm normalization, maximum 4."""
    return ((height-target_height).abs().sub(tolerance).clamp_min(0)/scale).square().clamp(max=4)
