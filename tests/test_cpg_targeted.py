"""Training curriculum contracts; deployment commands stay externally supplied."""
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from deploy.cpg.runtime import load_control_modules
core,_=load_control_modules()


def test_legacy_training_bias_preserves_samples_and_rng():
    x=torch.randn(100,3);before=x.clone();state=torch.random.get_rng_state()
    core.training_direction_bias(x,core.GaitConfig())
    torch.testing.assert_close(x,before)
    assert torch.equal(state,torch.random.get_rng_state())


def test_targeted_curriculum_retains_reverse_and_turning_examples():
    torch.manual_seed(73);x=torch.ones(10000,3)*.1
    cfg=core.GaitConfig(training_backward_fraction=.1,training_turn_fraction=.15)
    out=core.training_direction_bias(x,cfg)
    turn=(out[:,:2]==0).all(-1);reverse=out[~turn,0]<0
    assert .13<float(turn.float().mean())<.17
    assert .08<float(reverse.float().mean())<.12
    assert (out[:,2]==.1).all()
