"""Sagittal reflection of the explicitly versioned CPG 75/90/12 interface.

Only left/right reflection is used. Front/rear reflection would change the kick
semantics. Musical phrase phase stays fixed; diagonal gait phase shifts by pi.
"""
import torch

JOINT_SWAP=(1,0,3,2,5,4,7,6,9,8,11,10)
FOOT_SWAP=(1,0,3,2)


def mirror_joints(x):
    result=x[...,list(JOINT_SWAP)].clone()
    result[...,:4]*=-1
    return result


def mirror_observation(x):
    if x.shape[-1] not in (75,90):raise ValueError('Unsupported CPG observation version')
    y=x.clone();critic=x.shape[-1]==90
    offset=3 if critic else 0
    if critic:y[...,1]*=-1  # polar linear velocity
    y[...,offset+0]*=-1;y[...,offset+2]*=-1  # axial angular velocity
    y[...,offset+4]*=-1  # gravity vector y
    joint_start=offset+6
    count=4 if critic else 3  # critic includes effort before last action
    for i in range(count):
        start=joint_start+12*i;y[...,start:start+12]=mirror_joints(x[...,start:start+12])
    start=57 if critic else 42
    y[...,start:start+12]=mirror_joints(x[...,start:start+12])
    y[...,start+12:start+14]*=-1  # diagonal phase, not musical phrase phase
    y[...,start+16:start+20]=x[...,start+16:start+20][...,list(FOOT_SWAP)]
    y[...,start+21:start+23]*=-1  # vy and yaw command
    y[...,start+23:start+27]=x[...,start+23:start+27][...,list(FOOT_SWAP)]
    y[...,start+30]=x[...,start+31];y[...,start+31]=x[...,start+30]  # raised FL/FR
    return y


class SagittalActor(torch.nn.Module):
    """Exactly reflection-equivariant mean with shared MLP weights.

    The actor must receive the unnormalized, versioned 75-input interface.
    Buffers make the complete transform part of checkpoints and JIT/ONNX exports.
    PPO trains through both evaluations of the same MLP.
    """
    def __init__(self,base):
        super().__init__();self.base=base
        observation=mirror_observation(torch.arange(1,76,dtype=torch.float32))
        action=mirror_joints(torch.arange(1,13,dtype=torch.float32))
        self.register_buffer('observation_permutation',observation.abs().long()-1)
        self.register_buffer('observation_sign',observation.sign())
        self.register_buffer('action_permutation',action.abs().long()-1)
        self.register_buffer('action_sign',action.sign())

    def forward(self,observation:torch.Tensor)->torch.Tensor:
        direct=self.base(observation)
        reflected=self.base(observation.index_select(-1,self.observation_permutation)*self.observation_sign)
        reflected=reflected.index_select(-1,self.action_permutation)*self.action_sign
        return .5*(direct+reflected)

    def __getitem__(self,index):
        # The installed Isaac exporter reads actor[0].in_features to construct
        # its dummy input; actual inference still calls this wrapper's forward.
        return self.base[index]


@torch.no_grad()
def augment(env=None,obs=None,actions=None):
    if env is not None and not getattr(env.unwrapped,'_cpg_symmetry_checked',False):
        expected=[f'{foot}_{joint}_joint' for joint in ('hip','thigh','calf') for foot in ('FL','FR','RL','RR')]
        if list(env.unwrapped.scene['robot'].data.joint_names)!=expected:
            raise ValueError('CPG reflection requires explicit FL/FR/RL/RR joint-block ordering')
        env.unwrapped._cpg_symmetry_checked=True
    out=None
    if obs is not None:
        n=obs.batch_size[0];out=obs.repeat(2)
        for key in ('policy','critic'):
            if key in obs:out[key][n:]=mirror_observation(obs[key])
    action_out=torch.cat((actions,mirror_joints(actions)),0) if actions is not None else None
    return out,action_out
