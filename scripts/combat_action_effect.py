"""CUDA-only causal motor inputs and factorized forward model; no actor control."""
import math
from train_combat_bc import torch,nn
from combat_action_geometry import features_for_version,CENTERED_VERSION

POSITION_COLUMNS=tuple(list(range(6))+list(range(13,73))+list(range(185,190))+list(range(166,169))+list(range(190,192)))
DIRECTION_COLUMNS=tuple(list(range(6))+list(range(8,10))+list(range(15,17))+list(range(181,188))+[165,161]+list(range(190,192)))


def inputs(causal,timing):
    assert causal.is_cuda and timing.is_cuda and causal.ndim==2 and causal.shape[1]==185
    assert timing.shape==(len(causal),2)
    yaw=torch.atan2(causal[:,6],causal[:,7]);cy,sy=yaw.cos(),yaw.sin()
    velocity=causal[:,10:13]*400
    local=torch.stack((cy*velocity[:,0]+sy*velocity[:,1],-sy*velocity[:,0]+cy*velocity[:,1],velocity[:,2]),-1)/400
    delta=causal[:,164]*math.pi;f,s=causal[:,162],causal[:,163]
    requested=torch.stack((f*delta.cos()+s*delta.sin(),f*delta.sin()-s*delta.cos()),-1)
    return torch.cat((causal,local,requested,timing),-1)


class FactorizedMotor(nn.Module):
    def __init__(self):
        super().__init__()
        assert torch.cuda.is_available()
        self.register_buffer('position_columns',torch.tensor(POSITION_COLUMNS,dtype=torch.long,device='cuda'))
        self.register_buffer('direction_columns',torch.tensor(DIRECTION_COLUMNS,dtype=torch.long,device='cuda'))
        self.position=nn.Sequential(nn.Linear(76,32,device='cuda'),nn.ReLU(),nn.Linear(32,3,device='cuda'))
        self.direction=nn.Sequential(nn.Linear(21,16,device='cuda'),nn.ReLU(),nn.Linear(16,2,device='cuda'))
        with torch.no_grad():
            for branch in (self.position,self.direction):
                branch[-1].weight.zero_();branch[-1].bias.zero_()
    def forward(self,x):
        assert x.is_cuda and x.ndim==2 and x.shape[1]==192
        return torch.cat((self.position(x[:,self.position_columns]),self.direction(x[:,self.direction_columns])),-1)


def projected_geometry(causal,motor,extrapolate_target=True):
    """Use forecasts only; native motor targets must never be passed here at inference.

    Muzzle displacement is normalized /32 in the OLD observed yaw-local frame.
    Move the target oppositely in that frame, leaving centered muzzle height
    unchanged. This represents translated future eye/muzzle, including a
    predicted duck-height change. Actor inputs/actions themselves are untouched.
    """
    assert causal.is_cuda and motor.is_cuda and motor.shape==(len(causal),5)
    shifted=causal.clone()
    if extrapolate_target:
        shifted[:,101:104]+=causal[:,104:105]*causal[:,105:108]*(400*.1/512)
    shifted[:,101:104]-=motor[:,:3]*(32/512)
    shifted[:,164]+=motor[:,4]*(5/180)
    shifted[:,165]+=motor[:,3]*(5/180)
    return features_for_version(shifted,CENTERED_VERSION)
