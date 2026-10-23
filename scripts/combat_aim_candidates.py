"""CUDA prototype for an augmented PPO aim choice. Not yet a runtime policy.

The action event is (original Gaussian latent, categorical correction). Physical
angles are derived afterward. The Gaussian must NEVER score transformed latent.
"""
import math
from train_combat_bc import torch,nn

VERSION='combat_joint_aim_candidates_v1'
INPUT_WIDTH=1359  # complete1147 observation + base-action185 + hint25 + timing2
OFFSETS=(0.,-5.,-2.5,2.5,5.)


def inputs(features,base_action,hit_logits,timing):
    n=len(features)
    for value,width in ((features,1147),(base_action,185),(hit_logits,25),(timing,2)):
        assert value.is_cuda and value.shape==(n,width) and bool(torch.isfinite(value).all())
    # Separate score-function gradients train the source policy. A candidate
    # head must not introduce a pathwise gradient through sampled actions/hints.
    return torch.cat((features.detach(),base_action.detach(),hit_logits.detach(),timing.detach()),-1)


def eligible(base_action):
    assert base_action.is_cuda and base_action.ndim==2 and base_action.shape[1]==185
    return (base_action[:,77]>0)&(base_action[:,151]==0)&(base_action[:,99]>0)&(base_action[:,147]>0)


class AimCandidateHead(nn.Module):
    def __init__(self):
        super().__init__()
        assert torch.cuda.is_available()
        self.layers=nn.Sequential(nn.Linear(INPUT_WIDTH,64,device='cuda'),nn.ReLU(),nn.Linear(64,25,device='cuda'))
        with torch.no_grad():
            self.layers[-1].weight.zero_();self.layers[-1].bias.zero_()
            # Same marginal distribution as predictive exploration:0.5baseline,
            # remaining0.5 split evenly over24nonzero corrections.
            self.layers[-1].bias[0]=math.log(24)
    def forward(self,x):
        assert x.is_cuda and x.ndim==2 and x.shape[1]==INPUT_WIDTH
        return self.layers(x)


def choose(logits,gate_uniform,offset_uniform,active):
    assert logits.is_cuda and logits.ndim==2 and logits.shape[1]==25
    n=len(logits)
    assert active.is_cuda and active.dtype==torch.bool and active.shape==(n,)
    for u in (gate_uniform,offset_uniform):
        assert u.is_cuda and u.shape==(n,) and bool(((u>=0)&(u<1)).all())
    # Two independent uniforms permit exact initial assignment compatibility
    # with the established frame-hash sampler, without actor RNG consumption.
    p=logits.softmax(-1)
    tail=logits[:,1:].softmax(-1)
    cdf=tail.cumsum(-1);cdf[:,-1]=1
    offset=torch.searchsorted(cdf.contiguous(),offset_uniform[:,None].contiguous(),right=True)[:,0]+1
    return torch.where(active&(gate_uniform>=p[:,0]),offset,torch.zeros_like(offset))


def log_probability(logits,choice,active):
    assert logits.is_cuda and logits.ndim==2 and logits.shape[1]==25
    assert choice.is_cuda and choice.dtype==torch.long and choice.shape==(len(logits),)
    assert bool(((choice>=0)&(choice<25)).all()) and bool((active|(choice==0)).all())
    value=logits.log_softmax(-1).gather(1,choice[:,None])[:,0]
    return torch.where(active,value,torch.zeros_like(value))


def entropy(logits,active):
    logp=logits.log_softmax(-1)
    value=-(logp.exp()*logp).sum(-1)
    return torch.where(active,value,torch.zeros_like(value))


def apply_offsets(source_latent,mode,choice):
    assert source_latent.is_cuda and source_latent.ndim==2 and source_latent.shape[1]==4
    assert mode.is_cuda and mode.dtype==torch.long and mode.shape==(len(source_latent),)
    assert choice.is_cuda and choice.dtype==torch.long and choice.shape==mode.shape
    assert bool(((mode==0)|(mode==1)).all()) and bool(((choice>=0)&(choice<25)).all())
    offsets=torch.tensor(OFFSETS,dtype=source_latent.dtype,device='cuda')
    delta=torch.stack((offsets[choice//5],offsets[choice%5]),-1)
    scale=torch.where(mode==1,15.,180.).to(source_latent.dtype)[:,None]
    aim=((source_latent[:,2:].tanh()*scale+delta)/scale).clamp(-1+1e-6,1-1e-6).atanh()
    # Candidate0 preserves exact source latent, including saturated edge cases.
    aim=torch.where((choice!=0)[:,None],aim,source_latent[:,2:])
    return torch.cat((source_latent[:,:2],aim),-1)


def joint_log_probability(source_log_probability,logits,choice,active):
    # Caller supplies original-Gaussian/category terms, but the fire term must
    # be conditioned on FINAL physical aim when conditional fire is nonzero.
    assert source_log_probability.is_cuda and source_log_probability.shape==(len(logits),)
    return source_log_probability+log_probability(logits,choice,active)


MASK=(1<<64)-1
def _mix(x):
    x=((x^(x>>30))*0xbf58476d1ce4e5b9)&MASK
    x=((x^(x>>27))*0x94d049bb133111eb)&MASK
    return x^(x>>31)


def frame_uniforms(seed,frame,life):
    """Pure PRNG metadata, not neural computation; source actor RNG untouched."""
    word=_mix((seed^(frame*0x9e3779b97f4a7c15)^(life*0xd1b54a32d192ed03))&MASK)
    k=_mix(word^0x8cb92baa3f3d8dd7)%24
    jitter=(_mix(word^0x4f1bbcdc6762c5d9)>>11)/2**53
    return (word>>11)/2**53,(k+jitter)/24
