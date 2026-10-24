"""Experimental CUDA residual for fresh memory-only coarse gaze and fire.

This module is not a supported runtime policy header. Native integration and
checkpoint migration must be admitted separately before policy export.
"""
from train_combat_bc import torch,nn

VERSION='combat_memory_recovery_prototype_v1'
INPUT_WIDTH=46
PARAMETERS=2083


def active_mask(features):
    assert features.is_cuda and features.ndim==2 and features.shape[1]==1147
    visible=features[:,73:169].reshape(-1,8,12)[:,:,0].bool().any(-1)
    memory=features[:,881:911]
    return ~visible & (memory[:,0]==1) & (memory[:,1]<=10./200.)


def inputs(features,raw):
    assert raw.is_cuda and raw.shape==(len(features),81)
    assert bool(torch.isfinite(features).all()) and bool(torch.isfinite(raw).all())
    x=torch.cat((features[:,:13].detach(),features[:,881:911].detach(),raw[:,2:5].detach().clamp(-20,20)),-1)
    assert x.shape==(len(features),INPUT_WIDTH)
    return x


def build_zero(seed=2026102409):
    assert torch.cuda.is_available(),'CUDA only; no CPU neural fallback'
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed)
        head=nn.Sequential(nn.Linear(INPUT_WIDTH,32,device='cuda'),nn.ReLU(),
            nn.Linear(32,16,device='cuda'),nn.ReLU(),nn.Linear(16,3,device='cuda'))
        with torch.no_grad():head[-1].weight.zero_();head[-1].bias.zero_()
    assert sum(p.numel() for p in head.parameters())==PARAMETERS
    return head


def project(head,features,raw):
    assert all(p.is_cuda for p in head.parameters())
    delta=head(inputs(features,raw));assert delta.shape==(len(features),3)
    assert bool(torch.isfinite(delta).all())
    result=raw.clone()
    result[:,2:5]=raw[:,2:5]+delta*active_mask(features)[:,None]
    return result
