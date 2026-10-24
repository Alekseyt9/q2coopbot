"""Weapon-scoped prototype: memory gaze on all weapons, fire residual only MG."""
from combat_memory_recovery import torch,nn,INPUT_WIDTH,PARAMETERS,active_mask,inputs,build_zero

VERSION='combat_memory_recovery_prototype_v2'


def project(head,features,raw):
    assert all(p.is_cuda for p in head.parameters())
    delta=head(inputs(features,raw));assert delta.shape==(len(features),3)
    assert bool(torch.isfinite(delta).all())
    active=active_mask(features)
    # v10 equipped weapon index4 is MG; inventory/request masks are not equipment.
    fire=active & (features[:,1125]==1)
    result=raw.clone()
    result[:,2:4]=raw[:,2:4]+delta[:,:2]*active[:,None]
    result[:,4]=raw[:,4]+delta[:,2]*fire
    return result
