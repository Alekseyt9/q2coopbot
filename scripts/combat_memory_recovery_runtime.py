"""Versioned optional actor-raw recovery projection; CUDA neural execution only."""
import math
from combat_memory_recovery_v2 import torch,nn,build_zero,project

VERSION='combat_memory_coarse_recovery_v1'


def validate(model):
    spec=model.get('memory_recovery')
    if spec is None:return
    assert model['feature_version']=='combat_features_v10' and model.get('spatial_aim') and not model.get('entity_attention')
    assert model.get('aim_mode_head')=='combat_target_coarse_fine_aim_v1'
    assert set(spec)=={'version','layers'} and spec['version']==VERSION and len(spec['layers'])==3
    width=46
    for layer,n in zip(spec['layers'],(32,16,3)):
        assert set(layer)=={'weight','bias'} and len(layer['weight'])==len(layer['bias'])==n
        assert all(len(row)==width and all(math.isfinite(v) and abs(v)<=1e4 for v in [*row,b]) for row,b in zip(layer['weight'],layer['bias']))
        width=n


def build(model):
    validate(model)
    if model.get('memory_recovery') is None:return None
    head=build_zero()
    with torch.no_grad():
        for index,layer in zip((0,2,4),model['memory_recovery']['layers']):
            head[index].weight.copy_(torch.tensor(layer['weight'],device='cuda',dtype=head[index].weight.dtype))
            head[index].bias.copy_(torch.tensor(layer['bias'],device='cuda',dtype=head[index].bias.dtype))
    return head


def apply(head,features,raw):
    shape=raw.shape;assert shape[:-1]==features.shape[:-1] and shape[-1]==81
    return project(head,features.reshape(-1,1147),raw.reshape(-1,81)).reshape(shape)


def export(model,head):
    if head is None:
        assert model.get('memory_recovery') is None;return
    from ppo_combat import layers
    model['memory_recovery']=dict(version=VERSION,layers=layers(head));validate(model)
