"""Observed motor residual and explicit stable reference; CUDA neural work only."""
import copy
import math
from train_combat_bc import torch,nn

VERSION='combat_observed_motor_residual_v1'
INPUT_WIDTH=756


def validate(model):
    spec=model.get('motor_residual')
    if spec is None:return
    assert model['feature_version']=='combat_features_v10' and model.get('aim_mode_head')=='combat_target_coarse_fine_aim_v1'
    assert model.get('spatial_aim') and model.get('sampled_motion_aim') and not model.get('entity_attention')
    assert spec['version']==VERSION and len(spec['layers'])==3
    width=INPUT_WIDTH
    for layer,out in zip(spec['layers'],(64,32,2)):
        assert len(layer['bias'])==out and len(layer['weight'])==out
        assert all(len(row)==width for row in layer['weight'])
        assert all(math.isfinite(v) and abs(v)<=1e4 for row,bias in zip(layer['weight'],layer['bias']) for v in [*row,bias])
        width=out


def inputs(features):
    assert features.is_cuda and features.shape[-1]==1147
    return torch.cat((features[...,:169],features[...,466:786],features[...,854:1121]),-1)


def build(model):
    validate(model)
    if not model.get('motor_residual'):return None
    from ppo_combat import network
    return network(model['motor_residual']['layers']).to('cuda')


def apply(module,features,raw):
    assert raw.is_cuda and raw.shape[-1]==81
    result=raw.clone();result[...,:2]=result[...,:2]+module(inputs(features))
    return result


def reference(module,features,raw):
    if module is None:return raw
    result=raw.clone();result[...,:2]=result[...,:2]-module(inputs(features))
    return result


def export(model,module):
    if module is None:return
    from ppo_combat import layers
    model['motor_residual']=dict(version=VERSION,layers=layers(module))
    validate(model)


def initialize(model):
    assert torch.cuda.is_available() and not model.get('motor_residual')
    result=copy.deepcopy(model)
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(20261022)
        module=nn.Sequential(nn.Linear(INPUT_WIDTH,64,device='cuda'),nn.ReLU(),
                             nn.Linear(64,32,device='cuda'),nn.ReLU(),nn.Linear(32,2,device='cuda'))
        with torch.no_grad():module[-1].weight.zero_();module[-1].bias.zero_()
        export(result,module)
    return result


def freeze_scope(actor,std,optimizer,train_motor=True):
    assert actor.motor_branch is not None
    parameters=list(actor.parameters())+[std]
    motor=set(actor.motor_branch.parameters())
    selected=motor if train_motor else set(parameters)-motor
    optimizer.zero_grad(set_to_none=True)
    frozen={p:p.detach().clone() for p in parameters if p not in selected}
    states={p:copy.deepcopy(optimizer.state.get(p,{})) for p in frozen}
    for p in parameters:p.requires_grad_(p in selected)
    def project():
        # Guarded fallback clears every Adam first moment; restore frozen ones.
        with torch.no_grad():
            for p,value in frozen.items():
                p.copy_(value)
                if states[p]:optimizer.state[p]=copy.deepcopy(states[p])
                else:optimizer.state.pop(p,None)
    return project,sum(p.numel() for p in selected)
