"""Observed-state weapon residual; numerical work requires CUDA.

Direct observed state + soft selected-target context -> weapon-only residual.
No prescribed optimal weapon, availability changes or native hidden state.
"""
import math
from train_combat_bc import torch,nn
from combat_target_head import availability

VERSION='combat_observed_weapon_state_residual_prototype_v1'
CONTRACT_VERSION='combat_observed_weapon_state_residual_v1'
INPUT_WIDTH=154


def validate(model):
    spec=model.get('weapon_state_residual')
    if spec is None:return
    if (model.get('feature_version')!='combat_features_v10' or
        model.get('aim_mode_head')!='combat_target_coarse_fine_aim_v1' or
        model.get('weapon_head')!='combat_masked_weapon_v1' or
        model.get('target_head')!='combat_target_conditioned_aim_v1' or
        not model.get('spatial_aim') or model.get('entity_attention') or
        spec.get('version')!=CONTRACT_VERSION or len(spec.get('layers',[]))!=3):
        raise ValueError('Invalid observed weapon residual contract')
    width=INPUT_WIDTH
    for layer,out in zip(spec['layers'],(64,32,12)):
        if (len(layer.get('bias',[]))!=out or len(layer.get('weight',[]))!=out or
            any(len(row)!=width for row in layer['weight']) or
            any(not math.isfinite(v) or abs(v)>1e4 for row,bias in zip(layer['weight'],layer['bias']) for v in [*row,bias])):
            raise ValueError('Invalid observed weapon residual parameters')
        width=out


def build(model):
    validate(model)
    if model.get('weapon_state_residual') is None:return None
    from ppo_combat import network
    return network(model['weapon_state_residual']['layers']).to('cuda')


def apply(branch,features,raw):
    correction=branch(inputs(features,raw));result=raw.clone()
    result[...,8:20]=result[...,8:20]+correction
    return result


def export(model,branch):
    if branch is None:return
    from ppo_combat import layers
    model['weapon_state_residual']={**model['weapon_state_residual'],'layers':layers(branch)}
    validate(model)


def optimizer(actor,std,actor_lr,weapon_lr):
    """Keep old parameter IDs and log_std; append the new branch in group2."""
    branch=getattr(actor,'weapon_state_branch',None)
    if branch is None:return torch.optim.Adam(list(actor.parameters())+[std],lr=actor_lr)
    if not math.isfinite(weapon_lr) or not 0<weapon_lr<=.01:raise ValueError('Invalid weapon residual learning rate')
    old=[value for name,value in actor.named_parameters() if not name.startswith('weapon_state_branch.')]+[std]
    return torch.optim.Adam([dict(params=old,lr=actor_lr),
        dict(params=list(branch.parameters()),lr=weapon_lr,name='weapon_state_residual')],lr=actor_lr)


def inputs(features,raw):
    if not features.is_cuda or not raw.is_cuda:
        raise ValueError('CUDA required for weapon residual prototype')
    if features.shape[:-1]!=raw.shape[:-1] or features.shape[-1]!=1147 or raw.shape[-1]!=81:
        raise ValueError('Expected aligned v10 observations/precision outputs')
    shape=features.shape[:-1];x=features.reshape(-1,1147);r=raw.reshape(-1,81)
    if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(r).all()):
        raise ValueError('Nonfinite observed prototype inputs')
    valid=availability(x)
    probabilities=r[:,20:29].masked_fill(~valid,-torch.inf).softmax(-1)
    enemy=x[:,73:169].reshape(-1,8,12);typed=x[:,466:786].reshape(-1,8,40)
    weights=probabilities[:,1:]
    selected_enemy=(weights[...,None]*enemy).sum(1)
    selected_type=(weights[...,None]*typed).sum(1)
    present=enemy[...,:1]>0
    count=present.sum(1).clamp_min(1)
    mean=torch.where(present,enemy,0).sum(1)/count
    maximum=enemy.masked_fill(~present,-torch.inf).amax(1)
    maximum=torch.where(present.any(1),maximum,0)
    # Common20 includes observed gun_frame/30 at index5; inventory31 and
    # current/own-request26 preserve known/unknown masks and observed ages.
    result=torch.cat((x[:,:20],x[:,814:845],x[:,1121:1147],selected_enemy,
        selected_type,mean,maximum,probabilities[:,:1]),-1)
    assert result.shape[-1]==INPUT_WIDTH and bool(torch.isfinite(result).all())
    return result.reshape(*shape,INPUT_WIDTH)


class WeaponStateResidual(nn.Module):
    def __init__(self):
        super().__init__()
        if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
        self.branch=nn.Sequential(nn.Linear(INPUT_WIDTH,64,device='cuda'),nn.ReLU(),
            nn.Linear(64,32,device='cuda'),nn.ReLU(),nn.Linear(32,12,device='cuda'))
        with torch.no_grad(): self.branch[-1].weight.zero_();self.branch[-1].bias.zero_()

    def forward(self,features,raw):
        return apply(self.branch,features,raw)
