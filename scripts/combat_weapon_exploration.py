"""Observed-state weapon temperature; differentiable CUDA PPO contract."""
from combat_body_features import VERSIONS as BODY_FEATURE_VERSIONS, WIDTH as BODY_FEATURE_WIDTH, legacy as body_legacy
import math
from train_combat_bc import torch

VERSION='combat_observed_grenades_weapon_temperature_v1'
BLASTER_VERSION='combat_observed_blaster_grenades_weapon_temperature_v2'


def validate(model):
    spec=model.get('weapon_exploration')
    if spec is None:return
    temperature=spec.get('temperature')
    blaster=spec.get('blaster_temperature')
    valid_version=(spec.get('version')==VERSION and set(spec)=={'version','temperature'} or
        spec.get('version')==BLASTER_VERSION and set(spec)=={'version','temperature','blaster_temperature'} and
        type(blaster) in (int,float) and math.isfinite(blaster) and 1<=blaster<=4)
    if (not valid_version or
        isinstance(temperature,bool) or not isinstance(temperature,(int,float)) or
        not math.isfinite(temperature) or not 1<=temperature<=4 or
        model.get('feature_version') not in BODY_FEATURE_VERSIONS or
        model.get('weapon_head')!='combat_masked_weapon_v1' or
        model.get('aim_mode_head')!='combat_target_coarse_fine_aim_v1' or
        not model.get('spatial_aim') or model.get('entity_attention')):
        raise ValueError('Invalid observed weapon exploration contract')


def apply(spec,features,raw):
    if spec is None or (spec['temperature']==1 and spec.get('blaster_temperature',1)==1):return raw
    features = body_legacy(features)
    if not features.is_cuda or not raw.is_cuda:
        raise ValueError('CUDA required for weapon exploration')
    if features.shape[:-1]!=raw.shape[:-1] or features.shape[-1]!=1147 or raw.shape[-1]!=81:
        raise ValueError('Invalid exploration features/outputs')
    x=features.reshape(-1,1147)
    # Temporal batches include zero-padded context rows. They never enter
    # likelihood evaluation and cannot enable this observed-state gate.
    values=x[:,833:845]
    if not bool(((values==0)|(values==1)).all()):
        raise ValueError('Invalid exploration availability flags')
    mask=values.bool()
    gate=((x[:,1132]==1)&mask[:,1:5].any(-1)).reshape(*features.shape[:-1],1)
    if spec['version']==BLASTER_VERSION:
        blaster_gate=((x[:,1122]==1)&mask[:,2:].any(-1)).reshape(*features.shape[:-1],1)
        temperature=torch.where(blaster_gate,torch.full_like(raw[...,:1],spec['blaster_temperature']),
            torch.where(gate,torch.full_like(raw[...,:1],spec['temperature']),torch.ones_like(raw[...,:1])))
        result=raw.clone()
        result[...,8:20]=raw[...,8:20]/temperature
        return result
    result=raw.clone()
    result[...,8:20]=torch.where(gate,raw[...,8:20]/spec['temperature'],raw[...,8:20])
    return result
