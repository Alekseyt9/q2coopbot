"""CUDA previous-executed-command feedback for movement means only."""
import math
import torch

VERSION='combat_previous_command_inertia_v1'
EPS=2.**-12

def validate(header):
    assert isinstance(header,dict) and set(header)=={'version','rho'} and header['version']==VERSION
    rho=header['rho'];assert not isinstance(rho,bool) and isinstance(rho,(int,float)) and math.isfinite(rho) and 0<=rho<=.95
    return float(rho)

def apply(features,raw,header):
    rho=validate(header)
    assert features.device.type==raw.device.type=='cuda'
    assert features.shape[:-1]==raw.shape[:-1] and features.shape[-1]>=15 and raw.shape[-1]>=2
    assert bool(torch.isfinite(features[...,13:15]).all()) and bool(torch.isfinite(raw[...,:2]).all())
    if rho==0:return raw
    # Binary-exact clipping and CUDA float64 scalar math avoid unstable
    # inverse-tanh rounding near full forward/side commands.
    previous=features[...,13:15].to(torch.float64).clamp(-1,1)
    desired=raw[...,:2].to(torch.float64).tanh()
    blended=((1-rho)*desired+rho*previous).clamp(-1+EPS,1-EPS)
    result=raw.clone();result[...,:2]=torch.atanh(blended).to(raw.dtype)
    return result

def movement_log_std(source,header):
    rho=validate(header);assert source.device.type=='cuda' and source.shape==(4,)
    assert bool(torch.isfinite(source).all())
    if rho==0:return source
    result=source.clone();result[:2]=result[:2]+.5*math.log1p(-rho*rho)
    return result
