"""Shared learned spatial fine/mode residual, independent of base encoder."""
import copy
from train_combat_bc import torch,nn
from ppo_combat import network,layers
from combat_precision_head import validate_precision_model

VERSION='combat_shared_spatial_fine_aim_v1'
COARSE_VERSION='combat_shared_spatial_coarse_fine_aim_v2'
INPUT_WIDTH=119

def validate(model):
    validate_precision_model(model)
    spec=model.get('spatial_aim');assert spec and spec['version'] in (VERSION,COARSE_VERSION) and len(spec['layers'])==3
    width=INPUT_WIDTH
    for i,l in enumerate(spec['layers']):
        assert len(l['bias'])==len(l['weight']) and 1<=len(l['bias'])<=256
        assert all(len(row)==width for row in l['weight'])
        assert all(abs(v)<=1e4 for row,b in zip(l['weight'],l['bias']) for v in row+[b])
        width=len(l['bias'])
    assert width==(6 if spec['version']==COARSE_VERSION else 4)

def inputs(features,raw):
    assert features.shape[:-1]==raw.shape[:-1] and features.shape[-1] in (854,881,1121) and raw.shape[-1]==81
    shape=features.shape[:-1];x=features[...,:854].reshape(-1,854);r=raw.reshape(-1,81);n=len(x)
    e=x[:,73:169].reshape(n,8,12);typed=x[:,466:786].reshape(n,8,40);bbox=x[:,426:466].reshape(n,8,5)
    common=torch.cat((x[:,:20],x[:,810:845],r[:,:2].tanh()),-1)[:,None,:].expand(-1,8,-1)
    p=e[:,:,2:5]*512;own=torch.stack((x[:,10]*x[:,7]+x[:,11]*x[:,6],-x[:,10]*x[:,6]+x[:,11]*x[:,7],x[:,12]),-1)
    velocity=(e[:,:,6:9]-own[:,None,:])*400;distance=(e[:,:,1]*512).clamp_min(1)
    closing=e[:,:,5]*(p*velocity).sum(-1)/p.norm(dim=-1).clamp_min(1)/400
    angular=e[:,:,5]*(p[:,:,0]*velocity[:,:,1]-p[:,:,1]*velocity[:,:,0])/p[:,:,:2].square().sum(-1).clamp_min(1)/4
    derived=torch.stack((1/(1+distance/64),torch.atan2(typed[:,:,37]*64,distance)/torch.pi,torch.atan2((typed[:,:,39]-typed[:,:,38])*32,distance)/torch.pi,closing.clamp(-4,4),angular.clamp(-4,4)),-1)
    result=torch.cat((common,e,typed,bbox,derived),-1)
    assert result.shape[-1]==INPUT_WIDTH and bool(torch.isfinite(result).all())
    return result.reshape(*shape,8,INPUT_WIDTH)

def apply(branch,features,raw):
    correction=branch(inputs(features,raw));result=raw.clone()
    result[...,47:63]=result[...,47:63]+correction[...,:2].reshape(*raw.shape[:-1],16)
    result[...,65:81]=result[...,65:81]+correction[...,2:4].reshape(*raw.shape[:-1],16)
    if correction.shape[-1]==6:
        result[...,29:45]=result[...,29:45]+correction[...,4:6].reshape(*raw.shape[:-1],16)
    return result

def initialize(model):
    assert not model.get('spatial_aim');result=copy.deepcopy(model);validate_precision_model(result)
    branch=nn.Sequential(nn.Linear(INPUT_WIDTH,64,device='cuda'),nn.ReLU(),nn.Linear(64,32,device='cuda'),nn.ReLU(),nn.Linear(32,4,device='cuda'))
    with torch.no_grad():branch[-1].weight.zero_();branch[-1].bias.zero_()
    result['spatial_aim']=dict(version=VERSION,layers=layers(branch));validate(result)
    return result

def migrate_coarse(model):
    """Append zero coarse residuals; existing actor/value/fine/mode preserved."""
    result=copy.deepcopy(model)
    if not result.get('spatial_aim'):result=initialize(result)
    validate(result);spec=result['spatial_aim']
    if spec['version']==COARSE_VERSION:return result
    last=spec['layers'][-1];width=len(last['weight'][0])
    last['weight'].extend([[0.]*width,[0.]*width]);last['bias'].extend([0.,0.])
    spec['version']=COARSE_VERSION;validate(result);return result

class SpatialActor(nn.Module):
    def __init__(self,base,spec,model=None):
        super().__init__();self.base=base;self.branch=network(spec['layers'])
        from combat_shared_target import build
        self.target_branch=build(model) if model is not None else None
    def forward(self,x,*args,**kwargs):
        output=self.base(x,*args,**kwargs)
        raw=output[0] if isinstance(output,tuple) else output
        result=apply(self.branch,x,raw)
        if self.target_branch is not None:
            from combat_shared_target import apply as apply_target
            result=apply_target(self.target_branch,x,result)
        return (result,*output[1:]) if isinstance(output,tuple) else result
    def single(self,x):return self(x[:,None,:])[0][:,0,:]
    def export(self):return self.base.export()
    def __iter__(self):return iter(self.base)

def wrap(base,model):
    if not model.get('spatial_aim'):return base
    validate(model);return SpatialActor(base,model['spatial_aim'],model)

def export(model,actor):
    if isinstance(actor,SpatialActor):
        model['spatial_aim']={**model['spatial_aim'],'layers':layers(actor.branch)}
        from combat_shared_target import export as export_target
        export_target(model,actor.target_branch)
