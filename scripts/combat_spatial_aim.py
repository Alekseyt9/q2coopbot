"""Shared learned spatial fine/mode residual, independent of base encoder."""
from combat_body_features import VERSIONS as BODY_FEATURE_VERSIONS, WIDTH as BODY_FEATURE_WIDTH, legacy as body_legacy
from combat_projectile_type_features import WIDTH as PROJECTILE_FEATURE_WIDTH
from combat_memory_route_features import WIDTH as ROUTE_FEATURE_WIDTH
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
    assert features.shape[:-1]==raw.shape[:-1] and features.shape[-1] in (854,881,1121,1147,BODY_FEATURE_WIDTH,PROJECTILE_FEATURE_WIDTH,ROUTE_FEATURE_WIDTH) and raw.shape[-1]==81
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
        from combat_weapon_state_residual import build as build_weapon
        self.weapon_state_branch=build_weapon(model) if model is not None else None
        from combat_weapon_exploration import validate as validate_exploration
        if model is not None:validate_exploration(model)
        self.weapon_exploration=model.get('weapon_exploration') if model is not None else None
        from combat_sampled_motion_aim import build as build_motion
        self.sampled_motion_aim_branch=build_motion(model) if model is not None else None
        from combat_motor_residual import build as build_motor
        self.motor_branch=build_motor(model) if model is not None else None
        from combat_conditional_fire import build as build_fire
        self.conditional_fire_branch=build_fire(model) if model is not None else None
        from combat_memory_recovery_runtime import build as build_recovery
        self.memory_recovery_branch=build_recovery(model) if model is not None else None
        from combat_memory_route_movement import build as build_routes
        self.memory_route_branch=build_routes(model) if model is not None else None
        self.movement_inertia=copy.deepcopy(model.get('movement_inertia')) if model is not None else None
        if self.movement_inertia is not None:
            from combat_action_inertia import validate as validate_inertia
            validate_inertia(self.movement_inertia)
            assert model['feature_version'] in BODY_FEATURE_VERSIONS and not model.get('memory_recovery') and not model.get('entity_attention')
            assert not model.get('diagnostic_movement_probe') and not model.get('diagnostic_continuation_probe') and not model.get('execution_mode')
        if self.sampled_motion_aim_branch is not None:
            self.sampled_motion_aim_branch.requires_motor_reference=self.motor_branch is not None
            self.sampled_motion_aim_branch.requires_conditional_fire=self.conditional_fire_branch is not None
    def forward(self,x,*args,**kwargs):
        output=self.base(x,*args,**kwargs)
        raw=output[0] if isinstance(output,tuple) else output
        result=apply(self.branch,x,raw)
        if self.target_branch is not None:
            from combat_shared_target import apply as apply_target
            result=apply_target(self.target_branch,x,result)
        if self.weapon_state_branch is not None:
            from combat_weapon_state_residual import apply as apply_weapon
            result=apply_weapon(self.weapon_state_branch,x,result)
        if self.weapon_exploration is not None:
            from combat_weapon_exploration import apply as apply_exploration
            result=apply_exploration(self.weapon_exploration,x,result)
        if self.motor_branch is not None:
            from combat_motor_residual import apply as apply_motor
            result=apply_motor(self.motor_branch,x,result)
        if self.memory_recovery_branch is not None:
            from combat_memory_recovery_runtime import apply as apply_recovery
            result=apply_recovery(self.memory_recovery_branch,x,result)
        if self.movement_inertia is not None:
            from combat_action_inertia import apply as apply_inertia
            result=apply_inertia(x,result,self.movement_inertia)
        if self.memory_route_branch is not None:
            from combat_memory_route_movement import apply as apply_routes
            result=apply_routes(self.memory_route_branch,x,result)
        return (result,*output[1:]) if isinstance(output,tuple) else result
    def single(self,x):
        output=self(x[:,None,:]);raw=output[0] if isinstance(output,tuple) else output
        return raw[:,0,:]
    def export(self):return self.base.export()
    def __iter__(self):return iter(self.base)

def wrap(base,model):
    from combat_memory_route_movement import validate as validate_routes
    validate_routes(model)
    from combat_memory_recovery_runtime import validate as validate_recovery
    validate_recovery(model)
    from combat_conditional_fire import validate as validate_fire
    validate_fire(model)
    from combat_weapon_state_residual import validate as validate_weapon
    validate_weapon(model)
    from combat_weapon_exploration import validate as validate_exploration
    validate_exploration(model)
    if not model.get('spatial_aim'):return base
    validate(model);return SpatialActor(base,model['spatial_aim'],model)

def export(model,actor):
    if isinstance(actor,SpatialActor):
        model['spatial_aim']={**model['spatial_aim'],'layers':layers(actor.branch)}
        from combat_shared_target import export as export_target
        export_target(model,actor.target_branch)
        from combat_weapon_state_residual import export as export_weapon
        export_weapon(model,actor.weapon_state_branch)
        from combat_sampled_motion_aim import export as export_motion
        export_motion(model,actor.sampled_motion_aim_branch)
        from combat_motor_residual import export as export_motor
        export_motor(model,actor.motor_branch)
        from combat_conditional_fire import export as export_fire
        export_fire(model,actor.conditional_fire_branch)
        from combat_memory_recovery_runtime import export as export_recovery
        export_recovery(model,actor.memory_recovery_branch)
        from combat_memory_route_movement import export as export_routes
        export_routes(model,actor.memory_route_branch)
        if actor.movement_inertia is not None:model['movement_inertia']=copy.deepcopy(actor.movement_inertia)
