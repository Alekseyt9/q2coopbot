"""Causal CUDA features for candidate MG aim relative to an observed bbox.

Input185 = existing fire181 + observed recoil4 (known,pitch/32,yaw/32,roll/32).
No native outcomes, trace queries, next observations or hidden weapon state.
Muzzle v1/v2 retains the historical right-hand approximation; v3 uses native
CENTER_HANDED (ConnectRequest hand=2). All versions use the CURRENT observed
pose/recoil and do not predict next recoil, movement or eye height.
"""
import math
from train_combat_bc import torch

VERSION='combat_observed_action_geometry_v1'
SMOOTH_VERSION='combat_observed_action_geometry_v2'
CENTERED_VERSION='combat_observed_action_geometry_v3'
ORIGIN_NAMES=('yaw_sin','yaw_cos','pitch_sin','pitch_cos','yaw_error_pi','pitch_error_pi',
    'yaw_error_over_halfwidth','pitch_error_over_halfheight','yaw_outside_pi','pitch_outside_pi',
    'closest_bbox_distance','ray_slab_margin','ray_bbox_intersection')
NAMES=('known','inverse_distance','relative_transverse_angular_speed')+tuple(
    f'{origin}_{name}' for origin in ('eye','muzzle') for name in ORIGIN_NAMES)
WIDTH=len(NAMES)


def names_for_version(version):
    assert version in (VERSION,SMOOTH_VERSION,CENTERED_VERSION)
    return tuple(n.replace('ray_bbox_intersection','ray_bbox_soft_intersection') for n in NAMES) if version!=VERSION else NAMES


def features_for_version(x,version):
    assert version in (VERSION,SMOOTH_VERSION,CENTERED_VERSION)
    return features(x,smooth=version!=VERSION,centered=version==CENTERED_VERSION)


def features(x,smooth=False,centered=False):
    assert x.is_cuda and x.ndim==2 and x.shape[1]==185 and bool(torch.isfinite(x).all())
    e,t=x[:,99:111],x[:,111:151]
    valid=(x[:,151]==0)&(e[:,0]>0)&(t[:,36]>0)
    yaw=torch.atan2(torch.where(valid,x[:,6],0.),torch.where(valid,x[:,7],1.))
    pitch=torch.atan2(torch.where(valid,x[:,8],0.),torch.where(valid,x[:,9],1.))
    local=torch.where(valid[:,None],e[:,2:5]*512,torch.tensor([256.,0.,0.],device='cuda'))
    cy,sy=yaw.cos(),yaw.sin()
    position=torch.stack((cy*local[:,0]-sy*local[:,1],sy*local[:,0]+cy*local[:,1],local[:,2]),-1)
    hw=torch.where(valid,t[:,37]*64,16.).clamp_min(1.)
    bottom=torch.where(valid,t[:,38]*64,-24.)
    top=torch.where(valid,t[:,39]*64,32.)
    height=torch.where(top-bottom<16,(top+bottom)/2,torch.minimum(torch.maximum(torch.full_like(top,22.),bottom+8),top-8))
    eye=torch.where(x[:,3]>0,torch.full_like(top,-2.),torch.full_like(top,22.))
    kick=torch.where((x[:,181]>0)[:,None],x[:,182:185]*32*math.pi/180,0.)
    aim_yaw=yaw+x[:,164]*math.pi+kick[:,1]
    aim_pitch=pitch+x[:,165]*math.pi+kick[:,0]
    # Same right-vector convention as Quake AngleVectors (right at yaw0 = -Y).
    roll=kick[:,2]
    right=torch.stack((-roll.sin()*aim_pitch.sin()*aim_yaw.cos()+roll.cos()*aim_yaw.sin(),
        -roll.sin()*aim_pitch.sin()*aim_yaw.sin()-roll.cos()*aim_yaw.cos(),-roll.sin()*aim_pitch.cos()),-1)
    zero=torch.zeros_like(eye)
    eye_origin=torch.stack((zero,zero,eye),-1)
    # ConnectRequest advertises hand=2: P_ProjectSource removes the lateral
    # offset entirely. Preserve historical v1/v2 geometry under their versions.
    muzzle_origin=torch.stack((zero,zero,eye-8),-1)+(0 if centered else 8)*right
    direction=torch.stack((aim_pitch.cos()*aim_yaw.cos(),aim_pitch.cos()*aim_yaw.sin(),-aim_pitch.sin()),-1)
    enemy_world=torch.stack((cy*e[:,6]-sy*e[:,7],sy*e[:,6]+cy*e[:,7],e[:,8]),-1)*400
    velocity=(enemy_world-x[:,10:13]*400)*(e[:,5]>0)[:,None]
    transverse=torch.linalg.vector_norm(torch.linalg.cross(direction,velocity),dim=-1)/position.norm(dim=-1).clamp_min(1)
    common=torch.stack((valid.to(x.dtype),1/(1+position.norm(dim=-1)/64),transverse.clamp(0,4)/4),-1)
    wrap=lambda a:torch.atan2(a.sin(),a.cos())
    result=[common]
    for origin in (eye_origin,muzzle_origin):
        relative=position-origin
        distance=relative[:,:2].norm(dim=-1).clamp_min(1.)
        target_yaw=torch.atan2(relative[:,1],relative[:,0])
        target_pitch=-torch.atan2(relative[:,2]+height,distance)
        dy,dp=wrap(target_yaw-aim_yaw),wrap(target_pitch-aim_pitch)
        corners=torch.stack([torch.atan2(relative[:,1]+j*hw,relative[:,0]+i*hw) for i in (-1,1) for j in (-1,1)],-1)
        halfwidth=wrap(corners-target_yaw[:,None]).abs().amax(-1).clamp_min(.001)
        pitch_top=-torch.atan2(relative[:,2]+top,distance)
        pitch_bottom=-torch.atan2(relative[:,2]+bottom,distance)
        halfheight=((pitch_bottom-pitch_top).abs()/2).clamp_min(.001)
        pitch_outside=(pitch_top-aim_pitch).clamp_min(0)+(aim_pitch-pitch_bottom).clamp_min(0)
        lower=torch.stack((relative[:,0]-hw,relative[:,1]-hw,relative[:,2]+bottom),-1)
        upper=torch.stack((relative[:,0]+hw,relative[:,1]+hw,relative[:,2]+top),-1)
        safe=torch.where(direction.abs()<1e-6,torch.where(direction<0,-1e-6,1e-6),direction)
        a,b=lower/safe,upper/safe
        near=torch.minimum(a,b).amax(-1).clamp_min(0);far=torch.maximum(a,b).amin(-1)
        gap=far-near
        center=(lower+upper)/2;half=(upper-lower)/2
        closest=(center*direction).sum(-1).clamp_min(0)[:,None]*direction
        outside=(closest-center).abs().sub(half).clamp_min(0).norm(dim=-1)/half.amax(-1).clamp_min(1)
        result.append(torch.stack((dy.sin(),dy.cos(),dp.sin(),dp.cos(),dy/math.pi,dp/math.pi,
            (dy.abs()/halfwidth).clamp_max(8)/8,(dp.abs()/halfheight).clamp_max(8)/8,
            (dy.abs()-halfwidth).clamp_min(0)/math.pi,pitch_outside/math.pi,
            outside.clamp_max(4)/4,(gap/distance).clamp(-4,4)/4,((gap+1)/2).clamp(0,1) if smooth else (gap>=0).to(x.dtype)),-1))
    output=torch.cat(result,-1)*valid[:,None]
    assert output.shape==(len(x),WIDTH) and WIDTH==29 and bool(torch.isfinite(output).all())
    return output
