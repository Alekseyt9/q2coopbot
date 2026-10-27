"""Causal candidate geometry and requested-motion contrasts, all math on CUDA.

Only the existing pre-gate context187 is accepted. Requested movement is an
action-space proxy, not a forecast of actual motion or future enemy state.
"""
import math

from combat_action_effect import inputs as motor_inputs
from combat_action_geometry import CENTERED_VERSION, features_for_version
from ppo_combat import torch

VERSION='combat_teacher_physical_context_v1'
GEOMETRY_NAMES=('inverse_distance','source_relative_angular_speed')+tuple(
    prefix+name for prefix in ('source_','anchor_minus_source_') for name in (
        'yaw_error_over_halfwidth','pitch_error_over_halfheight','yaw_outside_pi',
        'pitch_outside_pi','closest_bbox_distance','ray_slab_margin','ray_bbox_soft_intersection'))
MOVEMENT_NAMES=('source_requested_radial','anchor_requested_radial',
    'source_requested_transverse_abs','anchor_requested_transverse_abs',
    'requested_vector_change','own_observed_radial_velocity',
    'target_observed_radial_velocity','target_velocity_known')
NAMES=GEOMETRY_NAMES+MOVEMENT_NAMES


def candidate_contexts(context):
    assert context.is_cuda and context.ndim==2 and context.shape[1]==187
    assert bool(torch.isfinite(context).all())
    source=context[:,:185].clone()
    anchor=source.clone()
    # Matches the wide transform's bounded atanh/tanh physical angles.
    anchor[:,164:166]=context[:,185:187].clamp(-1+1e-6,1-1e-6)
    pitch=torch.atan2(context[:,8],context[:,9])*180/math.pi
    for value in (source,anchor):
        # Same absolute pitch limit as policy.Command. Quantization, movement
        # displacement and future recoil are deliberately not forecast here.
        value[:,165]=((pitch+180*value[:,165]).clamp(-89,89)-pitch)/180
    return source,anchor


def features(context):
    source,anchor=candidate_contexts(context)
    g0,g1=(features_for_version(value,CENTERED_VERSION) for value in (source,anchor))
    assert bool((g0[:,0]==1).all()) and bool((g1[:,0]==1).all()), 'Unavailable current observed bbox'
    geometric=torch.cat((g0[:,1:3],g0[:,22:29],g1[:,22:29]-g0[:,22:29]),-1)
    timing=torch.zeros((len(context),2),device='cuda',dtype=context.dtype)
    m0,m1=(motor_inputs(value,timing) for value in (source,anchor))
    position=context[:,101:103]
    unit=position/position.norm(dim=-1,keepdim=True).clamp_min(1e-9)
    tangent=torch.stack((-unit[:,1],unit[:,0]),-1)
    radial=lambda v:(v*unit).sum(-1)
    transverse=lambda v:(v*tangent).sum(-1).abs()
    wish0,wish1=m0[:,188:190],m1[:,188:190]
    known=(context[:,104]>0).to(context.dtype)
    movement=torch.stack((radial(wish0),radial(wish1),transverse(wish0),transverse(wish1),
        (wish1-wish0).norm(dim=-1),radial(m0[:,185:187]),
        radial(context[:,105:107])*known,known),-1)
    output=torch.cat((geometric,movement),-1)
    assert output.shape==(len(context),24) and len(GEOMETRY_NAMES)==16 and len(NAMES)==24
    assert bool(torch.isfinite(output).all())
    return output
