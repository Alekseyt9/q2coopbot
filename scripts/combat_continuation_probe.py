"""Explicit diagnostic future-policy RNG fork; metadata and CUDA receipts only."""
import math
from train_combat_bc import torch

VERSION='combat_policy_continuation_rng_fork_v1'
POST_WINDOW_VERSION='combat_post_movement_window_rng_fork_v2'


def validate(model):
    p=model.get('diagnostic_continuation_probe')
    if p is None:return
    assert set(p)=={'version','fork_frame','seed'} and p['version'] in (VERSION,POST_WINDOW_VERSION)
    assert type(p['fork_frame']) is int and 1<p['fork_frame']<=2**31-1
    assert type(p['seed']) is int and 0<=p['seed']<=2**31-1
    from combat_body_features import VERSIONS
    wide=model.get('wide_aim')
    legacy=wide is None and model['feature_version'] in ('combat_features_v10','combat_features_v12_projectile_types_neutral') and model['action_distribution']=='combat_joint_aim_candidates_v1'
    current=wide is not None and wide['enabled'] and model['feature_version'] in VERSIONS and model['action_distribution']=='combat_joint_aim_wide_v2'
    post_window=p['version']==POST_WINDOW_VERSION and wide is None and model['feature_version']=='combat_features_v13_memory_static_routes' and model['action_distribution']=='combat_joint_aim_candidates_v1'
    assert (p['version']==VERSION and (legacy or current)) or post_window,'Unsupported continuation policy schema'
    assert model.get('aim_candidate_head') and not model.get('deterministic')
    assert not any(model.get(k) for k in ('diagnostic_joint_aim_probe','execution_mode','geometry_aim_ranker','geometry_aim_exploration'))
    m=model.get('diagnostic_movement_probe')
    if post_window:
        from combat_movement_probe import validate as movement_validate
        movement_validate(model)
        assert m is not None and m['version']=='combat_window_movement_mean_shift_v1'
        assert m['frame']+m['duration_frames']==p['fork_frame'] and p['fork_frame']<=2**31-1
        assert not any(model.get(k) for k in ('diagnostic_weapon_probe','diagnostic_anchor_probe','diagnostic_wide_gate_probe'))
        return
    assert m is None or m['frame']+1==p['fork_frame']
    if current:
        assert not any(model.get(k) for k in ('diagnostic_movement_probe','diagnostic_weapon_probe','diagnostic_anchor_probe'))
    gate=model.get('diagnostic_wide_gate_probe')
    assert gate is None or (current and gate['frame']+1==p['fork_frame'])


def effective_seed(model,row):
    p=model.get('diagnostic_continuation_probe')
    if p and row['frame']>=p['fork_frame']:return p['seed']
    return row['sample']['sampling_seed']


def verify_draw_receipts(model,rows,raw,source,final,features,target,mode,vertical,weapon,actor):
    validate(model);p=model['diagnostic_continuation_probe'];ids=[];receipts=[]
    for i,row in enumerate(rows):
        receipt=row['sample'].get('continuation_probe')
        assert row['sample']['sampling_seed']==row['seed'],'Initial episode seed was overwritten'
        if row['frame']<p['fork_frame']:
            assert receipt is None,'Continuation receipt in prefix'
            continue
        assert receipt is not None and set(receipt)=={'version','fork_frame','seed','reset','gaussian_noise','fire_uniform'}
        assert receipt['version']==p['version'] and receipt['fork_frame']==p['fork_frame'] and receipt['seed']==p['seed']
        assert type(receipt['reset']) is bool and receipt['reset']==(row['frame']==p['fork_frame'])
        assert len(receipt['gaussian_noise'])==4 and all(math.isfinite(v) for v in receipt['gaussian_noise'])
        assert math.isfinite(receipt['fire_uniform']) and 0<=receipt['fire_uniform']<1
        ids.append(i);receipts.append(receipt)
    if not ids:
        assert max(row['frame'] for row in rows)<p['fork_frame'],'Missing continuation receipts'
        return dict(observed=False,rows=0,gaussian_max_error=0.,fire_verified=True)
    assert sum(r['reset'] for r in receipts)==1,'Exact fork frame missing or duplicated'
    from combat_precision_head import selected_means
    from combat_conditional_fire import logits as fire_logits
    idx=torch.tensor(ids,device='cuda',dtype=torch.long)
    means=selected_means(raw,target,mode,features,source[:,:2],actor.sampled_motion_aim_branch,vertical,actor.motor_branch)
    if model.get('diagnostic_movement_probe'):
        from combat_movement_probe import shifts
        frames=torch.tensor([r['frame'] for r in rows],device='cuda',dtype=torch.long)
        means=torch.cat((means[:,:2]+shifts(model,frames,means.dtype),means[:,2:]),-1)
    noise=torch.tensor([r['gaussian_noise'] for r in receipts],device='cuda',dtype=means.dtype)
    error=float((means[idx]+noise-source[idx]).abs().max());assert error<.001,error
    logits=fire_logits(actor.conditional_fire_branch,features,raw,target,mode,final,vertical,weapon)
    uniform=torch.tensor([r['fire_uniform'] for r in receipts],device='cuda',dtype=logits.dtype)
    actual=torch.tensor([bool(rows[i]['sample']['attack']) for i in ids],device='cuda')
    assert torch.equal(uniform<logits[idx].sigmoid(),actual),'Continuation fire draw differs'
    return dict(observed=True,rows=len(ids),gaussian_max_error=error,fire_verified=True)
