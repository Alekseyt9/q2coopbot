"""Explicit one-frame/window Gaussian mean interventions; CUDA helpers."""
import math
from train_combat_bc import torch

VERSION='combat_one_frame_movement_mean_shift_v1'
WINDOW_VERSION='combat_window_movement_mean_shift_v1'


def duration(probe):
    return probe['duration_frames'] if probe['version']==WINDOW_VERSION else 1


def active(probe,frame):
    return probe['frame']<=frame<probe['frame']+duration(probe)


def validate(model):
    probe=model.get('diagnostic_movement_probe')
    if probe is None:return
    if probe['version']==VERSION:
        assert set(probe)=={'version','frame','mean_shift'}
    else:
        assert probe['version']==WINDOW_VERSION and set(probe)=={'version','frame','mean_shift','duration_frames'}
        assert type(probe['duration_frames']) is int and 2<=probe['duration_frames']<=10
    assert type(probe['frame']) is int and 0<probe['frame']<=2**31-1
    assert probe['frame']+duration(probe)-1<=2**31-1
    assert len(probe['mean_shift'])==2 and all(type(v) in (int,float) and math.isfinite(v) and abs(v)<=2 for v in probe['mean_shift'])
    assert model['feature_version']=='combat_features_v10' and model['action_distribution']=='combat_joint_aim_candidates_v1'
    assert model.get('aim_candidate_head') and not model.get('deterministic')
    assert not any(model.get(k) for k in ('diagnostic_joint_aim_probe','execution_mode','geometry_aim_ranker','geometry_aim_exploration'))


def shifts(model,frames,dtype=torch.float32):
    validate(model)
    assert frames.is_cuda and frames.ndim==1 and frames.dtype==torch.long
    result=torch.zeros((len(frames),2),device='cuda',dtype=dtype)
    probe=model.get('diagnostic_movement_probe')
    if probe is not None:
        delta=torch.tensor(probe['mean_shift'],device='cuda',dtype=dtype)
        mask=(frames>=probe['frame']) & (frames<probe['frame']+duration(probe))
        result=torch.where(mask[:,None],delta[None,:],result)
    return result


def verify_draw_receipts(model,rows,raw,std,source,final,features,target,mode,vertical,weapon,actor):
    """Check actual recorded additive noises and the reused fire draw on CUDA."""
    validate(model); probe=model['diagnostic_movement_probe']
    records=[]; indexes=[]
    for i,row in enumerate(rows):
        receipt=row['sample'].get('movement_probe')
        if not active(probe,row['frame']):
            assert receipt is None,'Movement receipt outside declared frame'
            continue
        assert receipt is not None and set(receipt)=={'version','frame','mean_shift','gaussian_noise','fire_uniform'}
        assert receipt['version']==probe['version'] and receipt['frame']==row['frame'] and receipt['mean_shift']==probe['mean_shift']
        assert len(receipt['gaussian_noise'])==4 and all(math.isfinite(v) for v in receipt['gaussian_noise'])
        assert math.isfinite(receipt['fire_uniform']) and 0<=receipt['fire_uniform']<1
        indexes.append(i); records.append(receipt)
    observed=[rows[i]['frame'] for i in indexes]
    assert observed and observed[0]==probe['frame'] and len(observed)==len(set(observed))
    assert len(records)<=duration(probe),'Duplicate intervention frames'
    if probe['version']==VERSION: assert len(records)==1
    from combat_precision_head import selected_means
    from combat_conditional_fire import logits as fire_logits
    idx=torch.tensor(indexes,device='cuda',dtype=torch.long)
    means=selected_means(raw,target,mode,features,source[:,:2],actor.sampled_motion_aim_branch,vertical,actor.motor_branch)
    frames=torch.tensor([r['frame'] for r in rows],device='cuda',dtype=torch.long)
    means=torch.cat((means[:,:2]+shifts(model,frames,means.dtype),means[:,2:]),-1)
    noise=torch.tensor([r['gaussian_noise'] for r in records],device='cuda',dtype=means.dtype)
    error=float((means[idx]+noise-source[idx]).abs().max()); assert error<.001,error
    fire=fire_logits(actor.conditional_fire_branch,features,raw,target,mode,final,vertical,weapon)
    uniform=torch.tensor([r['fire_uniform'] for r in records],device='cuda',dtype=fire.dtype)
    actual=torch.tensor([bool(rows[i]['sample']['attack']) for i in indexes],device='cuda')
    assert torch.equal(uniform<fire[idx].sigmoid(),actual),'Conditional fire differs from recorded random draw'
    return error
