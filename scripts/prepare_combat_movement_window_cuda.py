"""Seal a fixed multi-frame movement smoke using an existing TRAIN context."""
import argparse
import copy
import json
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha
from ppo_combat import torch,training_devices
from ppo_recurrent import prepare_context,sequence
from combat_joint_aim_policy import JointAim,base_model,probabilities,events
from combat_precision_head import selected_means
from combat_movement_probe import VERSION,WINDOW_VERSION,shifts,validate
from migrate_combat_navigation_cuda import branch
from fork_conditional_weapon_exploration_cuda import equal_tree


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','protocol','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();h=sha(path);assert expected is None or h==expected,str(path)
        pins[str(path)]=h;return path
    protocol=read(pin(a.protocol));assert protocol['version']=='combat_movement_window_smoke_v1'
    seal=read(pin(a.prepared/'complete.json'));old=read(pin(a.prepared/'report.json',seal['report_sha256']))
    assert old['device']=='cuda' and old['context']['split']=='train' and old['context']['context_id']==protocol['context_id']
    for path,h in old['inputs'].items():pin(path,h)
    source=Path(old['source']);model=read(pin(source/'weights.json',protocol['source_weights_sha256']))
    assert not model.get('diagnostic_movement_probe') and not model.get('diagnostic_continuation_probe')
    directory=Path(old['context']['directory'])/'cuda';meta=read(pin(directory/'report.json'))
    assert meta['observation_timing_contract']=='frozen_world_v1'
    rows=list(map(json.loads,pin(directory/'rollout.jsonl',meta['rollout_sha256']).open(encoding='utf-8-sig')))
    histories=list(map(json.loads,pin(directory/'sequence.jsonl',meta['sequence_sha256']).open(encoding='utf-8-sig')))
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    actor=branch(base_model(model,allow_shared=True),'actor').requires_grad_(False)
    joint=JointAim(model,allow_shared=True).requires_grad_(False)
    before=copy.deepcopy(actor.state_dict());joint_before=copy.deepcopy(joint.state_dict())
    prepared=prepare_context(histories,rows,'cuda');frame=old['context']['frame']
    frames=torch.tensor([r['frame'] for r in rows],device='cuda',dtype=torch.long)
    x=torch.tensor([r['features'] for r in rows],device='cuda',dtype=torch.float32)
    cat=lambda key:torch.tensor([r['sample'].get(key,0) for r in rows],device='cuda',dtype=torch.long)
    target,mode,vertical,weapon=(cat(k) for k in ('target','aim_mode','vertical','weapon'))
    final=torch.tensor([r['sample']['latent'] for r in rows],device='cuda')
    attack=torch.tensor([float(r['sample']['attack']) for r in rows],device='cuda')
    std=torch.tensor(model['log_std'],device='cuda')
    original,choice,timing,eligible=events(rows,x,target,mode,vertical,weapon)
    with torch.no_grad():
        raw,error=sequence(actor,prepared,32,True,'actor');assert error<2e-5
        means=selected_means(raw,target,mode,x,original[:,:2],actor.sampled_motion_aim_branch,vertical,actor.motor_branch)
        args=(std,original,final,attack,vertical,weapon,x,target,mode,choice,timing,
              actor.sampled_motion_aim_branch,actor.motor_branch,actor.conditional_fire_branch)
        base=probabilities(joint,raw,*args)
        lp_error=float((base[0]-torch.tensor([r['sample']['log_probability'] for r in rows],device='cuda')).abs().max())
        assert lp_error<.003,lp_error
    a.out.mkdir(parents=True);variants=[]
    for v in protocol['variants']:
        changed=copy.deepcopy(model);d=v['duration']
        header=dict(version=VERSION if d==1 else WINDOW_VERSION,frame=frame,mean_shift=v['mean_shift'])
        if d!=1:header['duration_frames']=d
        changed['diagnostic_movement_probe']=header;validate(changed)
        try:base_model(changed,allow_shared=True)
        except AssertionError:pass
        else:raise AssertionError('Diagnostic window admitted to ordinary training')
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            decoded=branch(base_model(changed,allow_probe=True,allow_shared=True),'actor').requires_grad_(False)
            decoded_joint=JointAim(changed,allow_probe=True,allow_shared=True).requires_grad_(False)
        assert equal_tree(decoded.state_dict(),before) and equal_tree(decoded_joint.state_dict(),joint_before)
        with torch.no_grad():
            replay,_=sequence(decoded,prepared,32,True,'actor');assert torch.equal(replay,raw)
            delta=shifts(changed,frames);scored=probabilities(decoded_joint,replay,*args,movement_shift=delta)
            analytic=(delta*(original[:,:2]-means[:,:2])/std[:2].exp().square()-.5*delta.square()/std[:2].exp().square()).sum(-1)
            error=float(((scored[0]-base[0])-analytic).abs().max());assert error<.001,error
            outside=(frames<frame)|(frames>=frame+d)
            assert torch.equal(scored[0][outside],base[0][outside]) and torch.equal(scored[2],base[2])
            if v['label']=='control':assert torch.equal(scored[0],base[0])
        folder=a.out/'models'/v['label'];folder.mkdir(parents=True);save(folder/'weights.json',changed)
        variants.append(dict(**v,index=len(variants),frame=frame,model=str((folder/'weights.json').resolve()),
                             model_sha256=sha(folder/'weights.json'),gaussian_joint_formula_max_error=error))
    assert len(variants)+1==protocol['native_cases']==protocol['cuda_cases']
    for path,h in pins.items():assert sha(path)==h,path
    report=dict(state='complete',version='combat_movement_probe_preparation_v1',device='cuda',diagnostic_only=True,
                context=old['context'],source=str(source),source_weights_sha256=protocol['source_weights_sha256'],
                baseline=old['baseline'],registry=old['registry'],protocol=str(a.protocol.resolve()),protocol_sha256=sha(a.protocol),
                variants=variants,inputs=pins,neural_parameters_exact=True,zero_control_exact=True,
                ordinary_outside_window_exact=True,source_lp_max_error=lp_error,window_smoke=True,
                training_performed=False,validation_used=False,confirmation_used=False,promotion=False)
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(dict(variants=len(variants),device='cuda',source_lp_max_error=lp_error))


if __name__=='__main__':main()
