"""CUDA ridge fit of 17 route-yaw parameters; typed sufficient-statistic resume."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from ppo_combat import torch,training_devices
from ppo_recurrent import prepare_context,sequence
from migrate_combat_navigation_cuda import branch
from combat_joint_aim_policy import base_model
from combat_memory_route_movement import VERSION,YAW_VERSION,inputs
from train_combat_memory_route_ppo import admit
from combat_checkpoint_scope import require_checkpoint_scope
from evaluate_registered_combat_models import read,save,sha

SCOPE='combat_observed_route_heading_yaw_ridge_v1'
CHECKPOINT='combat_route_yaw_ridge_checkpoint_v1'
RIDGE=.001


def zeroed(model):
    result=copy.deepcopy(model);head=result['memory_route_movement']
    assert head['version']==YAW_VERSION and len(head['layers'][-1]['bias'])==3
    head['layers'][-1]['weight'][2]=[0.]*16;head['layers'][-1]['bias'][2]=0.
    return result


def identity(model):
    return hashlib.sha256(json.dumps(zeroed(model),sort_keys=True,allow_nan=False).encode()).hexdigest()


def load_data(report,split,actor,admit_function=admit):
    matrices=[];targets=[];base=[];base3=[];turns=[];worlds=[];tokens=[]
    for e in report['episodes']:
        if e['split']!=split:continue
        assert sha(e['pack'])==e['pack_sha256'] and sha(e['labels'])==e['labels_sha256']
        rows,history,token,_=admit_function(Path(e['source_cuda']),e['source_model_sha256'],[])
        assert token==e['native_rollout_sha256']
        pack=np.load(e['pack'],allow_pickle=False);idx=torch.tensor(pack['indices'],device='cuda',dtype=torch.long)
        labels=read(e['labels']);assert len(idx)==len(labels)==e['rows']
        assert [rows[int(i)]['frame'] for i in pack['indices']]==[r['frame'] for r in labels]
        prepared=prepare_context(history,rows,'cuda');captured=[]
        handle=actor.memory_route_branch[-1].register_forward_pre_hook(lambda module,args:captured.append(args[0].detach()))
        with torch.no_grad():raw,context_error=sequence(actor,prepared,32,True,'actor')
        handle.remove();assert len(captured)==1 and context_error<2e-5
        _,locations,inverse,_=prepared;hidden=captured[0][locations[:,0],locations[:,1]][inverse][idx]
        assert hidden.shape==(len(idx),16)
        x=torch.tensor([r['features'] for r in rows],device='cuda');_,active=inputs(x,raw)
        assert bool(active[idx].all())
        target_turn=torch.tensor(pack['turn'],device='cuda',dtype=torch.float64)
        assert bool(torch.isfinite(target_turn).all()) and float(target_turn.abs().max())<=torch.pi
        matrices.append(torch.cat((hidden.double(),torch.ones(len(idx),1,device='cuda',dtype=torch.float64)),-1))
        targets.append(torch.atanh((target_turn/torch.pi).clamp(-.99,.99))-raw[idx,2].double())
        base.append(raw[idx,2].double());base3.append(raw[idx,:3].double());turns.append(target_turn)
        worlds.append([e['map'],e['seed']]);tokens.append(token)
    assert matrices and len(worlds)==len({tuple(w) for w in worlds}) and len(tokens)==len(set(tokens))
    return dict(x=torch.cat(matrices),y=torch.cat(targets),base=torch.cat(base),base3=torch.cat(base3),turn=torch.cat(turns),worlds=worlds,tokens=tokens)


def solve(gram,rhs):
    return torch.linalg.solve(gram+RIDGE*torch.eye(17,device='cuda',dtype=torch.float64),rhs)


def metrics(data,theta):
    latent=data['base']+data['x']@theta
    delta=torch.pi*latent.tanh()-data['turn'];angle=torch.atan2(delta.sin(),delta.cos()).abs()
    return dict(heading_mae_degrees=float(angle.mean()*180/torch.pi),
                mean_absolute_turn_degrees=float((latent.tanh()*180).abs().mean()),
                residual_latent_mse=float(((data['x']@theta)-data['y']).square().mean()))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True,type=Path);ap.add_argument('--data',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path);ap.add_argument('--resume',type=Path)
    a=ap.parse_args();assert not a.out.exists();training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    model=read(a.model);fixed=zeroed(model);frozen_sha=identity(model);model_hash=sha(a.model)
    report=read(a.data);assert report['version']=='combat_observed_static_route_heading_v1' and report['state']=='passed'
    assert report['ppo_eligible'] is False and report['executed_demonstration'] is False
    for p,h in report['inputs'].items():assert sha(p)==h
    sources=[p for p in report['inputs'] if Path(p).name=='weights.json'];assert len(sources)==1
    original=read(sources[0]);legacy=copy.deepcopy(fixed);h=legacy['memory_route_movement'];h['version']=VERSION
    h['layers'][-1]['weight'].pop();h['layers'][-1]['bias'].pop();assert legacy==original,'Frozen source behavior differs'
    actor=branch(base_model(fixed,allow_shared=True),'actor').requires_grad_(False)
    train,diagnostic=(load_data(report,s,actor) for s in ('train','diagnostic'))
    assert not({tuple(w) for w in train['worlds']}&{tuple(w) for w in diagnostic['worlds']})
    layer=model['memory_route_movement']['layers'][-1]
    initial=torch.tensor([*layer['weight'][2],layer['bias'][2]],device='cuda',dtype=torch.float64)
    gram=torch.zeros(17,17,device='cuda',dtype=torch.float64);rhs=torch.zeros(17,device='cuda',dtype=torch.float64)
    consumed=[];worlds=[];rows=0
    if a.resume:
        cp=torch.load(a.resume,map_location='cuda',weights_only=True);require_checkpoint_scope(cp,SCOPE)
        assert cp['version']==CHECKPOINT and cp['weights_sha256']==model_hash and cp['frozen_behavior_sha256']==frozen_sha and cp['ridge']==RIDGE
        gram,rhs=cp['gram'],cp['rhs'];consumed=cp['consumed'];worlds=cp['worlds'];rows=cp['rows']
        assert not(set(consumed)&set(train['tokens'])) and not({tuple(w) for w in worlds}&{tuple(w) for w in train['worlds']}),'Data already consumed'
        assert not({tuple(w) for w in worlds}&{tuple(w) for w in diagnostic['worlds']})
    else:assert bool(initial.eq(0).all()),'Fresh fit requires zero-migrated yaw row'
    before={s:metrics(d,initial) for s,d in [('train',train),('diagnostic',diagnostic)]}
    gram=gram+train['x'].T@train['x'];rhs=rhs+train['x'].T@train['y'];theta=solve(gram,rhs)
    assert bool(torch.isfinite(theta).all()) and float(theta.abs().max())<1e4
    after={s:metrics(d,theta) for s,d in [('train',train),('diagnostic',diagnostic)]}
    # Independent partition/readback verifies additive checkpoint statistics.
    n=len(train['x'])//2;x1,x2=train['x'][:n],train['x'][n:];y1,y2=train['y'][:n],train['y'][n:]
    if not a.resume:
        alternate=solve(x1.T@x1+x2.T@x2,x1.T@y1+x2.T@y2)
        torch.testing.assert_close(theta,alternate,atol=1e-8,rtol=1e-8)
    result=copy.deepcopy(model);head=result['memory_route_movement']['layers'][-1]
    head['weight'][2]=theta[:16].float().cpu().tolist();head['bias'][2]=float(theta[16].float())
    assert zeroed(result)==fixed and identity(result)==frozen_sha
    changed=branch(base_model(result,allow_shared=True),'actor').requires_grad_(False)
    for e in report['episodes']:
        rr,hh,_,_=admit(Path(e['source_cuda']),e['source_model_sha256'],[]);prepared=prepare_context(hh,rr,'cuda')
        with torch.no_grad():
            old,_=sequence(actor,prepared);new,_=sequence(changed,prepared)
            x=torch.tensor([r['features'] for r in rr],device='cuda');_,active=inputs(x,old)
            assert torch.equal(old[~active],new[~active]) and torch.equal(old[:,:2],new[:,:2]) and torch.equal(old[:,3:],new[:,3:])
    assert all(sha(p)==h for p,h in report['inputs'].items())
    a.out.mkdir(parents=True,exist_ok=False);save(a.out/'weights.json',result)
    cp=dict(version=CHECKPOINT,checkpoint_training_scope=SCOPE,weights_sha256=sha(a.out/'weights.json'),
        frozen_behavior_sha256=frozen_sha,ridge=RIDGE,gram=gram,rhs=rhs,rows=rows+len(train['x']),
        worlds=[*worlds,*train['worlds']],consumed=[*consumed,*train['tokens']])
    torch.save(cp,a.out/'checkpoint.pt');restored=torch.load(a.out/'checkpoint.pt',map_location='cuda',weights_only=True)
    assert torch.equal(solve(restored['gram'],restored['rhs']),theta)
    try:require_checkpoint_scope(restored)
    except ValueError:pass
    else:raise AssertionError('Ridge checkpoint admitted to generic PPO')
    save(a.out/'report.json',dict(version=SCOPE,state='passed',device=torch.cuda.get_device_name(),training_performed=True,
        trainable_parameters=17,solver='cuda_float64_ridge',ridge=RIDGE,before=before,after=after,
        rows=cp['rows'],fresh_rows=len(train['x']),diagnostic_rows=len(diagnostic['x']),frozen_other_fields_exact=True,
        full_context_inactive_and_other_raw_outputs_exact=True,checkpoint_readback_exact=True,
        inputs={str(a.model.resolve()):model_hash,str(a.data.resolve()):sha(a.data)},
        weights_sha256=sha(a.out/'weights.json'),checkpoint_sha256=sha(a.out/'checkpoint.pt'),
        append_resume=bool(a.resume),promotion=False,
        scope='Observed static waypoint heading auxiliary, not optimal executed teacher or PPO. Only last yaw row trained; new native quality required.'))
    print(json.dumps(dict(before=before,after=after)),flush=True)


if __name__=='__main__':main()
