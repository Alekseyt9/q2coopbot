"""CUDA ridge of F/S residuals in the fitted yaw's command frame; 34 params."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from ppo_combat import torch,training_devices
from migrate_combat_navigation_cuda import branch
from combat_joint_aim_policy import base_model
from train_combat_route_world_intent import blank,metrics
from train_combat_route_yaw_ridge import load_data
from train_combat_memory_route_ppo import admit
from ppo_recurrent import prepare_context,sequence
from combat_memory_route_movement import inputs
from combat_checkpoint_scope import require_checkpoint_scope
from evaluate_registered_combat_models import read,save,sha

SCOPE='combat_route_movement_given_yaw_ridge_v1'
RIDGE=.001


def frozen(model):
    result=copy.deepcopy(model);h=result['memory_route_movement']['layers'][-1]
    h['weight'][:2]=[[0.]*16,[0.]*16];h['bias'][:2]=[0.,0.]
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True,type=Path);ap.add_argument('--data',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path);ap.add_argument('--resume',type=Path)
    a=ap.parse_args();assert not a.out.exists();training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    model=read(a.model);model_hash=sha(a.model);body=blank(model);data=read(a.data)
    assert data['version']=='combat_observed_static_route_heading_v1' and data['state']=='passed' and data['ppo_eligible'] is False
    for p,h in data['inputs'].items():assert sha(p)==h
    sources=[p for p in data['inputs'] if Path(p).name=='weights.json'];assert len(sources)==1
    source=read(sources[0]);source['memory_route_movement']['version']='combat_memory_route_movement_yaw_v2'
    source['memory_route_movement']['layers'][-1]['weight'].append([0.]*16);source['memory_route_movement']['layers'][-1]['bias'].append(0.)
    assert blank(source)==body
    actor=branch(base_model(body,allow_shared=True),'actor').requires_grad_(False)
    train,diagnostic=(load_data(data,s,actor) for s in ('train','diagnostic'))
    assert not({tuple(w) for w in train['worlds']}&{tuple(w) for w in diagnostic['worlds']})
    h=model['memory_route_movement']['layers'][-1]
    before_theta=torch.tensor([r+[b] for r,b in zip(h['weight'],h['bias'])],device='cuda',dtype=torch.float64)
    yaw=train['base3'][:,2]+train['x']@before_theta[2]
    difference=train['turn']-torch.pi*yaw.tanh()
    desired=torch.stack((.5*difference.cos(),-.5*difference.sin()),-1)
    target=torch.atanh(desired)-train['base3'][:,:2]
    gram=torch.zeros(17,17,device='cuda',dtype=torch.float64);rhs=torch.zeros(17,2,device='cuda',dtype=torch.float64)
    anchor=before_theta[:2].T.clone();consumed=[];worlds=[];rows=0
    identity=hashlib.sha256(json.dumps(frozen(model),sort_keys=True).encode()).hexdigest()
    if a.resume:
        cp=torch.load(a.resume,map_location='cuda',weights_only=True);require_checkpoint_scope(cp,SCOPE)
        assert cp['version']=='combat_movement_given_yaw_ridge_checkpoint_v1' and cp['weights_sha256']==model_hash and cp['frozen_identity']==identity and cp['ridge']==RIDGE
        assert not(set(cp['consumed'])&set(train['tokens'])) and not({tuple(w) for w in cp['worlds']}&{tuple(w) for w in train['worlds']}),'Data already consumed'
        assert not({tuple(w) for w in cp['worlds']}&{tuple(w) for w in diagnostic['worlds']})
        gram,rhs,anchor,consumed,worlds,rows=(cp[k] for k in ('gram','rhs','anchor','consumed','worlds','rows'))
    gram=gram+train['x'].T@train['x'];rhs=rhs+train['x'].T@target
    solve=lambda g,b:torch.linalg.solve(g+RIDGE*torch.eye(17,device='cuda',dtype=torch.float64),b+RIDGE*anchor)
    fitted=solve(gram,rhs);assert bool(torch.isfinite(fitted).all()) and float(fitted.abs().max())<1e4
    theta=before_theta.clone();theta[:2]=fitted.T
    before={s:metrics(d,before_theta) for s,d in [('train',train),('diagnostic',diagnostic)]}
    after={s:metrics(d,theta) for s,d in [('train',train),('diagnostic',diagnostic)]}
    if not a.resume:
        n=len(train['x'])//2;x1,x2=train['x'][:n],train['x'][n:];y1,y2=target[:n],target[n:]
        torch.testing.assert_close(fitted,solve(x1.T@x1+x2.T@x2,x1.T@y1+x2.T@y2),atol=1e-8,rtol=1e-8)
    result=copy.deepcopy(model);last=result['memory_route_movement']['layers'][-1]
    last['weight'][:2]=theta[:2,:16].float().cpu().tolist();last['bias'][:2]=theta[:2,16].float().cpu().tolist()
    assert frozen(result)==frozen(model)
    original_actor=branch(base_model(model,allow_shared=True),'actor').requires_grad_(False)
    changed=branch(base_model(result,allow_shared=True),'actor').requires_grad_(False)
    for e in data['episodes']:
        rr,hh,_,_=admit(Path(e['source_cuda']),e['source_model_sha256'],[]);prepared=prepare_context(hh,rr,'cuda')
        with torch.no_grad():
            old,_=sequence(original_actor,prepared);new,_=sequence(changed,prepared)
            x=torch.tensor([r['features'] for r in rr],device='cuda');_,active=inputs(x,old)
            assert torch.equal(old[~active],new[~active]) and torch.equal(old[:,2:],new[:,2:])
    assert all(sha(p)==h for p,h in data['inputs'].items())
    a.out.mkdir(parents=True,exist_ok=False);save(a.out/'weights.json',result)
    cp=dict(version='combat_movement_given_yaw_ridge_checkpoint_v1',checkpoint_training_scope=SCOPE,
        weights_sha256=sha(a.out/'weights.json'),frozen_identity=identity,ridge=RIDGE,gram=gram,rhs=rhs,anchor=anchor,
        consumed=[*consumed,*train['tokens']],worlds=[*worlds,*train['worlds']],rows=rows+len(train['x']))
    torch.save(cp,a.out/'checkpoint.pt');loaded=torch.load(a.out/'checkpoint.pt',map_location='cuda',weights_only=True)
    assert torch.equal(fitted,solve(loaded['gram'],loaded['rhs']))
    try:require_checkpoint_scope(loaded)
    except ValueError:pass
    else:raise AssertionError('Conditional ridge admitted to PPO')
    save(a.out/'report.json',dict(version=SCOPE,state='passed',device=torch.cuda.get_device_name(),training_performed=True,
        trainable_parameters=34,solver='cuda_float64_ridge_conditioned_on_frozen_yaw',ridge=RIDGE,target_speed=.5,
        before=before,after=after,train_rows=len(train['x']),diagnostic_rows=len(diagnostic['x']),
        camera_and_other_fields_exact=True,inactive_and_nonmovement_raw_exact=True,checkpoint_readback_exact=True,
        inputs={str(a.model.resolve()):model_hash,str(a.data.resolve()):sha(a.data)},
        weights_sha256=sha(a.out/'weights.json'),checkpoint_sha256=sha(a.out/'checkpoint.pt'),promotion=False,
        scope='Geometric world-intent projected into fitted mean-camera frame, not native expert or PPO; no closed-loop quality claim.'))
    print(json.dumps(dict(before=before,after=after)),flush=True)


if __name__=='__main__':main()
