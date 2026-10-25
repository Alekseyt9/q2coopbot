"""Fresh CUDA PPO after executed-movement BC; motor+route actor and full critic."""
import argparse
import copy
import json
from pathlib import Path

from ppo_combat import torch, nn, training_devices, advantages, guarded_actor_step
from ppo_recurrent import prepare_context, sequence
from combat_joint_aim_policy import JointAim, events, probabilities
from combat_movement_ppo import MovementScope, SCOPE, PREFIXES
from train_combat_memory_route_ppo import admit, CONFIG_KEYS
from combat_memory_route_movement import inputs
from combat_route_task_objective import CONTINUITY,AUTONOMOUS,collect_evidence,overlay,task_advantages
from evaluate_registered_combat_models import read, save, sha

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--request',type=Path,required=True)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--resume',type=Path)
    ap.add_argument('--task-objective',choices=(AUTONOMOUS,),default=AUTONOMOUS)
    a=ap.parse_args();assert not a.out.exists();training_devices()
    torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    config=read(a.config)
    assert set(config)==CONFIG_KEYS and config['version']==SCOPE
    assert 0<config['gamma']<=1 and 0<=config['lambda']<=1 and 0<config['clip']<=.3
    assert 1<=config['actor_steps']<=100 and 1<=config['value_steps']<=100 and 0<config['target_kl']<=.05
    assert config['entropy']>=0 and config['max_grad_norm']>0
    torch.manual_seed(config['seed'])
    model=read(a.model);model_hash=sha(a.model)
    request=read(a.request)
    assert request['version']=='combat_motor_route_train_request_v1' and request['split']=='train'
    assert request['behavior_sha256']==model_hash and request['data']
    capture=Path(request['capture_root']).resolve()
    capture_spec=read(capture/'spec.json')
    assert capture_spec['version']=='combat_memory_route_candidate_native_v1'
    assert capture_spec['fixed_health_100'] and capture_spec['fresh_seed_offsets']
    assert read(capture/'acceptance.json')['state']=='passed'
    assert read(capture/'technical-parity.json')['candidate_sha256']==model_hash
    parent=read(a.model.parent/'report.json')
    assert parent['state']=='passed' and parent['weights_sha256']==model_hash and parent['training_performed']
    if a.resume:
        assert parent['version']==SCOPE and sha(a.resume)==parent['checkpoint_sha256']
    else:
        assert parent['version']=='combat_movement_demonstration_bc_v1'
        assert parent['optimizer_initialization']=='fresh_bc_not_ppo_resume'
        # The original fresh BC report predates the optional resume counters.
        assert parent['ppo_data_mixed'] is False and parent.get('resumed_steps',0)==0
        # The BC checkpoint is provenance only; never load its optimizer as PPO.
        assert sha(a.model.parent/'checkpoint.pt')==parent['checkpoint_sha256']
    eligible={str((capture/f'case-{v["index"]:02d}'/'cuda').resolve()) for v in capture_spec['variants']
              if v['label']=='zero' and v['model_sha256']==model_hash and v['context']['instance']['split']=='train'}
    assert len(set(request['data']))==len(request['data'])
    assert {str(Path(p).resolve()) for p in request['data']}==eligible, 'Use all own admitted TRAIN captures'
    scope=MovementScope(model,actor_lr=config['actor_lr'],value_lr=config['value_lr'])
    consumed=[]
    if a.resume:
        previous=torch.load(a.resume,map_location='cuda',weights_only=True)
        consumed=scope.restore(previous,model_hash,config,a.task_objective)
    rows,history,tokens,metadata=[],[],[],[]
    for directory in request['data']:
        rr,hh,token,meta=admit(Path(directory),model_hash,[*consumed,*tokens])
        assert not ({r['seed'] for r in rr}&{r['seed'] for r in rows}), 'Training episodes must have distinct seeds'
        rows.extend(rr);history.extend(hh);tokens.append(token);metadata.append(meta)
    assert len({m['reward_config_sha256'] for m in metadata})==1
    assert len({m['combat_execution_contract'] for m in metadata})==1
    prepared=prepare_context(history,rows,'cuda')
    x=torch.tensor([r['features'] for r in rows],device='cuda')
    cat=lambda key:torch.tensor([r['sample'].get(key,0) for r in rows],device='cuda',dtype=torch.long)
    target,mode,vertical,weapon=(cat(k) for k in ('target','aim_mode','vertical','weapon'))
    final=torch.tensor([r['sample']['latent'] for r in rows],device='cuda')
    attack=torch.tensor([float(r['sample']['attack']) for r in rows],device='cuda')
    original,choice,timing,_=events(rows,x,target,mode,vertical,weapon)
    joint=JointAim(model,allow_shared=True).requires_grad_(False)
    score_inputs=None
    if joint.geometry_head:
        joint=joint.double()
        score_inputs=(x.double(),original.double(),timing.double())
    frames=torch.tensor([r['frame'] for r in rows],device='cuda',dtype=torch.long)
    old=torch.tensor([r['sample']['log_probability'] for r in rows],device='cuda')
    def likelihood(raw):
        return probabilities(joint,raw,scope.std,original,final,attack,vertical,weapon,x,target,mode,choice,timing,
            scope.actor.sampled_motion_aim_branch,scope.actor.motor_branch,scope.actor.conditional_fire_branch,
            frames=frames,score_inputs=score_inputs)[:2]
    with torch.no_grad():
        raw,actor_memory=sequence(scope.actor,prepared,32,True,'actor')
        predicted,value_memory=sequence(scope.value,prepared,32,True,'value')
        lp,_=likelihood(raw)
        log_error=float((lp-old).abs().max())
        value_error=float((predicted[:,0]-torch.tensor([r['sample']['value'] for r in rows],device='cuda')).abs().max())
        assert log_error<.003 and value_error<1e-4 and max(actor_memory,value_memory)<2e-5,(log_error,value_error,actor_memory,value_memory)
        _,route_active=inputs(x,raw)
        active=torch.ones(len(rows),device='cuda',dtype=torch.bool) # Motor contributes at every admitted state.
    evidence,evidence_pins=collect_evidence(capture,rows,model_hash=model_hash)
    task_rows,task_boundaries=overlay(rows,evidence,a.task_objective)
    if a.task_objective==AUTONOMOUS:assert task_boundaries, 'No verified task boundary in requested objective batch'
    ad,ret=task_advantages(task_rows,config['gamma'],config['lambda'],advantages)
    ad=torch.tensor(ad,device='cuda');ret=torch.tensor(ret,device='cuda')
    ad=(ad-ad.mean())/(ad.std(unbiased=False)+1e-8)
    before={k:v.detach().clone() for k,v in scope.actor.state_dict().items() if k.startswith(PREFIXES)}
    def objective():
        current,_=sequence(scope.actor,prepared)
        lp,entropy=likelihood(current);delta=lp-old;ratio=delta.exp()
        loss=-torch.minimum(ratio*ad,ratio.clamp(1-config['clip'],1+config['clip'])*ad).mean()-config['entropy']*entropy.mean()
        # Gate trust region on changed states; inactive states must not dilute KL.
        kl=((ratio-1)-delta)[active].mean()
        return loss,kl
    trials=[]
    for _ in range(config['actor_steps']):
        loss,kl=objective()
        if float(kl.detach())>config['target_kl']:break
        scope.actor_optimizer.zero_grad();loss.backward()
        norm=nn.utils.clip_grad_norm_(scope.actor_parameters,config['max_grad_norm'])
        assert torch.isfinite(norm)
        trial=guarded_actor_step(scope.actor_parameters,scope.actor_optimizer,objective,loss.detach(),config['actor_lr'],config['target_kl'])
        trials.append(trial)
        if not trial['accepted']:break
    critic_losses=[]
    for _ in range(config['value_steps']):
        prediction,_=sequence(scope.value,prepared,32,False,'value')
        loss=(prediction[:,0]-ret).square().mean();assert torch.isfinite(loss)
        scope.value_optimizer.zero_grad();loss.backward()
        norm=nn.utils.clip_grad_norm_(scope.value.parameters(),config['max_grad_norm']);assert torch.isfinite(norm)
        scope.value_optimizer.step();critic_losses.append(float(loss.detach()))
    scope.verify_frozen()
    with torch.no_grad():
        after_raw,_=sequence(scope.actor,prepared)
        assert torch.equal(raw[:,2:],after_raw[:,2:]), 'Non-movement actor outputs changed'
        final_loss,final_kl=objective()
        assert torch.isfinite(final_loss) and float(final_kl)<=config['target_kl']+1e-6
    changed=any(not torch.equal(v,scope.actor.state_dict()[k]) for k,v in before.items())
    result=scope.export()
    assert all(sha(Path(p))==digest for p,digest in evidence_pins.items()), 'Task evidence changed during training'
    a.out.mkdir(parents=True,exist_ok=False)
    save(a.out/'weights.json',result)
    exported_hash=sha(a.out/'weights.json')
    checkpoint=scope.snapshot(exported_hash,config,[*consumed,*tokens],a.task_objective)
    torch.save(checkpoint,a.out/'checkpoint.pt')
    save(a.out/'config.json',config)
    save(a.out/'task-boundaries.json',dict(objective=a.task_objective,boundaries=task_boundaries,evidence=evidence,
        input_sha256=evidence_pins,native_labels_preserved=True,extra_penalty=0))
    report=dict(version=SCOPE,state='passed',device=torch.cuda.get_device_name(),rows=len(rows),seeds=sorted({r['seed'] for r in rows}),
        route_active_rows=int(route_active.sum()),movement_active_rows=int(active.sum()),actor_changed=changed,actor_trials=trials,actor_loss=float(final_loss),active_kl=float(final_kl),
        critic_losses=critic_losses,log_probability_error=log_error,value_error=value_error,
        frozen_actor_verified=True,nonmovement_outputs_exact=True,
        source_model_sha256=model_hash,weights_sha256=exported_hash,checkpoint_sha256=sha(a.out/'checkpoint.pt'),
        inputs={str(p.resolve()):sha(p) for p in (a.model,a.request,a.config)},consumed_rollouts=[*consumed,*tokens],
        task_objective=a.task_objective,task_boundaries=len(task_boundaries),
        optimizer_initialization='resumed_motor_route_ppo' if a.resume else 'fresh_ppo_after_bc_no_bc_moments',
        demonstration_rows_used=0,actor_scope='motor_and_route',trainable_parameters=sum(p.numel() for p in scope.actor_parameters),
        training_performed=True,promotion=False,scope='Fresh own on-policy TRAIN movement PPO after BC. No quality evaluation or superiority claim.')
    save(a.out/'report.json',report)
    print(json.dumps({k:report[k] for k in ('state','rows','route_active_rows','actor_changed','active_kl')},allow_nan=False),flush=True)


if __name__=='__main__':main()
