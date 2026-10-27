"""Dedicated option PPO. Frozen aim/fire/motor; registered TRAIN only."""
import argparse
import json
from pathlib import Path
import sys

REPO=next(p for p in Path(__file__).resolve().parents if (p/'go.mod').is_file())
sys.path.insert(0,str(REPO/'scripts'))
from ppo_combat import torch, nn, training_devices, validate_objective, guarded_actor_step
from ppo_recurrent import prepare_context, sequence, durable_write
from combat_joint_aim_policy import base_model
from migrate_combat_navigation_cuda import branch
from combat_movement_option_ppo import MovementOptionScope, SCOPE
from combat_movement_options import verify
from process_combat_architecture_pool import read, save, sha
from combat_movement_option_returns import training_windows, targets
from combat_movement_option_bootstrap import recover


def admit(request, model_hash, consumed, old_seeds, pin):
    assert request['version']=='combat_movement_options_train_request_v1' and request['split']=='train'
    plan=read(pin(request['plan'],request['plan_sha256']))
    pool=read(pin(request['pool_report'],request['pool_report_sha256']))
    assert plan['model_sha256']==model_hash and pool['state']=='complete' and pool['source_unchanged']
    assert any(p['sha256']==request['plan_sha256'] and p['model_sha256']==model_hash for p in pool['plans'])
    expected={}
    for task in plan['tasks']:
        assert task['split']=='train' and task['modes']==['learned']
        assert task['episode']['recipe']['require_native_objective']
        for instance in task['instances']:
            assert instance['split']=='train' and instance['engine_seed'] not in expected
            expected[instance['engine_seed']]=instance
    rows=[];history=[];tokens=[];seeds=set();metadata=[]
    for case in request['cases']:
        seed=case['seed']; directory=Path(case['data']).resolve();capture=Path(case['capture_root']).resolve()
        assert seed in expected and seed not in seeds and seed not in old_seeds
        jobs=[j for j in pool['jobs'] if j['seed']==seed and j['mode']=='learned']
        assert len(jobs)==1 and not jobs[0]['error'] and capture.is_relative_to(Path(jobs[0]['root']).resolve())
        assert read(pin(capture/'fixture-0.json'))==expected[seed]
        manifest=read(pin(capture/'manifest.json'));assert manifest['native_objective_required']
        cap=read(pin(capture/'report.json'))
        assert cap['capture_complete'] and cap['provenance_valid'] and len(cap['results'])==1
        result=cap['results'][0]
        assert result['capture_valid'] and result['native_objective_required'] and result['seed']==seed
        assert result['generated_fixture']==expected[seed]
        meta=read(pin(directory/'report.json'))
        proof=read(pin(directory/'cuda-verification.json',meta['cuda_verification_sha256']))
        assert meta['version']=='combat_movement_options_rollout_v1' and meta['ppo_training_ready'] is False
        assert meta['numerical_verification']=='cuda_verified_v1' and meta['model_sha256']==proof['model_sha256']==model_hash
        assert not meta.get('diagnostic_only') and not any(k.startswith('diagnostic_') for k in meta)
        assert proof['state']=='passed' and proof['device']=='cuda' and proof['execution_contract_verified']
        assert proof['movement_options']['enabled'] and not proof['ordinary_training_eligible']
        assert proof['movement_options_training_scope_required']==SCOPE
        assert meta['observation_timing_contract']=='frozen_world_v1'
        for path,digest in meta['source_sha256'].items():pin(path,digest)
        bound={str(Path(p).resolve()).lower() for p in meta['source_sha256']}
        assert all(str((capture/name).resolve()).lower() in bound for name in ('manifest.json','report.json'))
        token=meta['native_rollout_sha256'];assert token not in consumed and token not in tokens
        def load(name,digest):
            return [json.loads(l) for l in pin(directory/name,digest).open(encoding='utf-8-sig')]
        rr=load('rollout.jsonl',meta['rollout_sha256']);hh=load('sequence.jsonl',meta['sequence_sha256'])
        assert len(rr)==meta['rows']>0 and len(hh)==meta['sequence_rows']
        assert {r['seed'] for r in rr}=={seed} and {r['seed'] for r in hh}=={seed}
        rows.extend(rr);history.extend(hh);tokens.append(token);seeds.add(seed);metadata.append(meta)
    assert seeds==set(expected), 'All planned TRAIN outcomes must be admitted'
    assert len({m['reward_config_sha256'] for m in metadata})==1
    return rows,history,tokens,seeds,metadata


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('model','request','config','out'):ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--resume',type=Path)
    a=ap.parse_args();assert not a.out.exists()
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    pins={}
    def pin(path,digest=None):
        path=Path(path).resolve();actual=sha(path)
        assert digest is None or actual==digest,'Input changed: '+str(path)
        pins[str(path)]=actual;return path
    for path in Path(__file__).parent.glob('*.py'):pin(path)
    for path in (REPO/'scripts').glob('*.py'):pin(path)
    model_hash=sha(pin(a.model));model=read(a.model);config=read(pin(a.config))
    assert config['version']==SCOPE and config['lambda_cadence']=='decision'
    assert 1<=config['actor_steps']<=100 and 1<=config['value_steps']<=100
    assert 0<config['clip']<=.3 and 0<config['target_kl']<=.05
    assert config['entropy']>=0 and config['max_grad_norm']>0
    torch.manual_seed(config['seed'])
    scope=MovementOptionScope(model,config)
    consumed=[];old_seeds=[];updates=0
    if a.resume:
        cp=torch.load(pin(a.resume),map_location='cuda',weights_only=True)
        consumed,progress=scope.restore(cp,model_hash)
        old_seeds=progress['consumed_seeds'];updates=progress['updates_completed']
    rows,context,tokens,seeds,metadata=admit(read(pin(a.request)),model_hash,consumed,old_seeds,pin)
    for meta in metadata:validate_objective(config,meta)
    native_rows=len(rows)
    actor=branch(base_model(model,allow_options=True,allow_shared=True),'actor')
    actor.requires_grad_(False)
    full_prepared=prepare_context(context,rows,'cuda')
    with torch.no_grad():
        bootstrap_values,bootstrap_evidence=recover(rows,context,full_prepared,actor,scope.options)
    rows,groups,censored_tails=training_windows(rows,context,bootstrap_values)
    prepared=prepare_context(context,rows,'cuda')
    with torch.no_grad():
        raw,memory_error=sequence(actor,prepared,32,True,'actor')
        assert memory_error<2e-5
        _,_,verification=verify(model,rows,context,prepared,actor)
        root_ids=torch.tensor([w['row'] for w in groups],device='cuda',dtype=torch.long)
        x=torch.tensor([rows[w['row']]['features'] for w in groups],device='cuda',dtype=torch.float32)
        raw=raw[root_ids].detach()
        choices=torch.tensor([rows[w['row']]['sample']['movement_option']['choice'] for w in groups],device='cuda',dtype=torch.long)
        old=torch.tensor([rows[w['row']]['sample']['movement_option']['log_probability'] for w in groups],device='cuda')
        logits,value=scope.options(x,raw)
        lp=logits.log_softmax(-1).gather(-1,choices[:,None]).squeeze(-1)
        lp_error=float((lp-old).abs().max())
        recorded=torch.tensor([w['old_value'] for w in groups],device='cuda')
        value_error=float((value-recorded).abs().max())
        assert lp_error<.001 and value_error<.001
        adv,returns,rewards=targets(rows,groups,config['gamma'],config['lambda'],torch)
        ad=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    def objective():
        logits,_=scope.options(x,raw)
        all_lp=logits.log_softmax(-1);lp=all_lp.gather(-1,choices[:,None]).squeeze(-1)
        delta=lp-old;ratio=delta.exp();entropy=-(all_lp.exp()*all_lp).sum(-1)
        loss=-torch.minimum(ratio*ad,ratio.clamp(1-config['clip'],1+config['clip'])*ad).mean()-config['entropy']*entropy.mean()
        return loss,((ratio-1)-delta).mean()
    parameters=list(scope.options.actor.parameters());trials=[]
    for _ in range(config['actor_steps']):
        loss,kl=objective()
        if float(kl.detach())>config['target_kl']:break
        assert bool(torch.isfinite(loss));scope.actor_optimizer.zero_grad(set_to_none=True);loss.backward()
        assert bool(torch.isfinite(nn.utils.clip_grad_norm_(parameters,config['max_grad_norm'])))
        trial=guarded_actor_step(parameters,scope.actor_optimizer,objective,loss.detach(),config['actor_lr'],config['target_kl'])
        trials.append(trial)
        if not trial['accepted']:break
    for _ in range(config['value_steps']):
        _,value=scope.options(x,raw);loss=(value-returns).square().mean()
        assert bool(torch.isfinite(loss));scope.value_optimizer.zero_grad(set_to_none=True);loss.backward()
        assert bool(torch.isfinite(nn.utils.clip_grad_norm_(scope.options.value.parameters(),config['max_grad_norm'])))
        scope.value_optimizer.step()
    with torch.no_grad():_,kl=objective();assert float(kl)<=config['target_kl']+1e-6
    scope.verify()
    for path,digest in list(pins.items()):pin(path,digest)
    a.out.mkdir(parents=True)
    encoded=json.dumps(scope.export(),separators=(',',':'),allow_nan=False).encode()
    assert len(encoded)<16*1024**2
    durable_write(a.out/'weights.json',lambda f:f.write(encoded))
    progress=dict(updates_completed=updates+1,consumed_seeds=sorted(set(old_seeds)|seeds))
    cp=scope.snapshot(sha(a.out/'weights.json'),consumed+tokens,progress)
    cp['parent_checkpoint_sha256']=sha(a.resume) if a.resume else None
    durable_write(a.out/'checkpoint.pt',lambda f:torch.save(cp,f))
    reopened=MovementOptionScope(read(a.out/'weights.json'),config)
    restored_tokens,restored_progress=reopened.restore(
        torch.load(a.out/'checkpoint.pt',map_location='cuda',weights_only=True),sha(a.out/'weights.json'))
    assert restored_tokens==consumed+tokens and restored_progress==progress
    from combat_full_joint_ppo import equal_tree
    assert equal_tree(reopened.actor_optimizer.state_dict(),scope.actor_optimizer.state_dict())
    assert equal_tree(reopened.value_optimizer.state_dict(),scope.value_optimizer.state_dict())
    report=dict(state='passed',version=SCOPE,device='cuda',training_performed=True,promotion=False,
        native_rows=native_rows,rows=len(rows),decisions=len(groups),censored_tails=censored_tails,
        decision_bootstrap=bootstrap_evidence,
        train_seeds=sorted(seeds),source_policy_frozen=True,
        frozen_source_sha256=scope.source_sha256,checkpoint_reopened_verified=True,
        actor_steps=sum(t['accepted'] for t in trials),actor_trials=trials,
        value_steps=config['value_steps'],final_approx_kl=float(kl),old_log_probability_max_error=lp_error,
        old_value_max_error=value_error,verification=verification,inputs=pins,**progress)
    save(a.out/'report.json',report)
    save(a.out/'complete.json',dict(state='complete',weights_sha256=sha(a.out/'weights.json'),
        checkpoint_sha256=sha(a.out/'checkpoint.pt'),report_sha256=sha(a.out/'report.json')))
    print(json.dumps({k:v for k,v in report.items() if k not in ('inputs','actor_trials')}))


if __name__=='__main__':main()
