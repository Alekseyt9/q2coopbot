"""CUDA causal fire supervision with fixed non-fire policy and TRAIN retention.

This explicit trainer admits paired diagnostic interventions as preferences,
never as ordinary PPO samples. Diagnostic worlds never enter either loss.
"""
import argparse
import copy
import json
from pathlib import Path
from ppo_combat import torch, nn, training_devices, guarded_actor_step
from ppo_recurrent import prepare_context, sequence
from combat_full_joint_ppo import FullJointScope, SCOPE, CHECKPOINT, equal_tree
from combat_conditional_fire_scope import FireFreeze, fork_config, CYCLE_ACTOR_SCOPE
from combat_conditional_fire import logits
from combat_causal_fire_pairs import load
from combat_checkpoint_scope import require_checkpoint_scope
from train_combat_full_joint_ppo_cuda import admit
from evaluate_registered_combat_models import read, save, sha

VERSION='combat_causal_fire_preferences_v1'
CHECKPOINT_VERSION='combat_causal_fire_preferences_checkpoint_v1'


def context(scope, rows, history):
    prepared=prepare_context(history,rows,'cuda')
    with torch.no_grad():raw,_=sequence(scope.actor,prepared)
    x=torch.tensor([r['features'] for r in rows],device='cuda',dtype=torch.float32)
    samples=[r['sample'] for r in rows]
    values=[torch.tensor([s.get(k,0) for s in samples],device='cuda',dtype=torch.long)
        for k in ('target','aim_mode','vertical','weapon')]
    latent=torch.tensor([s['latent'] for s in samples],device='cuda',dtype=torch.float32)
    return x,raw.detach(),values,latent


def fire(scope,batch):
    x,raw,(target,mode,vertical,weapon),latent=batch
    return logits(scope.actor.conditional_fire_branch,x,raw,target,mode,latent,vertical,weapon)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('model','checkpoint','dataset','retention-request','config','out'):
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--steps',type=int,help='Stop at this total step for exact continuation')
    ap.add_argument('--technical-only',action='store_true')
    a=ap.parse_args();assert not a.out.exists()
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();digest=sha(path)
        assert expected is None or expected==digest,'Input changed: '+str(path)
        pins[str(path)]=digest;return path
    for name in (__file__,'combat_conditional_fire_scope.py','combat_causal_fire_pairs.py','combat_weapon_fire_residual.py','combat_conditional_fire.py','combat_full_joint_ppo.py'):
        pin(Path(__file__).with_name(Path(name).name))
    config=read(pin(a.config));assert config['version']==VERSION
    cycle=config.get('cycle_residual_only',False)
    assert type(cycle) is bool
    assert 1<=config['steps']<=1000 and 0<config['actor_lr']<=.01
    assert config['retention_weight']>0 and 0<config['mean_kl_cap']<=.01 and 0<config['max_kl_cap']<=.1
    assert config['retention_rows_per_world']==8 and config['max_grad_norm']>0
    stop=config['steps'] if a.steps is None else a.steps
    assert 1<=stop<=config['steps']
    data=load(pin(a.dataset));model=read(pin(a.model));mh=sha(a.model)
    cp=torch.load(pin(a.checkpoint),map_location='cuda',weights_only=True)
    continuing=cp.get('checkpoint_training_scope')==VERSION
    if continuing:
        require_checkpoint_scope(cp,VERSION)
        assert cp['version']==CHECKPOINT_VERSION and cp['preference_config']==config
        assert cp['dataset_sha256']==sha(a.dataset) and cp['retention_request_sha256']==sha(a.retention_request)
        assert cp['technical_only']==a.technical_only
        origin=cp['origin'];start=cp['preference_steps_completed']
        assert cp['source_code_sha256']=={p:sha(Path(p)) for p in cp['source_code_sha256']}
        view=copy.deepcopy(cp);view.update(version=CHECKPOINT,checkpoint_training_scope=SCOPE)
    else:
        require_checkpoint_scope(cp,SCOPE)
        assert not cp.get('technical_only') and cp['version']==CHECKPOINT
        if cycle:
            from combat_weapon_fire_residual import CYCLE_VERSION
            assert cp['architecture_migration']['version']==CYCLE_VERSION
            source_hash=cp['architecture_migration']['source_weights_sha256']
        else:source_hash=mh
        origin=dict(model_sha256=source_hash,checkpoint_sha256=sha(a.checkpoint));start=0;view=cp
    assert start<stop and data['source_model_sha256']==origin['model_sha256']
    scope=FullJointScope(model,cp['config'],allow_cycle_fire=cycle);consumed=scope.restore(view,mh)
    if not continuing:
        narrow=copy.deepcopy(scope.config);narrow.update(actor_scope=CYCLE_ACTOR_SCOPE if cycle else 'conditional_fire_only',actor_lr=config['actor_lr'])
        fork_config(scope,narrow)
    freeze=FireFreeze(scope)
    value_state=copy.deepcopy(scope.value.state_dict());value_adam=copy.deepcopy(scope.value_optimizer.state_dict())
    def verify_frozen():
        freeze.verify()
        assert equal_tree(scope.value.state_dict(),value_state),'Critic changed'
        assert equal_tree(scope.value_optimizer.state_dict(),value_adam),'Critic Adam changed'
    worlds=data['cases'];train=[c for c in worlds if c['split']=='train' and c['eligible_for_preference'] and c['target']['preferred_attack'] is not None]
    assert train and not {c['seed'] for c in worlds}&set(cp['consumed_seeds'])
    request=read(pin(a.retention_request))
    rows,history,_,seeds,_=admit(request,origin['model_sha256'],consumed,set(cp['consumed_seeds']),pin)
    assert not seeds&{c['seed'] for c in worlds}
    selected=[]
    for seed in sorted(seeds):
        candidates=[r for r in rows if r['seed']==seed and not (r['features'][1132]>.5 and abs(r['features'][5]-11/30)<1e-6)]
        assert candidates,'No non-phase retention observations'
        n=min(len(candidates),config['retention_rows_per_world'])
        selected.extend(candidates[(i*(len(candidates)-1))//max(1,n-1)] for i in range(n))
    ends={seed:max(r['index'] for r in selected if r['seed']==seed) for seed in seeds}
    retained_history=[r for r in history if r['index']<=ends[r['seed']]]
    retention=context(scope,selected,retained_history)
    pairs=context(scope,[c['input'] for c in train],[r for c in train for r in c['input']['sequence_prefix']])
    labels=torch.tensor([c['target']['preferred_attack'] for c in train],device='cuda',dtype=torch.float32)
    ids=[(r['seed'],r['index']) for r in selected]
    if continuing:
        assert ids==cp['retention_ids'];teacher=cp['retention_teacher_logits'].detach()
    else:
        with torch.no_grad():teacher=fire(scope,retention).detach().clone()
    teacher_distribution=torch.distributions.Bernoulli(logits=teacher)
    def objective():
        fit=nn.functional.binary_cross_entropy_with_logits(fire(scope,pairs),labels)
        candidate=torch.distributions.Bernoulli(logits=fire(scope,retention))
        kl=torch.distributions.kl_divergence(teacher_distribution,candidate)
        divergence=torch.maximum(kl.mean()/config['mean_kl_cap'],kl.max()/config['max_kl_cap'])
        return fit+config['retention_weight']*kl.mean(),divergence
    with torch.no_grad():before=float(objective()[0])
    trials=[];completed=start
    for step in range(start,stop):
        loss,_=objective();assert bool(torch.isfinite(loss))
        scope.actor_optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=nn.utils.clip_grad_norm_(freeze.parameters,config['max_grad_norm']);assert bool(torch.isfinite(norm))
        trial=guarded_actor_step(freeze.parameters,scope.actor_optimizer,objective,loss.detach(),config['actor_lr'],1.,
            group_lrs=[g['lr'] for g in scope.actor_optimizer.param_groups],momentum_parameters=freeze.parameters)
        trials.append(trial)
        if not trial['accepted']:break
        completed=step+1
    verify_frozen()
    with torch.no_grad():
        after,bound=objective();assert float(bound)<=1+1e-6
        kl=torch.distributions.kl_divergence(teacher_distribution,torch.distributions.Bernoulli(logits=fire(scope,retention)))
        if cycle:assert torch.equal(fire(scope,retention),teacher),'Off-cycle fire changed'
        train_probability=fire(scope,pairs).sigmoid().tolist()
    for path,digest in pins.items():assert sha(Path(path))==digest
    a.out.mkdir(parents=True)
    save(a.out/'weights.json',scope.export());new_hash=sha(a.out/'weights.json')
    snapshot=scope.snapshot(new_hash,consumed)
    snapshot.update(version=CHECKPOINT_VERSION,checkpoint_training_scope=VERSION,technical_only=a.technical_only,
        origin=origin,preference_config=config,preference_steps_completed=completed,dataset_sha256=sha(a.dataset),
        retention_request_sha256=sha(a.retention_request),retention_ids=ids,retention_teacher_logits=teacher,
        consumed_seeds=cp['consumed_seeds'],updates_completed=cp['updates_completed'],
        source_code_sha256={p:d for p,d in pins.items() if p.endswith('.py')})
    torch.save(snapshot,a.out/'checkpoint.pt')
    report=dict(version=VERSION,state='passed',device='cuda',technical_only=a.technical_only,
        steps_completed=completed,steps_requested=stop,accepted_steps=sum(t['accepted'] for t in trials),trials=trials,cycle_residual_only=cycle,
        train_worlds=len(train),train_labels=labels.tolist(),train_attack_probability=train_probability,
        retention_worlds=len(seeds),retention_rows=len(selected),retention_mean_kl=float(kl.mean()),retention_max_kl=float(kl.max()),
        initial_loss=before,final_loss=float(after),non_fire_parameters_and_adam_unchanged=True,critic_and_adam_unchanged=True,
        weights_sha256=new_hash,checkpoint_sha256=sha(a.out/'checkpoint.pt'),inputs=pins,
        scope='Engineering TRAIN preference fit only; no diagnostic loss, runtime forced action or native quality claim.')
    save(a.out/'report.json',report)
    save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(json.dumps({k:v for k,v in report.items() if k not in ('inputs','trials')}))


if __name__=='__main__':main()
