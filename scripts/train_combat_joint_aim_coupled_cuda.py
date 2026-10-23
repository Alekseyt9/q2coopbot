"""Matched old-actor-only versus joint actor/candidate PPO, CUDA only."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time
from evaluate_registered_combat_models import read,sha
from ppo_combat import torch,nn,layers,training_devices,advantages,validate_objective,guarded_actor_step
from ppo_recurrent import prepare_context,sequence,durable_json,durable_write
from combat_joint_aim_policy import JointAim,base_model,events,probabilities,validate
from combat_aim_candidates import VERSION,log_probability,entropy
from combat_precision_head import probabilities as base_probabilities
from combat_conditional_fire import logits as fire_logits
from combat_spatial_aim import export as export_actor_branches
from combat_weapon_state_residual import optimizer as actor_optimizer
from migrate_combat_navigation_cuda import branch
from fork_conditional_weapon_exploration_cuda import equal_tree
from train_combat_joint_aim_cuda import cuda_tree

_active_output=None


def split_optimizer(opt,base_groups):
    """Preserve the original IDs; normalize only the independent head IDs."""
    state=copy.deepcopy(opt.state_dict());assert len(state['param_groups'])==base_groups+1
    group=state['param_groups'][-1];ids=group['params'];assert len(ids)==4
    head_group=copy.deepcopy(group);head_group.pop('name',None);head_group['params']=list(range(4))
    head=dict(state={i:state['state'][j] for i,j in enumerate(ids) if j in state['state']},param_groups=[head_group])
    base=dict(state={i:v for i,v in state['state'].items() if i not in ids},param_groups=state['param_groups'][:base_groups])
    return base,head


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('source','corpus','compatibility','head_reference','out'):ap.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists()
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();digest=sha(path);assert expected is None or digest==expected,str(path)
        pins[str(path)]=digest;return path
    source=a.source.resolve();corpus=a.corpus.resolve();data=corpus/'processing-train/merged'
    seal=read(pin(source/'complete.json'))
    for name,key in (('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')):pin(source/name,seal[key])
    model=read(source/'weights.json');validate(model);core=base_model(model)
    closed=read(pin(corpus/'complete.json'));protocol=read(pin(corpus/'protocol.json',closed['protocol_sha256']))
    assert closed['state']=='complete' and closed['native_cases']==closed['cuda_cases']==32 and not closed['training_performed']
    assert protocol['train_only'] and protocol['ppo_training_objective'] and len(protocol['assignments'])==1
    assignment=protocol['assignments'][0];assert assignment['split']=='train' and len(set(assignment['seeds']))==32
    plan=read(pin(assignment['plan'],assignment['plan_sha256']))
    assert plan['model_sha256']==seal['weights_sha256'] and all(t['split']=='train' and t['modes']==['learned'] for t in plan['tasks'])
    meta=read(pin(data/'report.json'))
    assert meta['version']=='combat_ppo_rollout_v1' and meta['numerical_verification']=='cuda_verified_v1'
    assert meta['action_distribution']==VERSION and meta['model_sha256']==seal['weights_sha256']
    for name,key in (('rollout.jsonl','rollout_sha256'),('sequence.jsonl','sequence_sha256')):pin(data/name,meta[key])
    for path,digest in meta['source_sha256'].items():pin(path,digest)
    ps=read(pin(a.compatibility/'complete.json'));proof=read(pin(a.compatibility/'report.json',ps['report_sha256']))
    assert proof['state']=='complete' and proof['source_latent_and_native_commands_exact'] and proof['native_cases']==8 and proof['cuda_cases']==4
    proof_models={read(pin(path,digest))['model_sha256'] for path,digest in proof['inputs'].items() if Path(path).name=='cuda-verification.json'}
    assert proof_models=={seal['weights_sha256']}
    cp=torch.load(source/'checkpoint.pt',map_location='cuda',weights_only=True)
    source_snapshot=copy.deepcopy(cp)
    raw_cp=torch.load(source/'checkpoint.pt',map_location='cpu',weights_only=True)
    assert cp['weights_sha256']==seal['weights_sha256'] and cp['action_distribution']==VERSION
    assert meta['rollout_sha256'] not in cp['consumed_rollouts'] and not cp.get('joint_aim_updates_completed',0)
    assert cp['retention_weights']==[0.,0.] and cp['sequence_retention'] is None and cp['policy_update']['method']=='ppo'
    assert cp['critic_credit']['mode']=='legacy' and cp['actor_training']['scope']=='all'
    validate_objective(cp['config'],meta)
    reference_seal=read(pin(a.head_reference/'complete.json'))
    reference_report=read(pin(a.head_reference/'report.json',reference_seal['report_sha256']))
    reference=torch.load(pin(a.head_reference/'checkpoint.pt',reference_seal['checkpoint_sha256']),map_location='cuda',weights_only=True)
    assert reference_report['behavior_sha256']==seal['weights_sha256'] and reference_report['rollout_sha256']==meta['rollout_sha256']
    assert reference_report['critic_steps']==40
    rows=[json.loads(line) for line in (data/'rollout.jsonl').read_text().splitlines()]
    context=[json.loads(line) for line in (data/'sequence.jsonl').read_text().splitlines()]
    assert len(rows)==meta['rows']==4061 and len(context)==meta['sequence_rows']
    assert {r['seed'] for r in rows}==set(assignment['seeds'])
    prepared=prepare_context(context,rows,'cuda');x=torch.tensor([r['features'] for r in rows],device='cuda',dtype=torch.float32)
    cat=lambda key:torch.tensor([r['sample'].get(key,0) for r in rows],device='cuda',dtype=torch.long)
    target,mode,vertical,weapon=(cat(k) for k in ('target','aim_mode','vertical','weapon'))
    final=torch.tensor([r['sample']['latent'] for r in rows],device='cuda',dtype=torch.float32)
    attack=torch.tensor([float(r['sample']['attack']) for r in rows],device='cuda')
    source_z,choice,timing,active=events(rows,x,target,mode,vertical,weapon)
    native_lp=torch.tensor([r['sample']['log_probability'] for r in rows],device='cuda')
    initial_actor,initial_value,initial_joint=branch(core,'actor'),branch(core,'value'),JointAim(model)
    for key,module in (('actor',initial_actor),('value',initial_value),('aim_candidate_head',initial_joint.head)):
        assert equal_tree(cp[key],module.state_dict()),key
    initial_std=torch.tensor(model['log_std'],device='cuda');assert torch.equal(initial_std,cp['log_std'])
    def full_score(actor,joint,std,raw):
        return probabilities(joint,raw,std,source_z,final,attack,vertical,weapon,x,target,mode,choice,timing,
            actor.sampled_motion_aim_branch,actor.motor_branch,actor.conditional_fire_branch)
    cache=[];hook=initial_joint.head.register_forward_pre_hook(lambda module,args:cache.append(args[0].detach().clone()))
    with torch.no_grad():
        old_raw,actor_memory_error=sequence(initial_actor,prepared,32,True,'actor')
        old_value,value_memory_error=sequence(initial_value,prepared,32,True,'value')
        old_lp,old_entropy,old_logits,old_active=full_score(initial_actor,initial_joint,initial_std,old_raw)
        assert torch.equal(active,old_active) and len(cache)==1
        log_error=float((old_lp-native_lp).abs().max());assert log_error<.003
        value_error=float((old_value[:,0]-torch.tensor([r['sample']['value'] for r in rows],device='cuda')).abs().max())
        assert value_error<1e-4 and max(actor_memory_error,value_memory_error)<2e-5
        head_inputs=cache[0];old_cat=log_probability(old_logits,choice,active);old_base=(old_lp-old_cat).detach()
        old_logp25=old_logits.log_softmax(-1).detach()
    hook.remove();del initial_actor,initial_value,initial_joint
    adv,returns=advantages(rows,cp['config']['gamma'],cp['config']['lambda'])
    ad=torch.tensor(adv,device='cuda');ad=(ad-ad.mean())/(ad.std(unbiased=False)+1e-8);ret=torch.tensor(returns,device='cuda')
    rates=[g['lr'] for g in cp['actor_optimizer']['param_groups']];assert rates==[3e-5,.0003,.0003,.0003]
    config=dict(version='combat_joint_aim_coupled_training_v1',scopes=['base-actor-only','joint'],actor_steps=30,value_steps=40,
        base_group_lrs=rates,candidate_lr=.0003,value_lr=cp['config']['value_lr'],clip=cp['config']['clip'],target_kl=cp['config']['target_kl'],
        entropy=cp['config']['entropy'],max_grad_norm=cp['config']['max_grad_norm'],gamma=cp['config']['gamma'],lambda_=cp['config']['lambda'],
        objective_reward_sha256=cp['config']['objective_reward_sha256'],critic_credit='legacy',source=str(source),corpus=str(corpus),
        behavior_sha256=seal['weights_sha256'],rollout_sha256=meta['rollout_sha256'],predictor='frozen',head_reference_use='critic equality check only')
    a.out.mkdir(parents=True);durable_json(a.out/'protocol.json',config)
    global _active_output
    _active_output=a.out
    results=[];critic_states=[]
    for scope in config['scopes']:
        assert equal_tree(cp,source_snapshot),'A previous fit mutated the shared source checkpoint'
        directory=a.out/scope;directory.mkdir();durable_json(a.out/'progress.json',dict(stage='gpu-training',scope=scope))
        actor,value,joint=branch(core,'actor'),branch(core,'value'),JointAim(model)
        std=nn.Parameter(torch.tensor(model['log_std'],device='cuda'))
        joint.hints.requires_grad_(False);joint.head.requires_grad_(scope=='joint')
        before_hint=copy.deepcopy(joint.hints.state_dict());before_head=copy.deepcopy(joint.head.state_dict())
        opt=actor_optimizer(actor,std,rates[0],rates[1],rates[2],rates[3]);opt.load_state_dict(copy.deepcopy(cp['actor_optimizer']))
        assert equal_tree(cuda_tree(opt.state_dict()),cp['actor_optimizer'])
        head_opt=torch.optim.Adam(joint.head.parameters(),lr=config['candidate_lr']);head_opt.load_state_dict(copy.deepcopy(cp['aim_candidate_optimizer']))
        assert equal_tree(cuda_tree(head_opt.state_dict()),cp['aim_candidate_optimizer'])
        groups=len(opt.param_groups);head_group=copy.copy(head_opt.param_groups[0]);head_group['name']='aim_candidate_head'
        opt.add_param_group(head_group)
        for p in joint.head.parameters():
            if p in head_opt.state:opt.state[p]=copy.deepcopy(head_opt.state[p])
        before_base_opt,before_head_opt=split_optimizer(opt,groups)
        assert equal_tree(cuda_tree(before_base_opt),cp['actor_optimizer']) and equal_tree(cuda_tree(before_head_opt),cp['aim_candidate_optimizer'])
        frozen_head_states={p:copy.deepcopy(opt.state.get(p,{})) for p in joint.head.parameters()}
        value_opt=torch.optim.Adam(value.parameters(),lr=config['value_lr']);value_opt.load_state_dict(copy.deepcopy(cp['value_optimizer']))
        assert equal_tree(cuda_tree(value_opt.state_dict()),cp['value_optimizer'])
        torch.set_rng_state(raw_cp['rng']);torch.cuda.set_rng_state_all(raw_cp['cuda_rng'])
        def score(raw):
            lp,ent=base_probabilities(raw,std,source_z,attack,vertical,weapon,x,target,mode,
                actor.sampled_motion_aim_branch,actor.motor_branch,actor.conditional_fire_branch)
            before=torch.distributions.Bernoulli(logits=fire_logits(actor.conditional_fire_branch,x,raw,target,mode,source_z,vertical,weapon))
            after=torch.distributions.Bernoulli(logits=fire_logits(actor.conditional_fire_branch,x,raw,target,mode,final,vertical,weapon))
            base_lp=lp+after.log_prob(attack)-before.log_prob(attack)
            base_ent=ent+after.entropy()-before.entropy()
            logits=joint.head(head_inputs)
            return base_lp+log_probability(logits,choice,active),base_ent+entropy(logits,active),logits,base_lp,base_ent
        def current():
            raw,_=sequence(actor,prepared,32,False,'actor');return score(raw)
        def objective():
            lp,ent,_,_,_=current();delta=lp-native_lp;ratio=delta.exp()
            loss=-torch.minimum(ratio*ad,ratio.clamp(1-config['clip'],1+config['clip'])*ad).mean()-config['entropy']*ent.mean()
            kl=((ratio-1)-delta).mean();assert bool(torch.isfinite(loss)) and bool(torch.isfinite(kl))
            return loss,kl
        def project():
            std.clamp_(-8,1)
            if scope=='base-actor-only':
                for p,state in frozen_head_states.items():
                    if state:opt.state[p]=copy.deepcopy(state)
                    else:opt.state.pop(p,None)
        parameters=list(actor.parameters())+[std]+list(joint.head.parameters());group_lrs=rates+[config['candidate_lr']]
        trials=[];gradients=[];started=time.perf_counter()
        with torch.no_grad():initial_loss=float(objective()[0])
        for step in range(config['actor_steps']):
            loss,kl=objective()
            if float(kl.detach())>config['target_kl']:break
            opt.zero_grad(set_to_none=True);loss.backward()
            norm=nn.utils.clip_grad_norm_(parameters,config['max_grad_norm']);assert bool(torch.isfinite(norm));gradients.append(float(norm))
            trial=guarded_actor_step(parameters,opt,objective,loss.detach(),rates[0],config['target_kl'],project,group_lrs=group_lrs)
            trials.append(trial)
            if not trial['accepted']:break
        done=sum(t['accepted'] for t in trials);assert done>0 and max(gradients)>0
        for step in range(config['value_steps']):
            predicted,_=sequence(value,prepared,32,False,'value');loss=(predicted[:,0]-ret).square().mean();assert bool(torch.isfinite(loss))
            value_opt.zero_grad(set_to_none=True);loss.backward();norm=nn.utils.clip_grad_norm_(value.parameters(),config['max_grad_norm']);assert bool(torch.isfinite(norm));value_opt.step()
        torch.cuda.synchronize();seconds=time.perf_counter()-started
        assert equal_tree(cp,source_snapshot),'Optimizer state aliases the immutable source checkpoint'
        with torch.no_grad():
            final_loss,final_kl=objective();lp,ent,logits,base_lp,base_ent=current()
            raw,_=sequence(actor,prepared,32,False,'actor');flp,_,flogits,factive=full_score(actor,joint,std,raw)
            cache_error=float((lp-flp).abs().max());assert cache_error<1e-4 and torch.equal(logits,flogits) and torch.equal(factive,active)
            assert float(final_kl)<=config['target_kl']+1e-6 and not equal_tree(actor.state_dict(),cp['actor'])
            assert equal_tree(before_hint,joint.hints.state_dict()) and all(p.grad is None for p in joint.hints.parameters())
            cat_kl=(old_logp25.exp()*(old_logp25-logits.log_softmax(-1))).sum(-1)
            delta_base=base_lp-old_base;base_kl=((delta_base.exp()-1)-delta_base).mean()
            ratio=(lp-native_lp).exp();assert bool(torch.isfinite(ratio).all())
        base_state,head_state=split_optimizer(opt,groups)
        if scope=='base-actor-only':
            assert equal_tree(before_head,joint.head.state_dict()) and equal_tree(cuda_tree(head_state),cp['aim_candidate_optimizer'])
            assert all(p.grad is None for p in joint.head.parameters())
        else:assert not equal_tree(before_head,joint.head.state_dict()) and head_state['state']
        assert equal_tree(value.state_dict(),reference['value']) and equal_tree(cuda_tree(value_opt.state_dict()),reference['value_optimizer']),'Critic differs from head-only matched reference'
        if critic_states:assert equal_tree(value.state_dict(),critic_states[0])
        critic_states.append(copy.deepcopy(value.state_dict()))
        base_a,cell_a=actor.export();base_v,cell_v=value.export();key='attention' if core.get('attention') else 'memory'
        updated_core=copy.deepcopy(core);updated_core.update(actor=base_a,value=base_v,log_std=std.detach().tolist())
        updated_core[key]['actor']=cell_a;updated_core[key]['value']=cell_v;export_actor_branches(updated_core,actor)
        updated={**updated_core,'action_distribution':VERSION,'aim_candidate_head':copy.deepcopy(model['aim_candidate_head'])}
        updated['aim_candidate_head']['layers']=layers(joint.head.layers);validate(updated)
        # Validation constructors allocate randomly initialized modules before
        # loading exported tensors. Their throwaway draws must not alter the
        # resumable training RNG or the next matched fork's state.
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            decoded_actor=branch(base_model(updated),'actor');decoded_head=JointAim(updated)
            assert equal_tree(decoded_actor.state_dict(),actor.state_dict()) and equal_tree(decoded_head.head.state_dict(),joint.head.state_dict())
            with torch.no_grad():
                decoded_raw,_=sequence(decoded_actor,prepared,32,False,'actor')
                decoded_lp=full_score(decoded_actor,decoded_head,torch.tensor(updated['log_std'],device='cuda'),decoded_raw)[0]
                roundtrip_error=float((decoded_lp-lp).abs().max());assert roundtrip_error<1e-4
        for path,digest in pins.items():assert sha(Path(path))==digest,'Input changed: '+path
        durable_json(directory/'weights.json',updated)
        record=dict(version=config['version'],scope=scope,head_steps=done if scope=='joint' else 0,actor_steps=done,
            critic_steps=40,rollout_sha256=meta['rollout_sha256'],source_weights_sha256=seal['weights_sha256'],config=config)
        fork=copy.deepcopy(raw_cp)
        fork.update(weights_sha256=sha(directory/'weights.json'),actor=actor.state_dict(),value=value.state_dict(),log_std=std.detach(),
            actor_optimizer=base_state,value_optimizer=value_opt.state_dict(),aim_candidate_head=joint.head.state_dict(),aim_candidate_optimizer=head_state,
            consumed_rollouts=cp['consumed_rollouts']+[meta['rollout_sha256']],updates_completed=cp['updates_completed']+1,total_actor_steps=cp['total_actor_steps']+done,
            joint_aim_updates_completed=1,joint_aim_total_steps=done if scope=='joint' else 0,joint_aim_history=cp.get('joint_aim_history',[])+[record],
            actor_training={**cp['actor_training'],'scope':'joint_aim_'+scope},
            actor_training_forks=cp.get('actor_training_forks',[])+[dict(previous=cp['actor_training'],current_scope=scope,source_checkpoint_sha256=seal['checkpoint_sha256'])],
            rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        durable_write(directory/'checkpoint.pt',lambda f:torch.save(fork,f))
        loaded=torch.load(directory/'checkpoint.pt',map_location='cuda',weights_only=True)
        for k in ('actor','value','log_std','actor_optimizer','value_optimizer','aim_candidate_head','aim_candidate_optimizer'):
            assert equal_tree(cuda_tree(fork[k]),loaded[k]),'Roundtrip state differs: '+k
        assert equal_tree(cp['rng'],loaded['rng']) and equal_tree(cp['cuda_rng'],loaded['cuda_rng'])
        assert loaded['consumed_rollouts'][:-1]==cp['consumed_rollouts']
        report=dict(state='complete',version=config['version'],device='cuda',scope=scope,training_performed=True,promotion=False,
            rows=len(rows),eligible_rows=int(active.sum()),actor_steps=done,critic_steps=40,actor_trials=trials,gradient_norm_max=max(gradients),
            initial_loss=initial_loss,final_loss=float(final_loss),final_approx_kl=float(final_kl),base_approx_kl=float(base_kl),
            category_exact_kl_active_mean=float(cat_kl[active].mean()),base_entropy_before=float((old_entropy-entropy(old_logits,active)).mean()),base_entropy_after=float(base_ent.mean()),
            category_entropy_active_before=float(entropy(old_logits,active)[active].mean()),category_entropy_active_after=float(entropy(logits,active)[active].mean()),
            ratio_min=float(ratio.min()),ratio_max=float(ratio.max()),clip_fraction=float(((ratio-1).abs()>config['clip']).float().mean()),
            baseline_probability_after=float(logits.softmax(-1)[active,0].mean()),old_lp_max_error=log_error,current_cached_full_lp_max_error=cache_error,
            json_current_policy_replay_max_error=roundtrip_error,initial_actor_and_head_adam_exact=True,base_actor_changed=True,
            predictor_exact=True,head_frozen_exact=scope=='base-actor-only',critic_exact_to_head_only_reference=True,
            checkpoint_optimizer_roundtrip_exact=True,rng_preserved=True,source_checkpoint_memory_exact=True,seconds=seconds,
            source=str(source),behavior_sha256=seal['weights_sha256'],weights_sha256=sha(directory/'weights.json'),
            rollout_sha256=meta['rollout_sha256'],protocol_sha256=sha(a.out/'protocol.json'),inputs=pins,
            native_runtime_verified=False,quality_verified=False,
            explanation='All original actor branches/std update; candidate head '+('frozen exactly.' if scope=='base-actor-only' else 'updates jointly.'),
            validation_scope='TRAIN current-policy numerical acceptance only; no native nonuniform/base-actor runtime or combat superiority claim yet.')
        durable_json(directory/'report.json',report);durable_json(directory/'complete.json',dict(state='complete',version='combat_update_complete_v1',
            weights_sha256=report['weights_sha256'],checkpoint_sha256=sha(directory/'checkpoint.pt'),report_sha256=sha(directory/'report.json')))
        results.append(dict(scope=scope,actor_steps=done,report_sha256=sha(directory/'report.json'),weights_sha256=report['weights_sha256']))
        print(json.dumps({k:v for k,v in report.items() if k not in ('inputs','actor_trials')}),flush=True)
        del actor,value,joint,decoded_actor,decoded_head,loaded,opt,value_opt,head_opt
    frozen=a.out/'trainer-sources';frozen.mkdir();helpers={};script_root=Path(__file__).resolve().parent
    files={Path(__file__).resolve()}
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name and Path(name).resolve().parent==script_root and Path(name).suffix=='.py':files.add(Path(name).resolve())
    for path in sorted(files):shutil.copy2(path,frozen/path.name);helpers[path.name]=sha(frozen/path.name)
    report=dict(state='complete',version=config['version'],device='cuda',matched_source_and_train=True,critic_exact_across_all_three_fits=True,
        scopes=results,protocol_sha256=sha(a.out/'protocol.json'),inputs=pins,trainer_sources=helpers,training_performed=True,promotion=False,native_runtime_verified=False)
    durable_json(a.out/'report.json',report);durable_json(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json'),scopes=results))
    durable_json(a.out/'progress.json',dict(stage='complete',scopes=results))


if __name__=='__main__':
    try:main()
    except Exception as error:
        if _active_output is not None:
            durable_json(_active_output/'progress.json',dict(stage='failed',error=str(error),exception=type(error).__name__))
        raise
