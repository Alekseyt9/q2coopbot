"""Head-only augmented PPO on fresh, sealed registry TRAIN; CUDA required."""
import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import sys
import time
from evaluate_registered_combat_models import read,sha
from ppo_combat import torch,nn,layers,training_devices,advantages,validate_objective,guarded_actor_step
from ppo_recurrent import prepare_context,sequence,durable_json,durable_write
from combat_joint_aim_policy import JointAim,base_model,events,probabilities,validate
from combat_aim_candidates import VERSION,log_probability,entropy
from migrate_combat_navigation_cuda import branch
from fork_conditional_weapon_exploration_cuda import equal_tree


def cuda_tree(value):
    if isinstance(value,torch.Tensor):return value.to('cuda')
    if isinstance(value,dict):return {k:cuda_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cuda_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cuda_tree(v) for v in value)
    return value


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('source','corpus','compatibility','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists()
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();digest=sha(path);assert expected is None or digest==expected,str(path)
        pins[str(path)]=digest;return path
    source=a.source.resolve();root=a.corpus.resolve();data=root/'processing-train/merged'
    seal=read(pin(source/'complete.json'))
    for name,key in (('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')):pin(source/name,seal[key])
    model=read(source/'weights.json');validate(model)
    proof_seal=read(pin(a.compatibility/'complete.json'))
    proof=read(pin(a.compatibility/'report.json',proof_seal['report_sha256']))
    assert proof['state']=='complete' and proof['version']=='combat_joint_aim_native_compatibility_v1'
    assert proof['source_latent_and_native_commands_exact'] and proof['native_cases']==8 and proof['cuda_cases']==4
    proof_models=set()
    for path,digest in proof['inputs'].items():
        if Path(path).name=='cuda-verification.json':proof_models.add(read(pin(path,digest))['model_sha256'])
    assert proof_models=={seal['weights_sha256']},'Initial compatibility must bind this behavior model'
    closed=read(pin(root/'complete.json'));assert closed['state']=='complete' and not closed['training_performed']
    protocol=read(pin(root/'protocol.json',closed['protocol_sha256']))
    assert protocol['train_only'] and protocol['ppo_training_objective'] and len(protocol['assignments'])==1
    assignment=protocol['assignments'][0]
    assert assignment['split']=='train' and len(assignment['seeds'])==32 and len(set(assignment['seeds']))==32
    plan=read(pin(assignment['plan'],assignment['plan_sha256']))
    assert all(t['split']=='train' and t['modes']==['learned'] for t in plan['tasks'])
    assert plan['model_sha256']==seal['weights_sha256']
    assert closed['native_cases']==closed['cuda_cases']==32
    meta=read(pin(data/'report.json'))
    assert meta['version']=='combat_ppo_rollout_v1' and meta['action_distribution']==VERSION
    assert meta['numerical_verification']=='cuda_verified_v1' and meta['model_sha256']==seal['weights_sha256']
    for name,key in (('rollout.jsonl','rollout_sha256'),('sequence.jsonl','sequence_sha256')):pin(data/name,meta[key])
    for path,digest in meta['source_sha256'].items():pin(path,digest)
    cp=torch.load(source/'checkpoint.pt',map_location='cuda',weights_only=True)
    assert cp['weights_sha256']==seal['weights_sha256'] and cp['action_distribution']==VERSION
    assert meta['rollout_sha256'] not in cp['consumed_rollouts'],'TRAIN already consumed'
    assert cp.get('critic_credit',{}).get('mode','legacy')=='legacy'
    validate_objective(cp['config'],meta)
    config=dict(version='combat_joint_aim_head_training_v1',actor_steps=30,actor_lr=3e-4,value_steps=40,
        value_lr=cp['config']['value_lr'],gamma=cp['config']['gamma'],lambda_=cp['config']['lambda'],
        clip=cp['config']['clip'],entropy=cp['config']['entropy'],target_kl=cp['config']['target_kl'],
        max_grad_norm=cp['config']['max_grad_norm'],objective_reward_sha256=cp['config']['objective_reward_sha256'],
        actor_scope='aim_candidate_head_only',critic_credit='legacy_gae_then_value_fit',base_actor_and_predictor='frozen',
        retention_scope='legacy actor frozen exactly; no external penalty on new categorical head')
    rows=[json.loads(line) for line in (data/'rollout.jsonl').read_text().splitlines()]
    context=[json.loads(line) for line in (data/'sequence.jsonl').read_text().splitlines()]
    assert len(rows)==meta['rows'] and len(context)==meta['sequence_rows']
    assert {r['seed'] for r in rows}==set(assignment['seeds'])
    prepared=prepare_context(context,rows,'cuda')
    core=base_model(model);actor=branch(core,'actor');value=branch(core,'value');joint=JointAim(model)
    for key,module in (('actor',actor),('value',value),('aim_candidate_head',joint.head)):
        assert equal_tree(module.state_dict(),cp[key]),'JSON/checkpoint differs: '+key
    actor.requires_grad_(False);joint.hints.requires_grad_(False)
    actor_before=copy.deepcopy(actor.state_dict());hints_before=copy.deepcopy(joint.hints.state_dict());head_before=copy.deepcopy(joint.head.state_dict())
    std=torch.tensor(model['log_std'],device='cuda')
    assert torch.equal(std,cp['log_std'])
    opt=torch.optim.Adam(joint.head.parameters(),lr=config['actor_lr']);opt.load_state_dict(cp['aim_candidate_optimizer'])
    assert len(opt.param_groups)==1 and opt.param_groups[0]['lr']==config['actor_lr']
    value_opt=torch.optim.Adam(value.parameters(),lr=config['value_lr']);value_opt.load_state_dict(cp['value_optimizer'])
    fork=torch.load(source/'checkpoint.pt',map_location='cpu',weights_only=True)
    torch.set_rng_state(fork['rng']);torch.cuda.set_rng_state_all(fork['cuda_rng'])
    x=torch.tensor([r['features'] for r in rows],device='cuda',dtype=torch.float32)
    category=lambda key:torch.tensor([r['sample'].get(key,0) for r in rows],device='cuda',dtype=torch.long)
    target,mode,vertical,weapon=(category(k) for k in ('target','aim_mode','vertical','weapon'))
    final=torch.tensor([r['sample']['latent'] for r in rows],device='cuda',dtype=torch.float32)
    attack=torch.tensor([float(r['sample']['attack']) for r in rows],device='cuda')
    source_z,choice,timing,active=events(rows,x,target,mode,vertical,weapon)
    assert bool(active.any()) and len(choice[active].unique())==25,'Insufficient candidate coverage'
    native_lp=torch.tensor([r['sample']['log_probability'] for r in rows],device='cuda')
    def full_score(raw):
        return probabilities(joint,raw,std,source_z,final,attack,vertical,weapon,x,target,mode,choice,timing,
            actor.sampled_motion_aim_branch,actor.motor_branch,actor.conditional_fire_branch)
    cache=[]
    hook=joint.head.register_forward_pre_hook(lambda module,args:cache.append(args[0].detach().clone()))
    with torch.no_grad():
        raw,actor_memory_error=sequence(actor,prepared,32,True,'actor')
        old_value,value_memory_error=sequence(value,prepared,32,True,'value')
        old_lp,old_entropy,initial_logits,score_active=full_score(raw)
        assert torch.equal(active,score_active)
        log_error=float((old_lp-native_lp).abs().max())
        value_error=float((old_value[:,0]-torch.tensor([r['sample']['value'] for r in rows],device='cuda')).abs().max())
        assert log_error<.003 and value_error<1e-4 and max(actor_memory_error,value_memory_error)<2e-5
        assert len(cache)==1 and cache[0].is_cuda
        head_inputs=cache[0]
        old_cat=log_probability(initial_logits,choice,active)
        base_lp=(old_lp-old_cat).detach();base_entropy=(old_entropy-entropy(initial_logits,active)).detach()
        initial_logp25=initial_logits.log_softmax(-1).detach()
    hook.remove()
    adv,returns=advantages(rows,config['gamma'],config['lambda_'])
    ad=torch.tensor(adv,device='cuda');ad=(ad-ad.mean())/(ad.std(unbiased=False)+1e-8)
    ret=torch.tensor(returns,device='cuda')
    assert bool(torch.isfinite(ad).all()) and bool(torch.isfinite(ret).all())
    def current():
        logits=joint.head(head_inputs)
        lp=base_lp+log_probability(logits,choice,active)
        ent=base_entropy+entropy(logits,active)
        return lp,ent,logits
    def objective():
        lp,ent,_=current();delta=lp-native_lp;ratio=delta.exp()
        loss=-torch.minimum(ratio*ad,ratio.clamp(1-config['clip'],1+config['clip'])*ad).mean()-config['entropy']*ent.mean()
        kl=((ratio-1)-delta).mean()
        assert bool(torch.isfinite(loss)) and bool(torch.isfinite(kl))
        return loss,kl
    a.out.mkdir(parents=True);durable_json(a.out/'protocol.json',dict(config=config,source_weights_sha256=seal['weights_sha256'],rollout_sha256=meta['rollout_sha256'],
        compatibility_report_sha256=proof_seal['report_sha256'],training_split_only=True))
    durable_json(a.out/'progress.json',dict(stage='head-training',rows=len(rows),eligible=int(active.sum())))
    started=time.perf_counter();trials=[];gradients=[]
    with torch.no_grad():initial_loss=float(objective()[0])
    parameters=list(joint.head.parameters())
    for step in range(config['actor_steps']):
        loss,kl=objective()
        if float(kl.detach())>config['target_kl']:break
        opt.zero_grad(set_to_none=True);loss.backward()
        norm=nn.utils.clip_grad_norm_(parameters,config['max_grad_norm']);assert bool(torch.isfinite(norm))
        gradients.append(float(norm))
        trial=guarded_actor_step(parameters,opt,objective,loss.detach(),config['actor_lr'],config['target_kl'])
        trials.append(trial)
        if not trial['accepted']:break
    done=sum(t['accepted'] for t in trials)
    assert done>0 and max(gradients)>0 and not equal_tree(head_before,joint.head.state_dict()),'Head did not learn'
    durable_json(a.out/'progress.json',dict(stage='critic-training',head_steps=done))
    with torch.no_grad():value_loss_before=float((old_value[:,0]-ret).square().mean())
    value_losses=[]
    for step in range(config['value_steps']):
        predicted,_=sequence(value,prepared,32,False,'value');loss=(predicted[:,0]-ret).square().mean()
        assert bool(torch.isfinite(loss));value_opt.zero_grad(set_to_none=True);loss.backward()
        norm=nn.utils.clip_grad_norm_(value.parameters(),config['max_grad_norm']);assert bool(torch.isfinite(norm))
        value_opt.step();value_losses.append(float(loss.detach()))
    torch.cuda.synchronize();seconds=time.perf_counter()-started
    with torch.no_grad():
        loss,kl=objective();lp,_,logits=current()
        replay_raw,replay_actor_error=sequence(actor,prepared,32,True,'actor')
        assert torch.equal(replay_raw,raw),'Frozen actor outputs changed'
        full_lp,_,full_logits,full_active=full_score(replay_raw)
        cache_error=float((lp-full_lp).abs().max())
        assert cache_error<1e-4 and torch.equal(logits,full_logits) and torch.equal(full_active,active)
        logp25=logits.log_softmax(-1)
        category_kl=(initial_logp25.exp()*(initial_logp25-logp25)).sum(-1)
        assert float(category_kl.min())>-.00001 and float(kl)<=config['target_kl']+1e-6
        ratio=(lp-native_lp).exp();cat_ratio=(log_probability(logits,choice,active)-old_cat).exp()
        assert torch.allclose(ratio,cat_ratio,atol=.001,rtol=.001),'Frozen-source ratio does not reduce to category ratio'
        assert bool(torch.isfinite(ratio).all()) and bool(torch.isfinite(logp25).all())
        assert bool((lp[~active]-old_lp[~active]).abs().le(1e-5).all())
        changed=(logits.argmax(-1)!=initial_logits.argmax(-1))&active
        value_final,_=sequence(value,prepared,32,False,'value')
        value_loss_after=float((value_final[:,0]-ret).square().mean())
        assert equal_tree(actor_before,actor.state_dict()) and equal_tree(hints_before,joint.hints.state_dict())
        assert all(p.grad is None for p in actor.parameters()) and all(p.grad is None for p in joint.hints.parameters())
    updated=copy.deepcopy(model)
    updated['aim_candidate_head']['layers']=layers(joint.head.layers)
    base_v,cell_v=value.export();updated['value']=base_v
    key='attention' if core.get('attention') else 'memory';updated[key]['value']=cell_v
    validate(updated)
    restored=copy.deepcopy(updated);restored['aim_candidate_head']=copy.deepcopy(model['aim_candidate_head']);restored['value']=copy.deepcopy(model['value']);restored[key]['value']=copy.deepcopy(model[key]['value'])
    assert restored==model,'Unexpected model field changed'
    for path,digest in pins.items():assert sha(Path(path))==digest,'Input changed: '+path
    durable_json(a.out/'weights.json',updated)
    history=dict(version=config['version'],source_weights_sha256=seal['weights_sha256'],rollout_sha256=meta['rollout_sha256'],
        head_steps=done,critic_steps=config['value_steps'],final_approx_kl=float(kl),config=config)
    fork.update(weights_sha256=sha(a.out/'weights.json'),value=value.state_dict(),value_optimizer=value_opt.state_dict(),
        aim_candidate_head=joint.head.state_dict(),aim_candidate_optimizer=opt.state_dict(),
        consumed_rollouts=cp['consumed_rollouts']+[meta['rollout_sha256']],updates_completed=cp['updates_completed']+1,
        total_actor_steps=cp['total_actor_steps']+done,joint_aim_updates_completed=cp.get('joint_aim_updates_completed',0)+1,
        joint_aim_total_steps=cp.get('joint_aim_total_steps',0)+done,joint_aim_history=cp.get('joint_aim_history',[])+[history],
        rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
    durable_write(a.out/'checkpoint.pt',lambda f:torch.save(fork,f))
    loaded=torch.load(a.out/'checkpoint.pt',map_location='cuda',weights_only=True)
    for key in ('actor','log_std','actor_optimizer','actor_training','config','retention_weights','sequence_retention','model_migrations','rng','cuda_rng'):
        assert equal_tree(cp[key],loaded[key]),'Frozen state changed: '+key
    assert loaded['consumed_rollouts'][:-1]==cp['consumed_rollouts']
    for key,expected in (('value',value.state_dict()),('value_optimizer',value_opt.state_dict()),('aim_candidate_head',joint.head.state_dict()),('aim_candidate_optimizer',opt.state_dict())):
        assert equal_tree(cuda_tree(expected),loaded[key]),'Checkpoint roundtrip differs: '+key
    decoded=JointAim(read(a.out/'weights.json'))
    assert equal_tree(decoded.head.state_dict(),joint.head.state_dict())
    frozen=a.out/'trainer-sources';frozen.mkdir()
    helper_files={Path(__file__).resolve()}
    script_root=Path(__file__).resolve().parent
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name and Path(name).resolve().parent==script_root and Path(name).suffix=='.py':helper_files.add(Path(name).resolve())
    helpers={}
    for path in sorted(helper_files):
        target=frozen/path.name;shutil.copy2(path,target);helpers[path.name]=sha(target)
    report=dict(state='complete',version=config['version'],device='cuda',training_performed=True,promotion=False,
        rows=len(rows),sequence_rows=len(context),eligible_rows=int(active.sum()),candidate_counts=torch.bincount(choice[active],minlength=25).tolist(),
        actor_steps=done,critic_steps=config['value_steps'],head_parameters=sum(p.numel() for p in parameters),
        actor_trials=trials,gradient_norm_max=max(gradients),initial_loss=initial_loss,final_loss=float(loss),
        final_approx_kl=float(kl),category_exact_kl_active_mean=float(category_kl[active].mean()),
        category_exact_kl_all_rows_mean=float((category_kl*active).mean()),category_exact_kl_active_max=float(category_kl[active].max()),
        ratio_min=float(ratio.min()),ratio_max=float(ratio.max()),clip_fraction=float(((ratio-1).abs()>config['clip']).float().mean()),
        head_entropy_active_before=float(entropy(initial_logits,active)[active].mean()),head_entropy_active_after=float(entropy(logits,active)[active].mean()),
        baseline_probability_before=float(initial_logp25[active,0].exp().mean()),baseline_probability_after=float(logp25[active,0].exp().mean()),
        changed_argmax_rows=int(changed.sum()),old_log_probability_max_error=log_error,old_value_max_error=value_error,
        old_actor_memory_max_error=actor_memory_error,old_value_memory_max_error=value_memory_error,
        current_policy_cached_full_lp_max_error=cache_error,old_actor_and_adam_exact=True,old_rng_exact=True,predictor_exact=True,
        original_target_mode_weapon_pose_gaussian_terms_frozen=True,value_loss_before=value_loss_before,value_loss_after=value_loss_after,
        head_checkpoint_roundtrip_exact=True,seconds=seconds,behavior_sha256=seal['weights_sha256'],weights_sha256=sha(a.out/'weights.json'),
        rollout_sha256=meta['rollout_sha256'],sequence_sha256=meta['sequence_sha256'],protocol_sha256=sha(a.out/'protocol.json'),
        inputs=pins,trainer_sources=helpers,source=str(source),corpus=str(root),
        scope='Fresh TRAIN only. Full joint density = frozen original Gaussian/categories/final-action fire plus learned aim category. Cached frozen inputs verified against full CUDA scorer. Original actor/optimizer/RNG exact; critic updated with unchanged reward/GAE. Legacy actor training metadata retained as frozen provenance; new scope/config/history separate. Native nonuniform runtime and combat superiority not yet verified.')
    durable_json(a.out/'report.json',report)
    durable_json(a.out/'complete.json',dict(version='combat_update_complete_v1',state='complete',weights_sha256=report['weights_sha256'],
        checkpoint_sha256=sha(a.out/'checkpoint.pt'),report_sha256=sha(a.out/'report.json')))
    durable_json(a.out/'progress.json',dict(stage='complete',head_steps=done,rows=len(rows)))
    print(json.dumps({k:v for k,v in report.items() if k not in ('inputs','actor_trials','trainer_sources','scope','candidate_counts')}),flush=True)


if __name__=='__main__':main()
