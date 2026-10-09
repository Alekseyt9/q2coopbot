"""Offline PPO-Clip update. Go owns observations, sampling, commands and harness."""
import argparse, copy, json, math, pathlib, time, sys
sys.pycache_prefix=str(pathlib.Path(__file__).resolve().parents[1]/'workspace'/'build'/'python-cache')
# Shared F: cache configuration is set before importing torch.
from train_combat_bc import torch, nn, sha, training_devices
from combat_retention_bank import read as read_bank, validate as validate_bank, pin_bank
from combat_weapon_head import distribution as weapon_distribution, probabilities as action_probabilities, feature_mask, HEAD_VERSION

def network(layers):
    blocks=[]
    for i,l in enumerate(layers):
        linear=nn.Linear(len(l['weight'][0]),len(l['bias']))
        with torch.no_grad():linear.weight.copy_(torch.tensor(l['weight']));linear.bias.copy_(torch.tensor(l['bias']))
        blocks.append(linear)
        if i<2:blocks.append(nn.ReLU())
    return nn.Sequential(*blocks)

def layers(model):return [{'weight':l.weight.detach().cpu().tolist(),'bias':l.bias.detach().cpu().tolist()} for l in model if isinstance(l,nn.Linear)]

def restore_optimizer(optimizer,snapshot):
    # PyTorch can reuse same-device tensors from load_state_dict. Each trial
    # needs its own moments/step counters so a rejected step cannot mutate
    # the baseline used by later retries or the final rejection restore.
    optimizer.load_state_dict(copy.deepcopy(snapshot))

def guarded_actor_step(parameters,optimizer,objective,loss_before,base_lr,target_kl,project=None):
    """Backtrack Adam; repair only a confirmed uphill first-moment direction.

    Keep second moments and step counters. A rejected fallback restores every
    parameter and the complete optimizer snapshot, including learning rates.
    """
    parameters=list(parameters);saved=[p.detach().clone() for p in parameters]
    snapshot=copy.deepcopy(optimizer.state_dict());gradients=[p.grad.detach().clone() if p.grad is not None else None for p in parameters]
    result=dict(accepted=False,retry=None,direction_fallback=False,initial_grad_dot_delta=None)
    def restore():
        with torch.no_grad():
            for p,v in zip(parameters,saved):p.copy_(v)
        restore_optimizer(optimizer,snapshot)
    fallback=False
    for retry in range(13):
        restore()
        if fallback:
            # Retain accumulated squared-gradient scale and the Adam clock;
            # replacing only momentum makes this proposal follow today's grad.
            for state in optimizer.state.values():
                if 'exp_avg' in state:state['exp_avg'].zero_()
        for group in optimizer.param_groups:group['lr']=base_lr*(.5**retry)
        optimizer.step()
        with torch.no_grad():
            if project is not None:project()
            dot=sum((g*(p-v)).sum() for p,v,g in zip(parameters,saved,gradients) if g is not None)
            if retry==0:
                result['initial_grad_dot_delta']=float(dot)
                if torch.isfinite(dot) and dot>0:
                    fallback=True;result['direction_fallback']=True
                    # Re-evaluate repaired momentum at this same step size.
                    restore()
                    for state in optimizer.state.values():
                        if 'exp_avg' in state:state['exp_avg'].zero_()
                    for group in optimizer.param_groups:group['lr']=base_lr
                    optimizer.step()
                    if project is not None:project()
                    dot=sum((g*(p-v)).sum() for p,v,g in zip(parameters,saved,gradients) if g is not None)
            candidate_loss,candidate_kl=objective()
            accepted=bool(torch.isfinite(dot) and dot<=0 and torch.isfinite(candidate_loss) and torch.isfinite(candidate_kl) and candidate_kl<=target_kl and candidate_loss<=loss_before+1e-7)
        if accepted:
            for group in optimizer.param_groups:group['lr']=base_lr
            result.update(accepted=True,retry=retry,accepted_grad_dot_delta=float(dot));return result
    restore();return result

def log_prob(actor,std,x,z,attack,vertical,weapon=None,target=None,mode=None):
    raw=actor(x)
    if raw.shape[1]==81:
        from combat_precision_head import probabilities
        return probabilities(raw,std,z,attack,vertical,weapon,x,target,mode)
    if raw.shape[1]==45:
        from combat_target_head import probabilities
        return probabilities(raw,std,z,attack,vertical,weapon,x,target)
    # Keep vertical logits separate from the optional 12-way weapon head.
    # PPO uses pre-tanh latent likelihood; fixed Jacobian cancels in ratio.
    return action_probabilities(raw,std,z,attack,vertical,weapon,
                                feature_mask(x) if raw.shape[1]==20 else None)

def anchor_kl(raw,std,teacher,teacher_std,weights=None,weapon_mask=None):
    """KL(anchor || policy); legacy 8-output anchors retain legacy heads only."""
    if raw.shape[1] not in (8,20) or teacher.shape[1] not in (8,20):raise ValueError('Unknown anchor action width')
    if teacher.shape[1]==20 and raw.shape[1]!=20:raise ValueError('Weapon anchor requires weapon policy')
    distributions=torch.distributions
    normal=distributions.kl_divergence(distributions.Normal(teacher[:,:4],teacher_std.exp()),distributions.Normal(raw[:,:4],std.exp())).sum(1)
    attack=distributions.kl_divergence(distributions.Bernoulli(logits=teacher[:,4]),distributions.Bernoulli(logits=raw[:,4]))
    vertical=distributions.kl_divergence(distributions.Categorical(logits=teacher[:,5:8]),distributions.Categorical(logits=raw[:,5:8]))
    terms=normal+attack+vertical
    if teacher.shape[1]==20:
        terms=terms+distributions.kl_divergence(weapon_distribution(teacher[:,8:20],weapon_mask),weapon_distribution(raw[:,8:20],weapon_mask))
    terms=terms.clamp_min(0)
    return terms.mean() if weights is None else (terms*weights).sum()

def retention_schedule(anchor_sha,initial,horizon,updates,previous=None,mode='linear'):
    if mode not in ('linear','constant'):raise ValueError('Unknown retention mode')
    if not anchor_sha:
        if previous: raise ValueError('Resumed retention requires the pinned anchor')
        if mode!='linear':raise ValueError('Constant retention requires an anchor')
        return None,0.
    if not math.isfinite(initial) or initial<=0 or horizon<2:
        raise ValueError('Positive finite retention weight and at least two updates required')
    spec=dict(anchor_sha256=anchor_sha,initial_weight=initial,updates=horizon,start_updates=updates)
    # Preserve legacy linear checkpoints byte-for-byte; constant is explicit.
    if mode=='constant':spec['mode']=mode
    if previous:
        spec['start_updates']=previous['start_updates']
        if spec!=previous: raise ValueError('Retention schedule or anchor changed on resume')
    elapsed=updates-spec['start_updates']
    if elapsed<0: raise ValueError('Retention counter precedes schedule')
    return spec,initial if mode=='constant' else initial*max(0.,1-elapsed/(horizon-1))

def advantages(rows,gamma,lam):
    values=[r['sample']['value'] for r in rows];adv=[0.0]*len(rows)
    for i in reversed(range(len(rows))):
        r=rows[i];delta=r['reward']+gamma*(0 if r['terminal'] else r['next_value'])-values[i]
        consecutive=i+1<len(rows) and rows[i+1]['seed']==r['seed'] and rows[i+1]['index']==r['index']+1 and rows[i+1]['frame']==r['next_frame']
        adv[i]=delta+(gamma*lam*adv[i+1] if consecutive and not r['terminal'] and not r['truncated'] else 0)
    return adv,[a+v for a,v in zip(adv,values)]

def restore_checkpoint(path,model_path,config,actor,value,std,rollout_sha):
    """Load only our tensor/primitive checkpoint, and verify exported weights."""
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint.get('version')=='combat_ppo_checkpoint_v2':
        assert checkpoint['weights_sha256']==sha(model_path), 'Checkpoint belongs to other weights'
        consumed=checkpoint['consumed_rollouts'];updates=checkpoint['updates_completed'];steps=checkpoint['total_actor_steps']
    else:
        # One-time migration of the prior pilot, whose report pins its weights.
        report=json.loads(pathlib.Path(path).with_name('report.json').read_text())
        assert report['weights_sha256']==sha(model_path), 'Legacy checkpoint belongs to other weights'
        consumed=[report['rollout_sha256']];updates=1;steps=report['actor_steps']
    assert checkpoint['config']==config, 'Checkpoint config differs'
    assert rollout_sha not in consumed, 'Rollout already consumed by checkpoint'
    for key,module in [('actor',actor),('value',value)]:
        expected=module.state_dict();actual=checkpoint[key]
        assert expected.keys()==actual.keys()
        assert all(torch.equal(v.cpu(),actual[k].cpu()) for k,v in expected.items()), 'Checkpoint tensors differ from JSON weights'
    assert torch.equal(std.detach().cpu(),checkpoint['log_std'].cpu()), 'Checkpoint standard deviation differs'
    return checkpoint,consumed,updates,steps

def validate_objective(config,meta):
    expected=config.get('objective_reward_sha256')
    assert meta.get('reward_version') not in ('combat_reward_v2','combat_reward_v3','combat_reward_v4') or expected, 'Kill objective must be pinned in training config'
    if meta.get('reward_version') in ('combat_reward_v3','combat_reward_v4'):
        assert meta.get('aim_gamma')==config['gamma'], 'Aim shaping discount differs from PPO gamma'
    if expected:
        assert expected.lower()==meta.get('reward_config_sha256','').lower(), 'Rollout reward differs from checkpoint objective'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--init-bc');ap.add_argument('--model');ap.add_argument('--data');ap.add_argument('--resume');ap.add_argument('--config',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--anchor-model',type=pathlib.Path);ap.add_argument('--retention-weight',type=float,default=1.);ap.add_argument('--retention-updates',type=int,default=4);ap.add_argument('--retention-mode',choices=('linear','constant'),default='linear')
    ap.add_argument('--retention-bank',type=pathlib.Path);ap.add_argument('--bank-weight',type=float,default=1.);a=ap.parse_args()
    out=pathlib.Path(a.out);out.mkdir(exist_ok=False);config=json.loads(pathlib.Path(a.config).read_text(encoding='utf-8-sig'))
    assert config['version']=='combat_ppo_training_v1';torch.set_num_threads(2);torch.manual_seed(config['seed']);torch.use_deterministic_algorithms(True)
    if a.init_bc:
        assert not a.resume
        bc=json.loads(pathlib.Path(a.init_bc).read_text());assert bc['kind']=='combat_bc_mlp_v1'
        critic=copy.deepcopy(bc['layers']);critic[-1]={'weight':[[0.0]*len(critic[-1]['weight'][0])],'bias':[0.0]}
        model={'kind':'combat_ppo_v1','feature_version':bc['feature_version'],'actor':bc['layers'],'value':critic,'log_std':config['log_std'],'sampling_seed':0,'deterministic':False}
        (out/'weights.json').write_text(json.dumps(model,allow_nan=False))
        (out/'report.json').write_text(json.dumps({'scope':'BC initialized stochastic actor, zero value; no PPO update yet','bc_sha256':sha(pathlib.Path(a.init_bc)),'config_sha256':sha(pathlib.Path(a.config))},indent=2));return
    assert a.model and a.data
    model_path=pathlib.Path(a.model);root=pathlib.Path(a.data);model=json.loads(model_path.read_text());meta=json.loads((root/'report.json').read_text())
    assert not any(model.get(k) for k in ('memory','attention','entity_attention')), 'Architecture models require ppo_recurrent.py'
    validate_objective(config,meta)
    assert model['kind']=='combat_ppo_v1' and not model['deterministic'] and meta['version']=='combat_ppo_rollout_v1' and sha(model_path)==meta['model_sha256']
    weapon_head=model.get('weapon_head')
    target_head=model.get('target_head')
    if target_head:
        if model.get('aim_mode_head'):
            from combat_precision_head import validate_precision_model as validate_model
        else:
            from combat_target_head import validate_model
        validate_model(model)
        assert not a.anchor_model and not a.retention_bank, 'Target-head retention requires a target-aware anchor objective'
    elif weapon_head:assert weapon_head==HEAD_VERSION and model['feature_version']=='combat_features_v6' and len(model['actor'][-1]['bias'])==20
    else:assert len(model['actor'][-1]['bias'])==8
    assert meta.get('feature_version','combat_features_v1')==model['feature_version'], 'Rollout feature version differs'
    assert sha(root/'rollout.jsonl')==meta['rollout_sha256']
    for path,digest in meta['source_sha256'].items():assert sha(pathlib.Path(path))==digest, f'Changed rollout input {path}'
    rows=[json.loads(s) for s in (root/'rollout.jsonl').read_text().splitlines()];assert len(rows)==meta['rows'] and len(rows)>1
    assert all(r['sample']['version']==meta['policy_version'] for r in rows)
    adv,ret=advantages(rows,config['gamma'],config['lambda'])
    data=(torch.tensor([r['features'] for r in rows],dtype=torch.float32),torch.tensor([r['sample']['latent'] for r in rows]),torch.tensor([float(r['sample']['attack']) for r in rows]),torch.tensor([r['sample']['vertical'] for r in rows],dtype=torch.long),torch.tensor([r['sample']['log_probability'] for r in rows]),torch.tensor(adv),torch.tensor(ret))
    assert torch.cuda.is_available(),'CUDA required'
    data=tuple(t.to('cuda') for t in data)
    x,z,attack,vertical,old,ad,returns=data
    weapon=torch.tensor([r['sample'].get('weapon',0) for r in rows],dtype=torch.long,device='cuda') if weapon_head else None
    if not weapon_head:assert all(r['sample'].get('weapon',0)==0 for r in rows)
    target=torch.tensor([r['sample'].get('target',0) for r in rows],dtype=torch.long,device='cuda') if target_head else None
    mode=torch.tensor([r['sample'].get('aim_mode',0) for r in rows],dtype=torch.long,device='cuda') if model.get('aim_mode_head') else None
    if mode is None:assert all(r['sample'].get('aim_mode',0)==0 for r in rows)
    if not target_head:assert all(r['sample'].get('target',0)==0 for r in rows)
    actor=network(model['actor']).to('cuda');value=network(model['value']).to('cuda');std=nn.Parameter(torch.tensor(model['log_std'],device='cuda'))
    checkpoint=None;consumed=[];updates=0;total_steps=0;resume_sha=sha(pathlib.Path(a.resume)) if a.resume else None
    if a.resume:checkpoint,consumed,updates,total_steps=restore_checkpoint(a.resume,model_path,config,actor,value,std,meta['rollout_sha256'])
    anchor_sha=sha(a.anchor_model) if a.anchor_model else None
    retention,retention_weight=retention_schedule(anchor_sha,a.retention_weight,a.retention_updates,updates,checkpoint.get('retention') if checkpoint else None,a.retention_mode)
    teacher,teacher_std=None,None
    if a.anchor_model:
        anchor=json.loads(a.anchor_model.read_text())
        if anchor['kind']!='combat_ppo_v1' or anchor['feature_version']!=model['feature_version']:raise ValueError('Anchor feature contract differs')
        with torch.no_grad():teacher=network(anchor['actor']).to('cuda')(x)
        teacher_std=torch.tensor(anchor['log_std'],device='cuda')
    bank=None;bank_tensors={}
    if a.retention_bank and checkpoint and checkpoint.get('retention') and not checkpoint.get('retention_bank'):raise ValueError('Cannot add a bank to an already started retention branch')
    bank_spec=pin_bank(sha(a.retention_bank) if a.retention_bank else None,a.bank_weight,checkpoint.get('retention_bank') if checkpoint else None)
    if a.retention_bank:
        if not a.anchor_model:raise ValueError('Retention bank requires an anchor')
        bank=validate_bank(read_bank(a.retention_bank),anchor_sha,model['feature_version'],x.shape[1],{r['seed'] for r in rows})
        with torch.no_grad():
            teacher_network=network(anchor['actor']).to('cuda')
            for split in ('train','validation'):
                bx=torch.tensor([r['features'] for r in bank[split]],dtype=torch.float32,device='cuda');bw=torch.tensor([r['weight'] for r in bank[split]],dtype=torch.float32,device='cuda')
                bank_tensors[split]=(bx,bw,teacher_network(bx))
    with torch.no_grad():
        lp,_=log_prob(actor,std,x,z,attack,vertical,weapon,target,mode);log_error=float((lp-old).abs().max());value_error=float((value(x).squeeze(1)-torch.tensor([r['sample']['value'] for r in rows],device='cuda')).abs().max())
        assert log_error<1e-3 and value_error<1e-4, (log_error,value_error)
    benchmarks={}
    for device in training_devices():
        probe=network(model['actor']).to(device);pstd=nn.Parameter(torch.tensor(model['log_std'],device=device));px,pz,pa,pv,po,pad,pret=[t.to(device) for t in data];opt=torch.optim.Adam(list(probe.parameters())+[pstd],lr=config['actor_lr'])
        def step():
            opt.zero_grad();plp,ent=log_prob(probe,pstd,px,pz,pa,pv,weapon,target,mode);ratio=(plp-po).exp();loss=-torch.minimum(ratio*pad,ratio.clamp(1-config['clip'],1+config['clip'])*pad).mean()-config['entropy']*ent.mean()
            if teacher is not None:loss=loss+retention_weight*anchor_kl(probe(px),pstd,teacher.to(device),teacher_std.to(device),weapon_mask=feature_mask(px) if weapon_head else None)
            if bank is not None:
                bx,bw,bt=bank_tensors['train'];loss=loss+a.bank_weight*anchor_kl(probe(bx.to(device)),pstd,bt.to(device),teacher_std.to(device),bw.to(device),feature_mask(bx.to(device)) if weapon_head else None)
            loss.backward();opt.step()
        for _ in range(10):step()
        if device=='cuda':torch.cuda.synchronize()
        start=time.perf_counter()
        for _ in range(30):step()
        if device=='cuda':torch.cuda.synchronize()
        benchmarks[device]=(time.perf_counter()-start)/30
    device=min(benchmarks,key=benchmarks.get);actor=actor.to(device);value=value.to(device);std=nn.Parameter(std.detach().to(device));x,z,attack,vertical,old,ad,returns=[t.to(device) for t in data]
    if teacher is not None:teacher,teacher_std=teacher.to(device),teacher_std.to(device)
    bank_tensors={s:tuple(t.to(device) for t in tensors) for s,tensors in bank_tensors.items()}
    def bank_loss(split='train'):
        if bank is None:return torch.zeros((),device=device)
        bx,bw,bt=bank_tensors[split];return anchor_kl(actor(bx),std,bt,teacher_std,bw,feature_mask(bx) if weapon_head else None)
    def retention_loss():
        return anchor_kl(actor(x),std,teacher,teacher_std,weapon_mask=feature_mask(x) if weapon_head else None) if teacher is not None else torch.zeros((),device=device)
    with torch.no_grad():initial_anchor_kl=float(retention_loss())
    with torch.no_grad():initial_bank_kl={s:float(bank_loss(s)) for s in bank_tensors}
    ad=(ad-ad.mean())/(ad.std(unbiased=False)+1e-8)
    actor_opt=torch.optim.Adam(list(actor.parameters())+[std],lr=config['actor_lr']);value_opt=torch.optim.Adam(value.parameters(),lr=config['value_lr'])
    if checkpoint:
        actor_opt.load_state_dict(checkpoint['actor_optimizer']);value_opt.load_state_dict(checkpoint['value_optimizer'])
        # Restore after benchmark, which constructs temporary networks.
        torch.set_rng_state(checkpoint['rng'].cpu())
        if torch.cuda.is_available() and checkpoint.get('cuda_rng') is not None:
            assert len(checkpoint['cuda_rng'])==torch.cuda.device_count(), 'CUDA RNG device count differs'
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint['cuda_rng']])
    done=0;backtracks=0;rejected_steps=0;actor_trials=[];start=time.perf_counter()
    for _ in range(config['actor_steps']):
        lp,entropy=log_prob(actor,std,x,z,attack,vertical,weapon,target,mode);ratio=(lp-old).exp();kl=((ratio-1)-(lp-old)).mean()
        if float(kl.detach())>config['target_kl']:break
        loss=-torch.minimum(ratio*ad,ratio.clamp(1-config['clip'],1+config['clip'])*ad).mean()-config['entropy']*entropy.mean()
        loss=loss+retention_weight*retention_loss()+a.bank_weight*bank_loss()
        assert torch.isfinite(loss);actor_opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(actor.parameters())+[std],config['max_grad_norm'])
        # Checking KL only before the next step can leave an excessive proposal
        # published. Restore the optimizer too and retry a smaller step.
        def candidate_objective():
                candidate_lp,candidate_entropy=log_prob(actor,std,x,z,attack,vertical,weapon,target,mode);candidate_ratio=(candidate_lp-old).exp()
                candidate_kl=((candidate_ratio-1)-(candidate_lp-old)).mean()
                candidate_loss=-torch.minimum(candidate_ratio*ad,candidate_ratio.clamp(1-config['clip'],1+config['clip'])*ad).mean()-config['entropy']*candidate_entropy.mean()
                candidate_loss=candidate_loss+retention_weight*retention_loss()+a.bank_weight*bank_loss()
                return candidate_loss,candidate_kl
        trial=guarded_actor_step(list(actor.parameters())+[std],actor_opt,candidate_objective,loss.detach(),config['actor_lr'],config['target_kl'],lambda:std.clamp_(-8,1))
        actor_trials.append(trial)
        if not trial['accepted']:rejected_steps+=1;break
        backtracks+=trial['retry']
        done+=1
    for _ in range(config['value_steps']):
        vloss=((value(x).squeeze(1)-returns)**2).mean();assert torch.isfinite(vloss);value_opt.zero_grad();vloss.backward();nn.utils.clip_grad_norm_(value.parameters(),config['max_grad_norm']);value_opt.step()
    if device=='cuda':torch.cuda.synchronize()
    with torch.no_grad():
        lp,_=log_prob(actor,std,x,z,attack,vertical,weapon,target,mode);ratio=(lp-old).exp();final_kl=float(((ratio-1)-(lp-old)).mean());value_loss=float(((value(x).squeeze(1)-returns)**2).mean())
    updated={**model,'actor':layers(actor),'value':layers(value),'log_std':std.detach().cpu().tolist()}
    with torch.no_grad():final_anchor_kl=float(retention_loss())
    if a.anchor_model and sha(a.anchor_model)!=anchor_sha:raise ValueError('Anchor changed during update')
    if bank is not None:
        if sha(a.retention_bank)!=bank_spec['sha256']:raise ValueError('Bank changed during update')
        validate_bank(bank,anchor_sha,model['feature_version'],x.shape[1],{r['seed'] for r in rows})
    with torch.no_grad():final_bank_kl={s:float(bank_loss(s)) for s in bank_tensors}
    (out/'weights.json').write_text(json.dumps(updated,allow_nan=False))
    # Save optimizer/RNG with the checkpoint; future rollout still needs fresh seeds.
    checkpoint_output={'version':'combat_ppo_checkpoint_v2','weights_sha256':sha(out/'weights.json'),'consumed_rollouts':consumed+[meta['rollout_sha256']],'updates_completed':updates+1,'total_actor_steps':total_steps+done,'actor':actor.state_dict(),'value':value.state_dict(),'log_std':std.detach(),'actor_optimizer':actor_opt.state_dict(),'value_optimizer':value_opt.state_dict(),'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,'config':config,'retention':retention}
    if bank_spec is not None:checkpoint_output['retention_bank']=bank_spec
    torch.save(checkpoint_output,out/'checkpoint.pt')
    assert math.isfinite(final_kl) and final_kl<=config['target_kl']+1e-6
    report={'scope':'One PPO-Clip update on fresh stochastic first-life transitions; not combat acceptance or weapon learning','torch':torch.__version__,'trainer_sha256':sha(pathlib.Path(__file__)),'device':device,'benchmark_seconds_per_step':benchmarks,'rows':len(rows),'behavior_policy':meta['policy_version'],'behavior_sha256':sha(model_path),'rollout_sha256':meta['rollout_sha256'],'config':config,'config_sha256':sha(pathlib.Path(a.config)),'old_log_probability_max_error':log_error,'old_value_max_error':value_error,'actor_steps':done,'backtracks':backtracks,'rejected_steps':rejected_steps,'final_approx_kl':final_kl,'value_mse':value_loss,'seconds':time.perf_counter()-start,'weights_sha256':sha(out/'weights.json')}
    report.update({'resume_sha256':resume_sha,'updates_completed':updates+1,'total_actor_steps':total_steps+done,'consumed_rollouts':consumed+[meta['rollout_sha256']]})
    report.update(retention=retention,retention_weight=retention_weight,initial_anchor_kl=initial_anchor_kl,final_anchor_kl=final_anchor_kl)
    if bank_spec is not None:report.update(retention_bank=bank_spec,bank_kl_before=initial_bank_kl,bank_kl_after=final_bank_kl,bank_rows={s:len(bank[s]) for s in bank_tensors})
    if bank_spec is not None:report['bank_helper_sha256']=sha(pathlib.Path(__file__).with_name('combat_retention_bank.py'))
    report.update(weapon_head=weapon_head,actor_parameters=sum(p.numel() for p in actor.parameters())+std.numel(),critic_parameters=sum(p.numel() for p in value.parameters()))
    report.update(actor_step_guard='uphill_first_moment_v1',actor_trials=actor_trials,direction_fallbacks=sum(t['direction_fallback'] for t in actor_trials))
    if a.resume:assert sha(pathlib.Path(a.resume))==report['resume_sha256']
    (out/'trainer.py').write_bytes(pathlib.Path(__file__).read_bytes())
    assert sha(model_path)==meta['model_sha256'] and sha(root/'rollout.jsonl')==meta['rollout_sha256']
    for path,digest in meta['source_sha256'].items():assert sha(pathlib.Path(path))==digest
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
