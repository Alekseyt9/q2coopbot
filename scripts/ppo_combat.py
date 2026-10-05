"""Offline PPO-Clip update. Go owns observations, sampling, commands and harness."""
import argparse, copy, json, math, pathlib, time, sys
sys.pycache_prefix=str(pathlib.Path(__file__).resolve().parents[1]/'workspace'/'build'/'python-cache')
# Shared F: cache configuration is set before importing torch.
from train_combat_bc import torch, nn, sha

def network(layers):
    blocks=[]
    for i,l in enumerate(layers):
        linear=nn.Linear(len(l['weight'][0]),len(l['bias']))
        with torch.no_grad():linear.weight.copy_(torch.tensor(l['weight']));linear.bias.copy_(torch.tensor(l['bias']))
        blocks.append(linear)
        if i<2:blocks.append(nn.ReLU())
    return nn.Sequential(*blocks)

def layers(model):return [{'weight':l.weight.detach().cpu().tolist(),'bias':l.bias.detach().cpu().tolist()} for l in model if isinstance(l,nn.Linear)]

def log_prob(actor,std,x,z,attack,vertical):
    raw=actor(x);normal=torch.distributions.Normal(raw[:,:4],std.exp())
    bern=torch.distributions.Bernoulli(logits=raw[:,4]);cat=torch.distributions.Categorical(logits=raw[:,5:])
    # PPO uses pre-tanh latent likelihood; fixed map Jacobian cancels in ratio.
    lp=normal.log_prob(z).sum(1)+bern.log_prob(attack)+cat.log_prob(vertical)
    entropy=normal.entropy().sum(1)+bern.entropy()+cat.entropy()
    return lp,entropy

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
    ap=argparse.ArgumentParser();ap.add_argument('--init-bc');ap.add_argument('--model');ap.add_argument('--data');ap.add_argument('--resume');ap.add_argument('--config',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
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
    validate_objective(config,meta)
    assert model['kind']=='combat_ppo_v1' and not model['deterministic'] and meta['version']=='combat_ppo_rollout_v1' and sha(model_path)==meta['model_sha256']
    assert meta.get('feature_version','combat_features_v1')==model['feature_version'], 'Rollout feature version differs'
    assert sha(root/'rollout.jsonl')==meta['rollout_sha256']
    for path,digest in meta['source_sha256'].items():assert sha(pathlib.Path(path))==digest, f'Changed rollout input {path}'
    rows=[json.loads(s) for s in (root/'rollout.jsonl').read_text().splitlines()];assert len(rows)==meta['rows'] and len(rows)>1
    assert all(r['sample']['version']==meta['policy_version'] for r in rows)
    adv,ret=advantages(rows,config['gamma'],config['lambda'])
    data=(torch.tensor([r['features'] for r in rows],dtype=torch.float32),torch.tensor([r['sample']['latent'] for r in rows]),torch.tensor([float(r['sample']['attack']) for r in rows]),torch.tensor([r['sample']['vertical'] for r in rows],dtype=torch.long),torch.tensor([r['sample']['log_probability'] for r in rows]),torch.tensor(adv),torch.tensor(ret))
    x,z,attack,vertical,old,ad,returns=data
    actor=network(model['actor']);value=network(model['value']);std=nn.Parameter(torch.tensor(model['log_std']))
    checkpoint=None;consumed=[];updates=0;total_steps=0;resume_sha=sha(pathlib.Path(a.resume)) if a.resume else None
    if a.resume:checkpoint,consumed,updates,total_steps=restore_checkpoint(a.resume,model_path,config,actor,value,std,meta['rollout_sha256'])
    with torch.no_grad():
        lp,_=log_prob(actor,std,x,z,attack,vertical);log_error=float((lp-old).abs().max());value_error=float((value(x).squeeze(1)-torch.tensor([r['sample']['value'] for r in rows])).abs().max())
        assert log_error<1e-3 and value_error<1e-4, (log_error,value_error)
    benchmarks={}
    for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
        probe=network(model['actor']).to(device);pstd=nn.Parameter(torch.tensor(model['log_std'],device=device));px,pz,pa,pv,po,pad,pret=[t.to(device) for t in data];opt=torch.optim.Adam(list(probe.parameters())+[pstd],lr=config['actor_lr'])
        def step():
            opt.zero_grad();plp,ent=log_prob(probe,pstd,px,pz,pa,pv);ratio=(plp-po).exp();loss=-torch.minimum(ratio*pad,ratio.clamp(1-config['clip'],1+config['clip'])*pad).mean()-config['entropy']*ent.mean();loss.backward();opt.step()
        for _ in range(10):step()
        if device=='cuda':torch.cuda.synchronize()
        start=time.perf_counter()
        for _ in range(30):step()
        if device=='cuda':torch.cuda.synchronize()
        benchmarks[device]=(time.perf_counter()-start)/30
    device=min(benchmarks,key=benchmarks.get);actor=actor.to(device);value=value.to(device);std=nn.Parameter(std.detach().to(device));x,z,attack,vertical,old,ad,returns=[t.to(device) for t in data]
    ad=(ad-ad.mean())/(ad.std(unbiased=False)+1e-8)
    actor_opt=torch.optim.Adam(list(actor.parameters())+[std],lr=config['actor_lr']);value_opt=torch.optim.Adam(value.parameters(),lr=config['value_lr'])
    if checkpoint:
        actor_opt.load_state_dict(checkpoint['actor_optimizer']);value_opt.load_state_dict(checkpoint['value_optimizer'])
        # Restore after benchmark, which constructs temporary networks.
        torch.set_rng_state(checkpoint['rng'].cpu())
        if torch.cuda.is_available() and checkpoint.get('cuda_rng') is not None:
            assert len(checkpoint['cuda_rng'])==torch.cuda.device_count(), 'CUDA RNG device count differs'
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint['cuda_rng']])
    done=0;backtracks=0;rejected_steps=0;start=time.perf_counter()
    for _ in range(config['actor_steps']):
        lp,entropy=log_prob(actor,std,x,z,attack,vertical);ratio=(lp-old).exp();kl=((ratio-1)-(lp-old)).mean()
        if float(kl.detach())>config['target_kl']:break
        loss=-torch.minimum(ratio*ad,ratio.clamp(1-config['clip'],1+config['clip'])*ad).mean()-config['entropy']*entropy.mean()
        assert torch.isfinite(loss);actor_opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(actor.parameters())+[std],config['max_grad_norm'])
        saved_actor=copy.deepcopy(actor.state_dict());saved_std=std.detach().clone();saved_opt=copy.deepcopy(actor_opt.state_dict());accepted=False
        # Checking KL only before the next step can leave an excessive proposal
        # published. Restore the optimizer too and retry a smaller step.
        for retry in range(13):
            actor.load_state_dict(saved_actor)
            with torch.no_grad():std.copy_(saved_std)
            actor_opt.load_state_dict(saved_opt)
            for group in actor_opt.param_groups:group['lr']=config['actor_lr']*(.5**retry)
            actor_opt.step()
            with torch.no_grad():
                std.clamp_(-8,1);candidate_lp,candidate_entropy=log_prob(actor,std,x,z,attack,vertical);candidate_ratio=(candidate_lp-old).exp()
                candidate_kl=((candidate_ratio-1)-(candidate_lp-old)).mean()
                candidate_loss=-torch.minimum(candidate_ratio*ad,candidate_ratio.clamp(1-config['clip'],1+config['clip'])*ad).mean()-config['entropy']*candidate_entropy.mean()
                accepted=bool(torch.isfinite(candidate_kl) and torch.isfinite(candidate_loss) and candidate_kl<=config['target_kl'] and candidate_loss<=loss.detach()+1e-7)
            if accepted:backtracks+=retry;break
        if not accepted:
            actor.load_state_dict(saved_actor)
            with torch.no_grad():std.copy_(saved_std)
            actor_opt.load_state_dict(saved_opt);rejected_steps+=1;break
        done+=1
    for _ in range(config['value_steps']):
        vloss=((value(x).squeeze(1)-returns)**2).mean();assert torch.isfinite(vloss);value_opt.zero_grad();vloss.backward();nn.utils.clip_grad_norm_(value.parameters(),config['max_grad_norm']);value_opt.step()
    if device=='cuda':torch.cuda.synchronize()
    with torch.no_grad():
        lp,_=log_prob(actor,std,x,z,attack,vertical);ratio=(lp-old).exp();final_kl=float(((ratio-1)-(lp-old)).mean());value_loss=float(((value(x).squeeze(1)-returns)**2).mean())
    updated={**model,'actor':layers(actor),'value':layers(value),'log_std':std.detach().cpu().tolist()}
    (out/'weights.json').write_text(json.dumps(updated,allow_nan=False))
    # Save optimizer/RNG with the checkpoint; future rollout still needs fresh seeds.
    torch.save({'version':'combat_ppo_checkpoint_v2','weights_sha256':sha(out/'weights.json'),'consumed_rollouts':consumed+[meta['rollout_sha256']],'updates_completed':updates+1,'total_actor_steps':total_steps+done,'actor':actor.state_dict(),'value':value.state_dict(),'log_std':std.detach(),'actor_optimizer':actor_opt.state_dict(),'value_optimizer':value_opt.state_dict(),'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,'config':config},out/'checkpoint.pt')
    assert math.isfinite(final_kl) and final_kl<=config['target_kl']+1e-6
    report={'scope':'One PPO-Clip update on fresh stochastic first-life transitions; not combat acceptance or weapon learning','torch':torch.__version__,'trainer_sha256':sha(pathlib.Path(__file__)),'device':device,'benchmark_seconds_per_step':benchmarks,'rows':len(rows),'behavior_policy':meta['policy_version'],'behavior_sha256':sha(model_path),'rollout_sha256':meta['rollout_sha256'],'config':config,'config_sha256':sha(pathlib.Path(a.config)),'old_log_probability_max_error':log_error,'old_value_max_error':value_error,'actor_steps':done,'backtracks':backtracks,'rejected_steps':rejected_steps,'final_approx_kl':final_kl,'value_mse':value_loss,'seconds':time.perf_counter()-start,'weights_sha256':sha(out/'weights.json')}
    report.update({'resume_sha256':resume_sha,'updates_completed':updates+1,'total_actor_steps':total_steps+done,'consumed_rollouts':consumed+[meta['rollout_sha256']]})
    if a.resume:assert sha(pathlib.Path(a.resume))==report['resume_sha256']
    (out/'trainer.py').write_bytes(pathlib.Path(__file__).read_bytes())
    assert sha(model_path)==meta['model_sha256'] and sha(root/'rollout.jsonl')==meta['rollout_sha256']
    for path,digest in meta['source_sha256'].items():assert sha(pathlib.Path(path))==digest
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
