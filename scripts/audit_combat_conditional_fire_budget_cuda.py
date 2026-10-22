"""Audit the longer fire-only fit and compare exact initial parameter gradients."""
import argparse
import json
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('source','update','boundary','first-fit','corpus','relabel','out'):
        ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();assert not args.out.exists()
    from ppo_combat import torch,training_devices,advantages
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    from fork_conditional_weapon_exploration_cuda import equal_tree
    from migrate_combat_navigation_cuda import branch
    from ppo_recurrent import prepare_context,sequence
    from combat_conditional_fire import logits
    pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();digest=sha(path);assert expected is None or digest==expected
        pins[str(path)]=digest;return path
    for directory in (args.source,args.update,args.first_fit):
        seal=read(pin(directory/'complete.json'))
        for name,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:pin(directory/name,seal[key])
    source=torch.load(args.source/'checkpoint.pt',map_location='cuda',weights_only=True)
    initial=torch.load(pin(args.boundary),map_location='cuda',weights_only=True)
    after=torch.load(args.update/'checkpoint.pt',map_location='cuda',weights_only=True)
    first=torch.load(args.first_fit/'checkpoint.pt',map_location='cuda',weights_only=True)
    report=read(args.update/'report.json')
    assert source['config']['actor_steps']==300 and 0<report['actor_steps']<=300
    assert after['config']==source['config'] and report['actor_training']['scope']=='conditional-fire'
    assert report['trainable_actor_parameters']==6369 and sha(args.boundary)==report['initial_boundary_sha256']
    for key in ('actor','value','log_std','actor_optimizer','value_optimizer','rng','cuda_rng'):
        assert equal_tree(source[key],initial[key]),key
    for name,value in source['actor'].items():
        if not name.startswith('conditional_fire_branch.'):assert torch.equal(value,after['actor'][name]),name
    assert torch.equal(source['log_std'],after['log_std'])
    for group in source['actor_optimizer']['param_groups'][:-1]:
        for key in group['params']:
            assert equal_tree(source['actor_optimizer']['state'].get(key,{}),after['actor_optimizer']['state'].get(key,{})),key
    for key in source['actor_optimizer']['param_groups'][-1]['params']:
        assert not source['actor_optimizer']['state'].get(key)
        assert float(after['actor_optimizer']['state'][key]['step'])==report['actor_steps']
    assert equal_tree(first['value'],after['value']) and equal_tree(first['value_optimizer'],after['value_optimizer'])
    models=[read(d/'weights.json') for d in (args.source,args.first_fit,args.update)]
    actors=[branch(m,'actor') for m in models]
    assert equal_tree(actors[2].state_dict(),after['actor'])
    assert equal_tree(branch(models[2],'value').state_dict(),after['value'])
    data=args.corpus/'processing-train/merged';meta=read(pin(data/'report.json'))
    pin(data/'rollout.jsonl',meta['rollout_sha256']);pin(data/'sequence.jsonl',meta['sequence_sha256'])
    assert report['rollout_sha256']==meta['rollout_sha256'] and meta['rollout_sha256'] not in source['consumed_rollouts']
    assert after['consumed_rollouts']==source['consumed_rollouts']+[meta['rollout_sha256']]
    assert after['updates_completed']==source['updates_completed']+1
    assert after['total_actor_steps']==source['total_actor_steps']+report['actor_steps']
    rows=[json.loads(s) for s in (data/'rollout.jsonl').read_text().splitlines()]
    context=[json.loads(s) for s in (data/'sequence.jsonl').read_text().splitlines()]
    relabel_seal=read(pin(args.relabel/'complete.json'));pin(args.relabel/'report.json',relabel_seal['report_sha256'])
    assert not relabel_seal['ppo_ready'] and not relabel_seal['training_performed']
    changes=[json.loads(s) for s in pin(args.relabel/'reward-deltas.jsonl').read_text().splitlines()]
    updates={(v['seed'],v['index']):v for v in changes};assert len(updates)==len(rows)
    prepared=prepare_context(context,rows,'cuda');x=torch.tensor([v['features'] for v in rows]);z=torch.tensor([v['sample']['latent'] for v in rows])
    cat=lambda key:torch.tensor([v['sample'].get(key,0) for v in rows],dtype=torch.long)
    target,mode,vertical,weapon=(cat(k) for k in ('target','aim_mode','vertical','weapon'))
    attack=torch.tensor([float(v['sample']['attack']) for v in rows])
    misses=torch.tensor([updates[(v['seed'],v['index'])]['misses'] for v in rows])
    with torch.no_grad():
        raw,_=sequence(actors[0],prepared,check_states=True)
        probabilities=[logits(a.conditional_fire_branch,x,raw,target,mode,z,vertical,weapon).sigmoid() for a in actors]
        masks=dict(all=torch.ones_like(target,dtype=torch.bool),no_selected_target=target==0,native_mg_miss=misses>0,no_target_mg_miss=(target==0)&(misses>0))
        groups={name:dict(rows=int(mask.sum()),p_start=float(probabilities[0][mask].mean()),p_10_steps=float(probabilities[1][mask].mean()),
            p_long_fit=float(probabilities[2][mask].mean())) for name,mask in masks.items() if bool(mask.any())}
        old_fire=torch.distributions.Bernoulli(logits=logits(actors[0].conditional_fire_branch,x,raw,target,mode,z,vertical,weapon)).log_prob(attack)
    gradients={};gradient_stats={};params=list(actors[0].conditional_fire_branch.parameters())
    for name in ('control','cost'):
        altered=rows if name=='control' else [{**v,'reward':updates[(v['seed'],v['index'])]['reward']} for v in rows]
        adv,_=advantages(altered,source['config']['gamma'],source['config']['lambda']);adv=torch.tensor(adv)
        adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
        fire=torch.distributions.Bernoulli(logits=logits(actors[0].conditional_fire_branch,x,raw,target,mode,z,vertical,weapon))
        ratio=(fire.log_prob(attack)-old_fire).exp()
        assert float((ratio-1).abs().max().detach())<1e-5
        # Other heads are frozen and cancel in the ratio. Initial teacher=current
        # conditional retention has zero gradient; entropy of other heads is fixed.
        loss=-torch.minimum(ratio*adv,ratio.clamp(1-source['config']['clip'],1+source['config']['clip'])*adv).mean()-source['config']['entropy']*fire.entropy().mean()
        values=torch.autograd.grad(loss,params)
        vector=torch.cat([v.reshape(-1) for v in values]);gradients[name]=vector
        gradient_stats[name]=dict(norm=float(vector.norm()),nonzero_parameters=int((vector!=0).sum()),parameters=vector.numel())
    cosine=float(torch.nn.functional.cosine_similarity(gradients['control'],gradients['cost'],dim=0))
    for path,digest in pins.items():assert sha(Path(path))==digest
    save(args.out,dict(state='complete',device='cuda',actor_steps=report['actor_steps'],rows=len(rows),
        initial_resume_exact=True,frozen_actor_std_adam_exact=True,runtime_checkpoint_exact=True,rollout_consumed_once=True,
        critic_and_optimizer_same_as_10_step_fit=True,groups=groups,initial_parameter_gradients=gradient_stats,
        control_cost_gradient_cosine=cosine,inputs=pins,script_sha256=sha(__file__),training_performed=False,promotion=False,
        scope='Train-only audit of an alternative unconsumed-source budget fork; fixed actual behavior context. Exact fire-branch parameter gradients include PPO clipping/entropy at the initial policy. No native quality or reward-fork training acceptance.'))
    print(json.dumps(dict(actor_steps=report['actor_steps'],groups=groups,initial_parameter_gradients=gradient_stats,gradient_cosine=cosine)))


if __name__=='__main__':main()
