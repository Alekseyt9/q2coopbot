"""Seal a zero fire-branch checkpoint migration on actual CUDA TRAIN histories."""
import argparse
import copy
import json
import os
from pathlib import Path
from evaluate_registered_combat_models import read,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','data','out'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();assert not args.out.exists()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    from ppo_combat import torch
    assert torch.cuda.is_available()
    torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    from combat_conditional_fire import initialize,VERSION
    from migrate_combat_navigation_cuda import branch
    from ppo_recurrent import prepare_context,sequence,durable_json,durable_write
    from combat_precision_head import probabilities
    from combat_sequence_retention import policy_kl,CONDITIONAL_FIRE_EXPECTATION
    from combat_weapon_state_residual import optimizer
    from fork_conditional_weapon_exploration_cuda import equal_tree
    pins={};seal=read(args.source/'complete.json')
    for filename,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:
        path=args.source/filename;assert sha(path)==seal[key];pins[str(path.resolve())]=sha(path)
    model=read(args.source/'weights.json');assert model.get('motor_residual') and not model.get('conditional_fire')
    rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state_all()
    updated=initialize(model)
    assert torch.equal(rng,torch.get_rng_state()) and equal_tree(gpu_rng,torch.cuda.get_rng_state_all())
    cp=torch.load(args.source/'checkpoint.pt',map_location='cuda',weights_only=True)
    before,after=branch(model,'actor'),branch(updated,'actor')
    assert equal_tree(before.state_dict(),cp['actor'])
    for name,value in before.state_dict().items():assert torch.equal(value,after.state_dict()[name]),name
    assert equal_tree(branch(updated,'value').state_dict(),cp['value'])
    assert torch.equal(torch.tensor(model['log_std']),cp['log_std'])
    meta=read(args.data/'report.json');assert meta['model_sha256']==seal['weights_sha256']
    pins[str((args.data/'report.json').resolve())]=sha(args.data/'report.json')
    for filename,key in [('rollout.jsonl','rollout_sha256'),('sequence.jsonl','sequence_sha256')]:
        assert sha(args.data/filename)==meta[key];pins[str((args.data/filename).resolve())]=meta[key]
    rows=[json.loads(s) for s in (args.data/'rollout.jsonl').read_text().splitlines()]
    context=[json.loads(s) for s in (args.data/'sequence.jsonl').read_text().splitlines()]
    assert len(rows)==meta['rows'] and len(context)==meta['sequence_rows']
    prepared=prepare_context(context,rows,'cuda')
    x=torch.tensor([r['features'] for r in rows]);z=torch.tensor([r['sample']['latent'] for r in rows])
    categorical=lambda key:torch.tensor([r['sample'].get(key,0) for r in rows],dtype=torch.long)
    target,mode,vertical,weapon=(categorical(key) for key in ('target','aim_mode','vertical','weapon'))
    attack=torch.tensor([float(r['sample']['attack']) for r in rows])
    std=torch.nn.Parameter(torch.tensor(model['log_std']));new_std=torch.nn.Parameter(std.detach().clone())
    action_context=(target,mode,z,vertical,weapon)
    def likelihood(actor,raw,scale):
        return probabilities(raw,scale,z,attack,vertical,weapon,x,target,mode,
            actor.sampled_motion_aim_branch,actor.motor_branch,actor.conditional_fire_branch)[0]
    with torch.no_grad():
        old,old_memory=sequence(before,prepared,check_states=True)
        new,new_memory=sequence(after,prepared,check_states=True)
        assert torch.equal(old,new)
        old_lp,new_lp=likelihood(before,old,std),likelihood(after,new,new_std)
        assert torch.equal(old_lp,new_lp)
        native_error=float((old_lp-torch.tensor([r['sample']['log_probability'] for r in rows])).abs().max())
        assert native_error<.003 and max(old_memory,new_memory)<2e-5
        def retained_kl():
            return policy_kl(new,new_std,old,std,x,after.sampled_motion_aim_branch,
                before.sampled_motion_aim_branch,z[:,:2].detach(),after.motor_branch,before.motor_branch,
                after.conditional_fire_branch,None,action_context)
        initial_kl=retained_kl();assert float(initial_kl.abs().max())<1e-9
        after.conditional_fire_branch[-1].bias.fill_(.1)
        changed=likelihood(after,new,new_std)
        d=torch.distributions
        expected_lp=d.Bernoulli(logits=new[:,4]+.1).log_prob(attack)-d.Bernoulli(logits=new[:,4]).log_prob(attack)
        # Compare complete likelihoods to avoid cancellation of very rare-action LPs.
        assert torch.allclose(changed,old_lp+expected_lp,atol=2e-5,rtol=1e-6)
        observed=retained_kl()
        expected_kl=d.kl_divergence(d.Bernoulli(logits=old[:,4].double()),
            d.Bernoulli(logits=new[:,4].double()+float(after.conditional_fire_branch[-1].bias)))
        kl_error=float((observed-expected_kl).abs().max());assert kl_error<1e-8
    for name,p in after.named_parameters():p.requires_grad_(name.startswith('conditional_fire_branch.'))
    (-likelihood(after,new.detach(),new_std.detach()).mean()).backward()
    gradient=max(float(p.grad.abs().max()) for p in after.conditional_fire_branch.parameters() if p.grad is not None)
    assert gradient>0
    after.zero_grad(set_to_none=True)
    with torch.no_grad():after.conditional_fire_branch[-1].bias.zero_()
    for p in after.parameters():p.requires_grad_(True)
    rates=cp['actor_training']
    old_opt=optimizer(before,std,cp['config']['actor_lr'],rates['weapon_state_residual']['learning_rate'],rates['motor_residual']['learning_rate'])
    new_opt=optimizer(after,new_std,cp['config']['actor_lr'],rates['weapon_state_residual']['learning_rate'],rates['motor_residual']['learning_rate'])
    old_opt.load_state_dict(copy.deepcopy(cp['actor_optimizer']))
    old_named={**dict(before.named_parameters()),'@std':std};new_named={**dict(after.named_parameters()),'@std':new_std}
    for name,p in old_named.items():new_opt.state[new_named[name]]=copy.deepcopy(old_opt.state.get(p,{}))
    for new_group,old_group in zip(new_opt.param_groups,old_opt.param_groups):
        for key,value in old_group.items():
            if key!='params':new_group[key]=copy.deepcopy(value)
    saved=new_opt.state_dict()
    assert len(cp['actor_optimizer']['param_groups'])==3 and len(saved['param_groups'])==4
    for key,value in cp['actor_optimizer']['state'].items():assert equal_tree(value,saved['state'][key])
    for a,b in zip(cp['actor_optimizer']['param_groups'],saved['param_groups']):assert equal_tree(a,b)
    assert all(not new_opt.state.get(p) for p in after.conditional_fire_branch.parameters())
    receipt=dict(state='complete',device='cuda',version=VERSION,source_weights_sha256=seal['weights_sha256'],
        source_checkpoint_sha256=seal['checkpoint_sha256'],zero_raw_and_likelihood_exact=True,history_rows=len(rows),
        memory_max_error=max(old_memory,new_memory),native_log_probability_max_error=native_error,
        zero_conditional_kl_max=float(initial_kl.abs().max()),nonzero_conditional_kl_max_error=kl_error,
        conditional_retention_expectation=CONDITIONAL_FIRE_EXPECTATION,fire_gradient_max=gradient,
        initialization_rng_preserved=True,old_optimizer_ids_and_states_exact=True,new_optimizer_group_empty=True,
        previous_sequence_retention=cp.get('sequence_retention'),
        new_parameters=sum(p.numel() for p in after.conditional_fire_branch.parameters()),training_performed=False,promotion=False,
        inputs=[dict(path=p,sha256=d) for p,d in pins.items()],script_sha256=sha(__file__),
        scope='CUDA migration on existing own-source TRAIN histories, not training or fresh native compatibility. Old source-helper digests are historical; fresh captures are required before fitting.')
    args.out.mkdir(parents=True);durable_json(args.out/'weights.json',updated)
    # Deserialize CPU byte RNG without executing any neural computation on CPU.
    fork=torch.load(args.source/'checkpoint.pt',map_location='cpu',weights_only=True)
    fork.update(actor=after.state_dict(),actor_optimizer=saved,weights_sha256=sha(args.out/'weights.json'),sequence_retention=None)
    fork['actor_training']={**fork['actor_training'],'conditional_fire':dict(version=VERSION,learning_rate=.0003)}
    fork['model_migrations']=fork.get('model_migrations',[])+[receipt]
    durable_write(args.out/'checkpoint.pt',lambda f:torch.save(fork,f))
    reloaded=torch.load(args.out/'checkpoint.pt',map_location='cuda',weights_only=True)
    for key in cp:
        if key not in ('actor','actor_optimizer','weights_sha256','sequence_retention','actor_training','model_migrations'):
            assert equal_tree(cp[key],reloaded[key]),key
    assert equal_tree(branch(updated,'actor').state_dict(),reloaded['actor'])
    resumed=optimizer(branch(updated,'actor'),torch.nn.Parameter(new_std.detach().clone()),cp['config']['actor_lr'],
        rates['weapon_state_residual']['learning_rate'],rates['motor_residual']['learning_rate'])
    resumed.load_state_dict(copy.deepcopy(reloaded['actor_optimizer']))
    assert equal_tree(resumed.state_dict(),saved)
    for path,digest in pins.items():assert sha(Path(path))==digest
    durable_json(args.out/'report.json',receipt)
    durable_json(args.out/'complete.json',dict(version='combat_update_complete_v1',weights_sha256=sha(args.out/'weights.json'),checkpoint_sha256=sha(args.out/'checkpoint.pt'),report_sha256=sha(args.out/'report.json')))
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('inputs','scope','previous_sequence_retention')}))


if __name__=='__main__':main()
