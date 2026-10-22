"""Zero motor migration, nonzero reference/gradient tests, preserved CUDA state."""
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
    import torch
    assert torch.cuda.is_available()
    torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    from combat_motor_residual import initialize,VERSION,inputs
    from migrate_combat_navigation_cuda import branch
    from ppo_recurrent import prepare_context,sequence,durable_json,durable_write
    from combat_precision_head import probabilities,selected_means
    from combat_sequence_retention import policy_kl,normal_kl
    from combat_weapon_state_residual import optimizer
    from fork_conditional_weapon_exploration_cuda import equal_tree
    pins={};seal=read(args.source/'complete.json')
    for filename,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:
        path=args.source/filename;assert sha(path)==seal[key];pins[str(path.resolve())]=sha(path)
    model=read(args.source/'weights.json');rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state_all()
    updated=initialize(model)
    assert torch.equal(rng,torch.get_rng_state()) and equal_tree(gpu_rng,torch.cuda.get_rng_state_all())
    cp=torch.load(args.source/'checkpoint.pt',map_location='cuda',weights_only=True)
    before,after=branch(model,'actor'),branch(updated,'actor')
    assert equal_tree(before.state_dict(),cp['actor'])
    for name,value in before.state_dict().items():assert torch.equal(value,after.state_dict()[name])
    assert equal_tree(branch(updated,'value').state_dict(),cp['value'])
    meta=read(args.data/'report.json');assert meta['model_sha256']==seal['weights_sha256']
    for filename,key in [('rollout.jsonl','rollout_sha256'),('sequence.jsonl','sequence_sha256')]:
        assert sha(args.data/filename)==meta[key];pins[str((args.data/filename).resolve())]=meta[key]
    rows=[json.loads(s) for s in (args.data/'rollout.jsonl').read_text().splitlines()]
    context=[json.loads(s) for s in (args.data/'sequence.jsonl').read_text().splitlines()]
    prepared=prepare_context(context,rows,'cuda');x=torch.tensor([r['features'] for r in rows]);z=torch.tensor([r['sample']['latent'] for r in rows])
    target=torch.tensor([r['sample'].get('target',0) for r in rows],dtype=torch.long)
    mode=torch.tensor([r['sample'].get('aim_mode',0) for r in rows],dtype=torch.long)
    attack=torch.tensor([float(r['sample']['attack']) for r in rows]);vertical=torch.tensor([r['sample']['vertical'] for r in rows],dtype=torch.long)
    weapon=torch.tensor([r['sample'].get('weapon',0) for r in rows],dtype=torch.long);std=torch.nn.Parameter(torch.tensor(model['log_std']));new_std=torch.nn.Parameter(std.detach().clone())
    # Layout parity is a pure index mapping, evaluated here on CUDA.
    markers=torch.arange(1147,dtype=torch.float32)[None,:];selected=inputs(markers)
    assert selected.shape==(1,756) and selected[0,[168,169,488,489,755]].tolist()==[168,466,785,854,1120]
    with torch.no_grad():
        old,old_memory=sequence(before,prepared,check_states=True);new,new_memory=sequence(after,prepared,check_states=True)
        assert torch.equal(old,new)
        old_lp,_=probabilities(old,std,z,attack,vertical,weapon,x,target,mode,before.sampled_motion_aim_branch)
        new_lp,_=probabilities(new,new_std,z,attack,vertical,weapon,x,target,mode,after.sampled_motion_aim_branch,after.motor_branch)
        assert torch.equal(old_lp,new_lp)
        native_error=float((old_lp-torch.tensor([r['sample']['log_probability'] for r in rows])).abs().max());assert native_error<.003
        old_means=selected_means(old,target,mode,x,z[:,:2],before.sampled_motion_aim_branch,vertical)
        after.motor_branch[-1].bias.copy_(torch.tensor([.015,-.015]))
        perturbed,_=sequence(after,prepared)
        means=selected_means(perturbed,target,mode,x,z[:,:2],after.sampled_motion_aim_branch,vertical,after.motor_branch)
        aim_error=float((means[:,2:]-old_means[:,2:]).abs().max());assert aim_error<1e-5
        assert torch.equal(perturbed[:,2:],old[:,2:])
        altered=selected_means(perturbed,target,mode,x,z[:,:2]+.15,after.sampled_motion_aim_branch,vertical,after.motor_branch)
        actual_move_response=float((altered[:,2:]-means[:,2:]).abs().max());assert actual_move_response>1e-5
        total=policy_kl(perturbed,new_std,old,std,x,after.sampled_motion_aim_branch,before.sampled_motion_aim_branch,z[:,:2].detach(),after.motor_branch,None)
        motor_kl=normal_kl(old[:,:2].double(),perturbed[:,:2].double(),std[:2].double(),new_std[:2].double())
        residual_kl_error=float((total-motor_kl).abs().max());assert residual_kl_error<1e-6
    # Nonzero motor must train, while fixed-action aim cannot backprop through it.
    for name,param in after.named_parameters():param.requires_grad_(name.startswith('motor_branch.'))
    raw,_=sequence(after,prepared)
    aim=selected_means(raw,target,mode,x,z[:,:2],after.sampled_motion_aim_branch,vertical,after.motor_branch)[:,2:]
    aim.sum().backward();aim_gradient=max(float(p.grad.abs().max()) for p in after.motor_branch.parameters() if p.grad is not None)
    assert aim_gradient<1e-5
    after.zero_grad(set_to_none=True);raw,_=sequence(after,prepared);raw[:,:2].sum().backward()
    assert any(p.grad is not None and float(p.grad.abs().max())>0 for p in after.motor_branch.parameters())
    after.zero_grad(set_to_none=True)
    with torch.no_grad():after.motor_branch[-1].bias.zero_()
    old_opt=optimizer(before,std,cp['config']['actor_lr'],cp['actor_training']['weapon_state_residual']['learning_rate'])
    new_opt=optimizer(after,new_std,cp['config']['actor_lr'],cp['actor_training']['weapon_state_residual']['learning_rate'])
    old_opt.load_state_dict(copy.deepcopy(cp['actor_optimizer']))
    old_named={**dict(before.named_parameters()),'@std':std};new_named={**dict(after.named_parameters()),'@std':new_std}
    for name,parameter in old_named.items():new_opt.state[new_named[name]]=copy.deepcopy(old_opt.state.get(parameter,{}))
    for new_group,old_group in zip(new_opt.param_groups,old_opt.param_groups):
        for key,value in old_group.items():
            if key!='params':new_group[key]=copy.deepcopy(value)
    saved=new_opt.state_dict()
    for key,value in cp['actor_optimizer']['state'].items():assert equal_tree(value,saved['state'][key])
    for a,b in zip(cp['actor_optimizer']['param_groups'],saved['param_groups']):assert equal_tree(a,b)
    assert len(saved['param_groups'])==len(cp['actor_optimizer']['param_groups'])+1
    assert all(not new_opt.state.get(p) for p in after.motor_branch.parameters())
    receipt=dict(state='complete',device='cuda',version=VERSION,source_weights_sha256=seal['weights_sha256'],
        source_checkpoint_sha256=seal['checkpoint_sha256'],zero_raw_and_likelihood_exact=True,history_rows=len(rows),
        memory_max_error=max(old_memory,new_memory),native_log_probability_max_error=native_error,
        nonzero_fixed_action_aim_max_error=aim_error,actual_move_aim_response=actual_move_response,
        nonmovement_kl_max_error=residual_kl_error,aim_to_motor_gradient_max=aim_gradient,
        motor_gradient_nonzero=True,old_optimizer_ids_and_states_exact=True,new_optimizer_group_empty=True,
        new_parameters=sum(p.numel() for p in after.motor_branch.parameters()),training_performed=False,promotion=False,
        inputs=[dict(path=p,sha256=d) for p,d in pins.items()],script_sha256=sha(__file__),
        scope='CUDA zero/nonzero migration proof on actual full TRAIN histories. Native compatibility and combat improvement not yet measured.')
    args.out.mkdir(parents=True);durable_json(args.out/'weights.json',updated)
    # Preserve CPU RNG byte storage for the ordinary resume path; no CPU NN math.
    fork=torch.load(args.source/'checkpoint.pt',map_location='cpu',weights_only=True)
    fork.update(actor=after.state_dict(),actor_optimizer=saved,weights_sha256=sha(args.out/'weights.json'),sequence_retention=None)
    fork['actor_training']={**fork['actor_training'],'motor_residual':dict(version=VERSION,learning_rate=.0003)}
    fork['model_migrations']=fork.get('model_migrations',[])+[receipt]
    durable_write(args.out/'checkpoint.pt',lambda f:torch.save(fork,f))
    reloaded=torch.load(args.out/'checkpoint.pt',map_location='cuda',weights_only=True)
    for key in cp:
        if key not in ('actor','actor_optimizer','weights_sha256','sequence_retention','actor_training','model_migrations'):assert equal_tree(cp[key],reloaded[key]),key
    assert equal_tree(branch(updated,'actor').state_dict(),reloaded['actor'])
    for path,digest in pins.items():assert sha(Path(path))==digest
    durable_json(args.out/'report.json',receipt)
    durable_json(args.out/'complete.json',dict(version='combat_update_complete_v1',weights_sha256=sha(args.out/'weights.json'),checkpoint_sha256=sha(args.out/'checkpoint.pt'),report_sha256=sha(args.out/'report.json')))
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('inputs','scope')}))


if __name__=='__main__':main()
