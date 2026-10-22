"""Seal the matched original-actor-only control for the motor architecture."""
import argparse
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();assert not args.out.exists()
    import torch
    assert torch.cuda.is_available()
    from fork_conditional_weapon_exploration_cuda import equal_tree
    from migrate_combat_navigation_cuda import branch
    root=args.root.resolve();p=read(root/'protocol.json');source=Path(p['source']);update=root/'update'
    original=torch.load(source/'checkpoint.pt',map_location='cuda',weights_only=True)
    before=torch.load(root/'initial.pt',map_location='cuda',weights_only=True)
    after=torch.load(update/'checkpoint.pt',map_location='cuda',weights_only=True)
    report=read(update/'report.json');seal=read(update/'complete.json')
    for name,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:assert sha(update/name)==seal[key]
    assert sha(root/'initial.pt')==report['initial_boundary_sha256']
    for key in ['actor','value','log_std','actor_optimizer','value_optimizer','rng','cuda_rng']:assert equal_tree(before[key],original[key]),key
    assert report['actor_training']['scope']=='base-only' and report['device']=='cuda' and 0<report['actor_steps']<=10
    changed=[]
    for name,value in original['actor'].items():
        if name.startswith('motor_branch.'):assert torch.equal(value,after['actor'][name]),name
        else:changed.append(not torch.equal(value,after['actor'][name]))
    assert any(changed)
    for key in original['actor_optimizer']['param_groups'][-1]['params']:
        assert equal_tree(original['actor_optimizer']['state'].get(key,{}),after['actor_optimizer']['state'].get(key,{}))
    full=torch.load(Path(p['reference_full'])/'checkpoint.pt',map_location='cuda',weights_only=True)
    for key in ['value','value_optimizer']:assert equal_tree(after[key],full[key]),key
    for key in ['actor','value']:assert equal_tree(branch(read(update/'weights.json'),key).state_dict(),after[key]),key
    assert before['rollout_sha256'] not in original['consumed_rollouts']
    assert after['consumed_rollouts']==original['consumed_rollouts']+[before['rollout_sha256']]
    assert after['total_actor_steps']==original['total_actor_steps']+report['actor_steps']
    for path,digest in p['inputs'].items():assert sha(Path(path))==digest
    save(args.out,dict(state='complete',device='cuda',initial_state_exact=True,motor_parameters_and_adam_unchanged=True,
         original_actor_updated=True,critic_and_optimizer_exact_to_full=True,runtime_checkpoint_exact=True,
         source_rollout_unconsumed=True,actor_steps=report['actor_steps'],kl=report['final_approx_kl'],
         update_complete_sha256=sha(update/'complete.json'),protocol_sha256=sha(root/'protocol.json'),script_sha256=sha(__file__),training_performed=False,promotion=False))
    print(args.out)


if __name__=='__main__':main()
