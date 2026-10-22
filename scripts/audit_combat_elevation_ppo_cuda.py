"""Check a sealed source-to-elevation PPO update; CUDA numerical checks only."""
import argparse
from pathlib import Path
from evaluate_registered_combat_models import read, save, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();assert not args.out.exists()
    import torch
    assert torch.cuda.is_available()
    torch.set_default_device('cuda')
    from fork_conditional_weapon_exploration_cuda import equal_tree
    from migrate_combat_navigation_cuda import branch
    root=args.root.resolve();p=read(root/'protocol.json');source=Path(p['source']);update=root/'update'
    seal=read(update/'complete.json');report=read(update/'report.json')
    for file,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:
        assert sha(update/file)==seal[key]
    for path,digest in p['inputs'].items():assert sha(Path(path))==digest
    original=torch.load(source/'checkpoint.pt',map_location='cuda',weights_only=True)
    before=torch.load(root/'initial.pt',map_location='cuda',weights_only=True)
    after=torch.load(update/'checkpoint.pt',map_location='cuda',weights_only=True)
    assert sha(root/'initial.pt')==report['initial_boundary_sha256']
    for key in ['actor','value','log_std','actor_optimizer','value_optimizer','rng','cuda_rng']:
        assert equal_tree(before[key],original[key]),key
    assert before['rollout_sha256'] not in original['consumed_rollouts']
    assert after['consumed_rollouts']==original['consumed_rollouts']+[before['rollout_sha256']]
    assert after['total_actor_steps']==original['total_actor_steps']+report['actor_steps']
    assert report['device']=='cuda' and 0<report['actor_steps']<=10
    assert report['final_approx_kl']<=.005+1e-6
    assert report['actor_training']['scope']=='all' and report['critic_credit']['mode']=='legacy'
    assert after['config']==original['config'] and after['retention_weights']==original['retention_weights']
    assert not equal_tree(after['actor'],original['actor'])
    weights=read(update/'weights.json')
    for name in ['actor','value']:
        assert equal_tree(branch(weights,name).state_dict(),after[name]),name
    assert torch.equal(torch.tensor(weights['log_std']),after['log_std'])
    clocks={}
    for name,limit in [('actor_optimizer',report['actor_steps']),('value_optimizer',40)]:
        changes=[]
        assert after[name]['param_groups']==before[name]['param_groups']
        for key,state in after[name]['state'].items():
            old=before[name]['state'].get(key,{})
            delta=float(state.get('step',0))-float(old.get('step',0))
            assert delta in (0,limit),(name,key,delta)
            changes.append(delta)
        assert changes and max(changes)==limit
        if name=='value_optimizer':assert min(changes)==40
        clocks[name]=dict(states=len(changes),advanced=sum(x>0 for x in changes),steps=limit)
    save(args.out,dict(state='complete',device='cuda',source_boundary_exact=True,runtime_checkpoint_exact=True,
         rollout_previously_unconsumed=True,source_reward_config_preserved=True,optimizer_clocks=clocks,
         rows=report['rows'],actor_steps=report['actor_steps'],kl=report['final_approx_kl'],
         update_seal_sha256=sha(update/'complete.json'),protocol_sha256=sha(root/'protocol.json'),
         script_sha256=sha(__file__),training_performed=False,promotion=False))
    print(args.out)


if __name__=='__main__':main()
