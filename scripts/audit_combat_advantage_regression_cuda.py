"""CUDA boundary and exact uniform-control audit for AWR-style regression."""
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
    root=args.root.resolve();p=read(root/'protocol.json');assert read(root/'training-complete.json')['state']=='complete'
    original=torch.load(Path(p['source'])/'checkpoint.pt',map_location='cuda',weights_only=True)
    checkpoints={};reports={};inputs=[]
    for variant in p['variants']:
        update=root/variant;seal=read(update/'complete.json');report=read(update/'report.json')
        for file,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:
            assert sha(update/file)==seal[key];inputs.append(dict(path=str(update/file),sha256=seal[key]))
        before=torch.load(root/(variant+'-initial.pt'),map_location='cuda',weights_only=True)
        assert sha(root/(variant+'-initial.pt'))==report['initial_boundary_sha256']
        for key in ['actor','value','log_std','actor_optimizer','value_optimizer','rng','cuda_rng']:
            assert equal_tree(before[key],original[key]),(variant,key)
        assert before['rollout_sha256'] not in original['consumed_rollouts']
        assert report['device']=='cuda' and 0<report['actor_steps']<=10 and report['final_approx_kl']<=.005+1e-6
        checkpoints[variant]=torch.load(update/'checkpoint.pt',map_location='cuda',weights_only=True);reports[variant]=report
    expected=root.parent/'combat-elevation-ppo-v1-20261022/update/weights.json'
    assert sha(root/'legacy/weights.json')==sha(expected),'Default PPO changed'
    assert sha(root/'bc/weights.json')==sha(root/'uniform/weights.json'),'Uniform AWR differs from BC'
    for key in ['actor','value','log_std','actor_optimizer','value_optimizer','rng','cuda_rng','consumed_rollouts','total_actor_steps']:
        assert equal_tree(checkpoints['bc'][key],checkpoints['uniform'][key]),('uniform control',key)
    for variant in ['uniform','awr']:
        for key in ['value','value_optimizer']:
            assert equal_tree(checkpoints['bc'][key],checkpoints[variant][key]),('common critic',variant,key)
    assert not equal_tree(checkpoints['bc']['actor'],checkpoints['awr']['actor'])
    assert reports['awr']['policy_update']['weights']['strength']==1
    assert reports['uniform']['policy_update']['weights']['strength']==0
    for path,digest in p['inputs'].items():
        file=Path(path)
        if file.name=='ppo_recurrent.py':file=root/'legacy/trainer.py'
        if file.name=='combat_advantage_regression.py':file=root/'awr-helper.py'
        assert sha(file)==digest
    save(args.out,dict(state='complete',device='cuda',source_boundary_exact=True,legacy_ppo_runtime_exact=True,
         uniform_awr_bc_runtime_and_optimizer_exact=True,common_mc_critic_and_optimizer_exact=True,
         weighted_actor_different=True,source_rollout_unconsumed=True,inputs=inputs,
         variants={name:dict(actor_steps=r['actor_steps'],kl=r['final_approx_kl'],policy=r['policy_update']) for name,r in reports.items()},
         script_sha256=sha(__file__),training_performed=False,promotion=False))
    print(args.out)


if __name__=='__main__':main()
