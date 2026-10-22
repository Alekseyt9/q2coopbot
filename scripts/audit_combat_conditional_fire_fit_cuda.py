"""Verify conditional-fire fit resume, runtime export and frozen Adam on CUDA."""
import argparse
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','update','boundary','corpus','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--scope',choices=('all','conditional-fire'),required=True)
    args=parser.parse_args();assert not args.out.exists()
    import torch
    assert torch.cuda.is_available();torch.set_default_device('cuda')
    from fork_conditional_weapon_exploration_cuda import equal_tree
    from migrate_combat_navigation_cuda import branch
    from combat_sequence_retention import CONDITIONAL_FIRE_EXPECTATION
    source=torch.load(args.source/'checkpoint.pt',map_location='cuda',weights_only=True)
    before=torch.load(args.boundary,map_location='cuda',weights_only=True)
    after=torch.load(args.update/'checkpoint.pt',map_location='cuda',weights_only=True)
    report=read(args.update/'report.json');seal=read(args.update/'complete.json')
    for file,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:
        assert sha(args.update/file)==seal[key]
    assert sha(args.boundary)==report['initial_boundary_sha256']
    assert source['weights_sha256']==sha(args.source/'weights.json')
    for key in ('actor','value','log_std','actor_optimizer','value_optimizer','rng','cuda_rng'):
        assert equal_tree(before[key],source[key]),key
    meta=read(args.corpus/'processing-train/merged/report.json')
    assert report['rollout_sha256']==meta['rollout_sha256']==before['rollout_sha256']
    assert before['rollout_sha256'] not in source['consumed_rollouts']
    assert after['consumed_rollouts']==source['consumed_rollouts']+[before['rollout_sha256']]
    assert report['device']=='cuda' and report['actor_training']['scope']==args.scope
    assert 0<report['actor_steps']<=source['config']['actor_steps']==10
    assert after['config']==source['config']
    assert after['updates_completed']==source['updates_completed']+1
    assert after['total_actor_steps']==source['total_actor_steps']+report['actor_steps']
    assert after['sequence_retention']['conditional_fire_expectation']==CONDITIONAL_FIRE_EXPECTATION
    model=read(args.update/'weights.json')
    for key in ('actor','value'):
        assert equal_tree(branch(model,key).state_dict(),after[key]),key
    assert torch.equal(torch.tensor(model['log_std']),after['log_std'])
    assert any(not torch.equal(v,after['actor'][k]) for k,v in source['actor'].items() if k.startswith('conditional_fire_branch.'))
    groups=after['actor_optimizer']['param_groups']
    assert len(groups)==4 and groups[-1]['name']=='conditional_fire'
    for key in groups[-1]['params']:
        assert not source['actor_optimizer']['state'].get(key)
        assert float(after['actor_optimizer']['state'][key]['step'])==report['actor_steps']
    if args.scope=='conditional-fire':
        assert report['trainable_actor_parameters']==6369
        for key,value in source['actor'].items():
            if not key.startswith('conditional_fire_branch.'):
                assert torch.equal(value,after['actor'][key]),key
        assert torch.equal(source['log_std'],after['log_std'])
        for group in source['actor_optimizer']['param_groups'][:-1]:
            for key in group['params']:
                assert equal_tree(source['actor_optimizer']['state'].get(key,{}),after['actor_optimizer']['state'].get(key,{})),key
    save(args.out,dict(state='complete',device='cuda',scope=args.scope,
        initial_actor_critic_adam_rng_exact=True,rollout_consumed_once=True,runtime_checkpoint_exact=True,
        frozen_old_actor_std_adam_exact=args.scope=='conditional-fire',new_fire_parameters_changed=True,
        new_fire_adam_steps_exact=True,actor_steps=report['actor_steps'],rows=report['rows'],
        source_checkpoint_sha256=sha(args.source/'checkpoint.pt'),boundary_sha256=sha(args.boundary),
        corpus_complete_sha256=sha(args.corpus/'complete.json'),update_complete_sha256=sha(args.update/'complete.json'),
        script_sha256=sha(__file__),training_performed=False,promotion=False))
    print(args.out,flush=True)


if __name__=='__main__':main()
