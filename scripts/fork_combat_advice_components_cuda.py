"""Parameter ablation of closed corrective BC; CUDA only, no additional fitting.

Movement, aim and target/fire partitions cover the trained head rows. They are
parameter interventions, not isolated behavior: spatial aim consumes movement
and the protected weapon residual consumes soft target probabilities.
"""
import argparse
import copy
import pathlib

from evaluate_registered_combat_models import read, save, sha


def main():
    ap = argparse.ArgumentParser()
    for name in ('source', 'trained', 'out'): ap.add_argument('--'+name, type=pathlib.Path, required=True)
    ap.add_argument('--scope', choices=('movement', 'aim', 'target-fire'), required=True)
    args = ap.parse_args(); assert not args.out.exists()
    import torch
    assert torch.cuda.is_available(), 'CUDA required; no CPU fallback'
    torch.set_default_device('cuda'); torch.set_num_threads(2)
    from migrate_combat_navigation_cuda import branch
    from combat_spatial_aim import export as export_spatial
    from combat_weapon_state_residual import optimizer
    from fork_conditional_weapon_exploration_cuda import equal_tree
    from ppo_recurrent import durable_json, durable_write
    pins = {}; models = {}; checkpoints = {}
    for label, root in (('source',args.source), ('trained',args.trained)):
        seal = read(root/'complete.json'); pins[str((root/'complete.json').resolve())] = sha(root/'complete.json')
        for name,key in (('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')):
            assert sha(root/name) == seal[key]; pins[str((root/name).resolve())] = seal[key]
        models[label] = read(root/'weights.json')
        checkpoints[label] = torch.load(root/'checkpoint.pt',map_location='cuda',weights_only=True)
        assert equal_tree(branch(models[label],'actor').state_dict(),checkpoints[label]['actor'])
    source, trained = checkpoints['source'], checkpoints['trained']
    assert trained['reactive_advice_bc']['actor_adam_reset'] and trained['reactive_advice_bc']['scope'] == 'heads'
    assert equal_tree(source['value'],trained['value']) and torch.equal(source['log_std'],trained['log_std'])
    head_keys = ('base.head.weight','base.head.bias','base.residual.weight','base.residual.bias')
    aim_prefix = ('branch.', 'sampled_motion_aim_branch.')
    partitions = {name:torch.zeros(81,dtype=torch.bool,device='cuda') for name in ('movement','aim','target-fire')}
    partitions['movement'][:2] = True
    partitions['aim'][2:4] = True; partitions['aim'][29:81] = True
    partitions['target-fire'][4] = True; partitions['target-fire'][20:29] = True
    union = torch.stack(list(partitions.values())).sum(0)
    assert bool((union <= 1).all()) and bool((union[:5] == 1).all()) and bool((union[20:] == 1).all())
    assert not bool(union[5:20].any())
    for key,value in trained['actor'].items():
        if key in head_keys: assert torch.equal(value[union == 0],source['actor'][key][union == 0])
        elif not key.startswith(aim_prefix): assert torch.equal(value,source['actor'][key]), key
    chosen = partitions[args.scope]
    actor = branch(models['source'],'actor'); state = actor.state_dict()
    for key,value in state.items():
        if key in head_keys: value[chosen] = trained['actor'][key][chosen]
        elif args.scope == 'aim' and key.startswith(aim_prefix): value.copy_(trained['actor'][key])
    actor.load_state_dict(state)
    for key,value in actor.state_dict().items():
        if key in head_keys:
            assert torch.equal(value[chosen],trained['actor'][key][chosen])
            assert torch.equal(value[~chosen],source['actor'][key][~chosen])
        else:
            expected = trained if args.scope == 'aim' and key.startswith(aim_prefix) else source
            assert torch.equal(value,expected['actor'][key]), key
    model = copy.deepcopy(models['source'])
    base,cell = actor.export(); model['actor'] = base; model['attention']['actor'] = cell
    export_spatial(model,actor)
    assert equal_tree(branch(model,'actor').state_dict(),actor.state_dict())
    args.out.mkdir(parents=True)
    save(args.out/'protocol.json',dict(version='combat_advice_parameter_ablation_v1',scope=args.scope,
        inputs=[dict(path=p,sha256=h) for p,h in pins.items()],training_performed=False,
        scope_note=__doc__))
    durable_json(args.out/'weights.json',model)
    fork = copy.deepcopy(source)
    fork.update(actor={k:v.detach().clone() for k,v in actor.state_dict().items()},weights_sha256=sha(args.out/'weights.json'))
    std = torch.nn.Parameter(fork['log_std'].detach().clone())
    fork['actor_optimizer'] = optimizer(actor,std,fork['config']['actor_lr'],
        fork.get('actor_training',{}).get('weapon_state_residual',{}).get('learning_rate',.0003)).state_dict()
    fork['reactive_advice_bc'] = dict(trained['reactive_advice_bc'],component_scope=args.scope,
        trained_weights_sha256=sha(args.trained/'weights.json'),additional_fitting=False)
    durable_write(args.out/'checkpoint.pt',lambda f:torch.save(fork,f))
    assert equal_tree(torch.load(args.out/'checkpoint.pt',map_location='cuda',weights_only=True),fork)
    for path,digest in pins.items(): assert sha(path) == digest
    durable_json(args.out/'report.json',dict(state='complete',device='cuda',training_performed=False,promotion=False,
        scope=args.scope,source=str(args.source.resolve()),trained=str(args.trained.resolve()),
        source_weights_sha256=sha(args.source/'weights.json'),trained_weights_sha256=sha(args.trained/'weights.json'),
        actor_adam_reset=True,value_unchanged=True,std_unchanged=True,protected_parameter_rows_exact=True,
        inputs=[dict(path=p,sha256=h) for p,h in pins.items()],script_sha256=sha(__file__),
        note='Exact parameter partition, no refit. All causal outcomes require fresh matched native evaluation. Not output isolation or promotion.'))
    durable_json(args.out/'complete.json',dict(version='combat_update_complete_v1',weights_sha256=sha(args.out/'weights.json'),
        checkpoint_sha256=sha(args.out/'checkpoint.pt'),report_sha256=sha(args.out/'report.json')))
    print(args.out/'complete.json')


if __name__ == '__main__': main()
