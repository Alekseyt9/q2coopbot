"""Sealed v9 -> v10 migration of full actor/critic and Adam; CUDA only."""
import argparse
import copy
import pathlib

from combat_equipped_weapon_features import PREFIX, WIDTH, VERSION
from combat_input_checkpoint_expansion_cuda import expand_checkpoint_inputs
from migrate_combat_navigation_cuda import branch, output, validate, torch
from migrate_combat_navigation_checkpoint_cuda import verify
from ppo_recurrent import read, sha, durable_json, durable_write


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=pathlib.Path,required=True)
    parser.add_argument('--out',type=pathlib.Path,required=True)
    parser.add_argument('--fork-sequence-retention',action='store_true')
    args=parser.parse_args()
    assert torch.cuda.is_available() and not args.out.exists()
    torch.set_default_device('cuda')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(2)
    seal=read(args.source/'complete.json')
    for name,field in (('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')):
        assert sha(args.source/name)==seal[field], 'Broken source seal'
    old=read(args.source/'weights.json');assert old['feature_version']=='combat_features_v9'
    validate(old)
    cp=torch.load(args.source/'checkpoint.pt',map_location='cuda',weights_only=True)
    assert cp['weights_sha256']==seal['weights_sha256']
    previous_retention=copy.deepcopy(cp.get('sequence_retention'))
    assert bool(previous_retention)==args.fork_sequence_retention, 'Active sequence objective requires explicit migration fork'
    new=copy.deepcopy(old)
    for role in ('actor','value'):
        for row in new[role][0]['weight']:
            assert len(row)==PREFIX
            row.extend([0.]*(WIDTH-PREFIX))
    new['feature_version']=VERSION
    validate(new)
    modules={role:branch(old,role) for role in ('actor','value')}
    for role,module in modules.items():
        assert module.state_dict().keys()==cp[role].keys()
        assert all(torch.equal(v,cp[role][name]) for name,v in module.state_dict().items())
    transfer_source=copy.deepcopy(cp)
    transfer_source['sequence_retention']=None
    result,changes=expand_checkpoint_inputs(transfer_source,
        {role:[name for name,_ in module.named_parameters()] for role,module in modules.items()},PREFIX,WIDTH)
    errors={}
    with torch.no_grad():
        x=torch.randn(2,33,PREFIX,device='cuda')*.1
        x[...,426:466:5]=1
        tail=torch.zeros(2,33,26,device='cuda')
        tail[...,4]=1;tail[...,14]=1;tail[...,24]=1;tail[...,25]=.1
        nx=torch.cat((x,tail),-1)
        for role,module in modules.items():
            widened=branch(new,role)
            assert all(torch.equal(v,result[role][name]) for name,v in widened.state_dict().items())
            before,after=output(module,x),output(widened,nx)
            torch.testing.assert_close(before,after,rtol=1e-5,atol=1e-5)
            errors[role]=float((before-after).abs().max())
    # Optimizer.load_state_dict may retain CUDA tensor references. Exercise
    # disposable Adam on a deep copy, never the checkpoint being published.
    checks=verify(copy.deepcopy(result),new)
    for role in ('actor','value'):
        changed=next(c['parameter'] for c in changes if c['role']==role)
        ids=cp[role+'_optimizer']['param_groups'][0]['params']
        names=list(cp[role])+(['log_std'] if role=='actor' else [])
        for identifier,name in zip(ids,names):
            for field,before in cp[role+'_optimizer']['state'].get(identifier,{}).items():
                after=result[role+'_optimizer']['state'][identifier][field]
                if name==changed and field!='step':
                    assert torch.equal(before,after[:,:PREFIX]) and not bool(after[:,PREFIX:].any())
                else:assert torch.equal(before,after)
    args.out.mkdir(parents=True)
    durable_json(args.out/'weights.json',new)
    result['weights_sha256']=sha(args.out/'weights.json')
    receipt=dict(version='combat_equipped_checkpoint_migration_v1',device='cuda',
        source_weights_sha256=seal['weights_sha256'],source_checkpoint_sha256=seal['checkpoint_sha256'],
        source_seal_sha256=sha(args.source/'complete.json'),feature_version=VERSION,
        old_width=PREFIX,new_width=WIDTH,expanded_parameters=changes,output_max_error=errors,
        optimizer_checks=checks,previous_sequence_retention=previous_retention,
        sequence_objective_fork=args.fork_sequence_retention,training_performed=False,
        persisted_optimizer_steps=0,weights_sha256=result['weights_sha256'],
        updates_completed=result['updates_completed'],total_actor_steps=result['total_actor_steps'])
    result['model_migrations']=cp.get('model_migrations',[])+[copy.deepcopy(receipt)]
    durable_write(args.out/'checkpoint.pt',lambda f:torch.save(result,f))
    restored=torch.load(args.out/'checkpoint.pt',map_location='cuda',weights_only=True)
    verify(copy.deepcopy(restored),new)
    for field in ('config','consumed_rollouts','updates_completed','total_actor_steps','anchor_sha256','bank_sha256','actor_training'):
        assert restored[field]==cp[field]
    assert torch.equal(restored['rng'],cp['rng'])
    assert all(torch.equal(a,b) for a,b in zip(restored['cuda_rng'],cp['cuda_rng'])) and len(restored['cuda_rng'])==len(cp['cuda_rng'])
    assert restored['sequence_retention'] is None
    assert sha(args.source/'weights.json')==seal['weights_sha256'] and sha(args.source/'checkpoint.pt')==seal['checkpoint_sha256']
    receipt.update(checkpoint_sha256=sha(args.out/'checkpoint.pt'),persisted_restoration_verified=True,
        scope='Append zero columns, preserve old parameters/moments/clocks/RNG/history. Archive old sequence objective; new train-only own-history spec required. No training or gameplay quality claim.')
    durable_json(args.out/'report.json',receipt)
    durable_json(args.out/'complete.json',dict(version='combat_model_migration_complete_v1',weights_sha256=receipt['weights_sha256'],checkpoint_sha256=receipt['checkpoint_sha256'],report_sha256=sha(args.out/'report.json')))
    print(receipt,flush=True)


if __name__=='__main__':main()
