"""Verify source checkpoint parameter identities and prepare explicit migration layout.

No checkpoint/model publication or optimizer step. Existing trainer/runtime
must gain contract support before this layout can become a resumable model.
"""
import argparse
from ppo_recurrent import torch,read,sha,durable_json
from migrate_combat_navigation_cuda import branch
from combat_weapon_state_residual import WeaponStateResidual,VERSION
import pathlib

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=pathlib.Path,required=True)
    parser.add_argument('--out',type=pathlib.Path,required=True)
    args=parser.parse_args();assert not args.out.exists() and torch.cuda.is_available()
    torch.set_default_device('cuda');torch.set_num_threads(2)
    seal=read(args.source/'complete.json')
    for name,field in (('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')):
        assert sha(args.source/name)==seal[field]
    model=read(args.source/'weights.json');assert model['feature_version']=='combat_features_v10' and model.get('spatial_aim')
    cp=torch.load(args.source/'checkpoint.pt',map_location='cuda',weights_only=True)
    assert cp['weights_sha256']==seal['weights_sha256']
    actor=branch(model,'actor');critic=branch(model,'value')
    for role,module in (('actor',actor),('value',critic)):
        assert module.state_dict().keys()==cp[role].keys()
        assert all(torch.equal(value,cp[role][name]) for name,value in module.state_dict().items())
    names=[name for name,_ in actor.named_parameters()]+['log_std']
    groups=cp['actor_optimizer']['param_groups'];assert len(groups)==1
    identifiers=groups[0]['params'];assert len(identifiers)==len(names) and len(set(identifiers))==len(identifiers)
    assert set(identifiers)==set(cp['actor_optimizer']['state'])
    mapping=[]
    for identifier,name in zip(identifiers,names):
        tensor=cp['log_std'] if name=='log_std' else cp['actor'][name]
        state=cp['actor_optimizer']['state'][identifier]
        assert state['exp_avg'].shape==state['exp_avg_sq'].shape==tensor.shape
        mapping.append(dict(name=name,identifier=identifier,shape=list(tensor.shape),step=float(state['step'])))
    torch.cuda.manual_seed(20261011);prototype=WeaponStateResidual()
    additions=[]
    for identifier,(name,param) in enumerate(prototype.branch.named_parameters(),max(identifiers)+1):
        additions.append(dict(name='weapon_state_branch.'+name,identifier=identifier,
            shape=list(param.shape),moment_initialization='zeros',step=0))
    assert not set(a['identifier'] for a in additions).intersection(identifiers)
    assert not set(a['name'] for a in additions).intersection(names)
    receipt=dict(version='combat_weapon_state_migration_layout_v1',state='complete',device='cuda',
        source_weights_sha256=seal['weights_sha256'],source_checkpoint_sha256=seal['checkpoint_sha256'],
        residual_version=VERSION,legacy_actor_group=mapping,new_residual_group=additions,
        new_parameters=sum(p.numel() for p in prototype.parameters()),
        updates_completed=cp['updates_completed'],total_actor_steps=cp['total_actor_steps'],
        consumed_rollouts=len(cp['consumed_rollouts']),sequence_objective_requires_explicit_fork=bool(cp.get('sequence_retention')),
        optimizer_steps=0,weights_published=False,checkpoint_published=False,
        script_sha256=sha(pathlib.Path(__file__)),
        scope='CUDA-verified source tensor identities/shapes and noncolliding Adam parameter ID layout. Legacy log_std stays in original group before new group IDs. Does not implement migration, optimizer restore, Go contract or gameplay acceptance.')
    durable_json(args.out,receipt)
    print(dict(state=receipt['state'],old_parameters=len(mapping),new_parameter_tensors=len(additions),new_scalars=receipt['new_parameters']))

if __name__=='__main__':main()
