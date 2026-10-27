"""Zero cycle residual migration, preserving each old named Adam state on CUDA."""
import argparse
import copy
from pathlib import Path
from ppo_combat import torch,training_devices
from combat_full_joint_ppo import FullJointScope,equal_tree
from combat_weapon_fire_residual import initialize,CYCLE_VERSION
from evaluate_registered_combat_models import read,save,sha


def migrate(model,cp):
    assert not cp.get('technical_only') and not model.get('weapon_fire_residual')
    before=FullJointScope(model,cp['config']);before.restore(cp,cp['weights_sha256'])
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(20261027);changed=initialize(model,CYCLE_VERSION)
    after=FullJointScope(changed,cp['config'],allow_cycle_fire=True)
    new=copy.deepcopy(cp);new['actor']=copy.deepcopy(after.actor.state_dict());new['optimizer_layout']=after.layout()
    assert all(equal_tree(v,new['actor'][k]) for k,v in cp['actor'].items())
    old_names={name:index for layout,g in zip(cp['optimizer_layout'],cp['actor_optimizer']['param_groups']) for name,index in zip(layout['parameters'],g['params'])}
    optimizer=copy.deepcopy(after.actor_optimizer.state_dict())
    for layout,group,old_layout,old_group in zip(new['optimizer_layout'],optimizer['param_groups'],cp['optimizer_layout'],cp['actor_optimizer']['param_groups']):
        assert layout['name']==old_layout['name']
        assert [n for n in layout['parameters'] if n in old_names]==old_layout['parameters']
        params=group['params'];group.update(copy.deepcopy(old_group));group['params']=params
        for name,index in zip(layout['parameters'],params):
            if name in old_names:
                old=old_names[name]
                if old in cp['actor_optimizer']['state']:optimizer['state'][index]=copy.deepcopy(cp['actor_optimizer']['state'][old])
            else:assert name.startswith('actor.weapon_fire_residual_branch.')
    new['actor_optimizer']=optimizer
    for key in ('value','joint','log_std','value_optimizer','consumed_rollouts','consumed_seeds','updates_completed','rng','cuda_rng'):
        assert equal_tree(cp[key],new[key]),key
    new['architecture_migration']=dict(version=CYCLE_VERSION,source_weights_sha256=cp['weights_sha256'],new_parameters=9121,old_named_adam_preserved=True)
    new['initialization']='zero_cycle_fire_residual_preserved_full_adam'
    return changed,new


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('model','checkpoint','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    cp=torch.load(a.checkpoint,map_location='cuda',weights_only=True);assert cp['weights_sha256']==sha(a.model)
    model,new=migrate(read(a.model),cp);a.out.mkdir(parents=True);save(a.out/'weights.json',model)
    new['weights_sha256']=sha(a.out/'weights.json')
    scope=FullJointScope(model,new['config'],allow_cycle_fire=True);scope.restore(new,new['weights_sha256'])
    torch.save(new,a.out/'checkpoint.pt');reopened=torch.load(a.out/'checkpoint.pt',map_location='cuda',weights_only=True)
    scope.restore(reopened,new['weights_sha256']);assert equal_tree(new,reopened)
    save(a.out/'report.json',dict(state='passed',device='cuda',training_performed=False,version=CYCLE_VERSION,
        old_parameters_and_named_adam_exact=True,checkpoint_reopened=True,new_parameters=9121,
        weights_sha256=new['weights_sha256'],checkpoint_sha256=sha(a.out/'checkpoint.pt'),
        inputs={str(p.resolve()):sha(p) for p in (a.model,a.checkpoint,Path(__file__))},
        scope='Zero migration only. Separate CUDA/native behavior parity and quality required.'))
    save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json'),weights_sha256=new['weights_sha256'],checkpoint_sha256=sha(a.out/'checkpoint.pt')))
    print('Zero cycle residual migration and preserved Adam restore passed')


if __name__=='__main__':main()
