"""Append a zero shared target residual while preserving checkpoint/Adam lineage."""
import argparse
import copy
import pathlib

from ppo_recurrent import (torch, training_devices, read, sha, durable_json,
                           durable_write, Recurrent)
from combat_attention import CausalAttention
from combat_spatial_aim import wrap
from combat_shared_target import initialize, VERSION, validate
from ppo_combat import layers


def actor(model):
    if model.get('memory'):
        base = Recurrent(model['actor'], model['memory']['actor'])
    else:
        spec = model['attention']
        base = CausalAttention(model['actor'],spec['actor'],spec['heads'],spec['window'])
    return wrap(base,model).to('cuda')


def migrate_optimizer(before, old_names, new_names):
    result = copy.deepcopy(before)
    assert len(result['param_groups']) == 1, 'single actor Adam group required'
    old_ids = result['param_groups'][0]['params']
    assert old_ids == list(range(len(old_names)))
    mapping = {name: index for name,index in zip(old_names,old_ids)}
    result['state'] = {index:copy.deepcopy(before['state'][mapping[name]])
                       for index,name in enumerate(new_names)
                       if name in mapping and mapping[name] in before['state']}
    result['param_groups'][0]['params'] = list(range(len(new_names)))
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ('model','checkpoint','out'):
        parser.add_argument('--'+name,required=True,type=pathlib.Path)
    args = parser.parse_args()
    assert not args.out.exists(), 'fresh output required'
    training_devices()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    model = read(args.model)
    assert not model.get('shared_target') and model['feature_version']=='combat_features_v9'
    before = torch.load(args.checkpoint,map_location='cuda',weights_only=True)
    assert before['version']=='combat_architecture_checkpoint_v1'
    assert before['weights_sha256']==sha(args.model)
    assert before.get('actor_training',{}).get('scope')=='decisions'
    old = actor(model)
    assert old.state_dict().keys()==before['actor'].keys()
    for name,value in old.state_dict().items():
        assert torch.equal(value,before['actor'][name]), name
    migrated = copy.deepcopy(model)
    torch.manual_seed(20261011)
    migrated['shared_target'] = dict(version=VERSION,layers=layers(initialize()))
    validate(migrated)
    new = actor(migrated)
    for name,value in old.state_dict().items():
        assert torch.equal(value,new.state_dict()[name]), name
    x = torch.randn(4,33,1121,device='cuda')*.2
    x[...,73:169].reshape(4,33,8,12)[...,0]=1
    x[...,881:1121].reshape(4,33,8,30)[...,0]=1
    with torch.no_grad():
        a,b = old(x),new(x)
        assert len(a)==len(b)
        for left,right in zip(a,b):
            assert torch.equal(left,right), 'zero residual changed actor outputs/state'
    old_names = list(dict(old.named_parameters()))+['log_std']
    new_names = list(dict(new.named_parameters()))+['log_std']
    assert old_names[:-1]==new_names[:len(old_names)-1]
    updated = copy.deepcopy(before)
    updated['actor']=new.state_dict()
    updated['actor_optimizer']=migrate_optimizer(before['actor_optimizer'],old_names,new_names)
    optimizer = torch.optim.Adam(list(new.parameters())+
                                 [torch.nn.Parameter(before['log_std'].clone())],lr=before['config']['actor_lr'])
    optimizer.load_state_dict(updated['actor_optimizer'])
    old_ids = {name:i for i,name in enumerate(old_names)}
    for index,name in enumerate(new_names):
        previous = before['actor_optimizer']['state'].get(old_ids.get(name,-1),{})
        current = updated['actor_optimizer']['state'].get(index,{})
        assert previous.keys()==current.keys(), name
        for key,value in previous.items():
            assert torch.equal(value,current[key]), (name,key)
    # Migration is not a PPO update and consumes no rollout. Keep critic,
    # std, RNG, counters and every other field exactly as saved in the parent.
    lineage = dict(version=VERSION,parent_weights_sha256=sha(args.model),
                   parent_checkpoint_sha256=sha(args.checkpoint),
                   script_sha256=sha(pathlib.Path(__file__)))
    updated['model_migrations']=before.get('model_migrations',[])+[lineage]
    args.out.mkdir(parents=True)
    durable_json(args.out/'weights.json',migrated)
    updated['weights_sha256']=sha(args.out/'weights.json')
    durable_write(args.out/'checkpoint.pt',lambda stream:torch.save(updated,stream))
    report=dict(device='cuda',gpu=torch.cuda.get_device_name(),**lineage,
                weights_sha256=sha(args.out/'weights.json'),
                checkpoint_sha256=sha(args.out/'checkpoint.pt'),
                exact_actor_output_and_memory=True,old_actor_parameters_preserved=True,
                old_actor_adam_preserved=True,critic_and_std_preserved=True,
                training_history_and_rng_preserved=True,training_performed=False,
                new_parameter_scalars=sum(p.numel() for p in new.target_branch.parameters()),
                runtime_acceptance=False)
    report['version']='combat_shared_target_migration_v1'
    durable_json(args.out/'report.json',report)
    durable_json(args.out/'complete.json',dict(version='combat_migration_complete_v1',
                 weights_sha256=report['weights_sha256'],checkpoint_sha256=report['checkpoint_sha256'],
                 report_sha256=sha(args.out/'report.json')))
    print(report,flush=True)


if __name__=='__main__':
    main()
