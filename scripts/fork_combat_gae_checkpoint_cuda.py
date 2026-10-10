"""Explicit GAE-lambda-only checkpoint fork; verify unchanged state on CUDA."""
import argparse
import math
import pathlib
import shutil

from ppo_recurrent import read, sha, torch, training_devices, durable_json, durable_write


def validate_config_fork(old, new):
    assert set(old) == set(new), 'Config fields changed'
    assert {k:v for k,v in old.items() if k != 'lambda'} == {k:v for k,v in new.items() if k != 'lambda'}, 'Only GAE lambda may change'
    value = new['lambda']
    assert isinstance(value, (int,float)) and not isinstance(value,bool)
    assert math.isfinite(value) and 0 <= value <= 1 and value != old['lambda'], 'Invalid or unchanged lambda'


def same(left, right):
    if torch.is_tensor(left):
        assert torch.is_tensor(right) and left.dtype == right.dtype and left.shape == right.shape
        assert left.is_cuda and right.is_cuda and torch.equal(left,right)
    elif isinstance(left,dict):
        assert left.keys() == right.keys()
        for key in left: same(left[key],right[key])
    elif isinstance(left,(list,tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a,b in zip(left,right): same(a,b)
    else: assert left == right


def main():
    parser=argparse.ArgumentParser()
    for name in ('model','checkpoint','config','out'):
        parser.add_argument('--'+name,type=pathlib.Path,required=True)
    args=parser.parse_args()
    assert not args.out.exists(), 'Fresh fork output required'
    training_devices()
    cp=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    assert cp['version']=='combat_architecture_checkpoint_v1' and cp['weights_sha256']==sha(args.model)
    config=read(args.config);validate_config_fork(cp['config'],config)
    record=dict(version='combat_gae_config_fork_v1',source_checkpoint_sha256=sha(args.checkpoint),
                weights_sha256=sha(args.model),before_config=cp['config'],after_config=config,
                config_sha256=sha(args.config),training_performed=False)
    cp['config']=config
    cp['training_config_forks']=cp.get('training_config_forks',[])+[record]
    args.out.mkdir()
    shutil.copyfile(args.model,args.out/'weights.json')
    durable_write(args.out/'checkpoint.pt',lambda output:torch.save(cp,output))
    # CPU deserialization above preserves CPU RNG byte tensors needed by resume.
    # Verification of every saved tensor and optimizer/RNG/lineage field is CUDA.
    before=torch.load(args.checkpoint,map_location='cuda',weights_only=True)
    after=torch.load(args.out/'checkpoint.pt',map_location='cuda',weights_only=True)
    assert after['config']==config and after['training_config_forks'][-1]==record
    for key in before:
        if key not in ('config','training_config_forks'): same(before[key],after[key])
    assert after.get('training_config_forks',[])[:-1]==before.get('training_config_forks',[])
    assert sha(args.model)==sha(args.out/'weights.json')
    receipt={**record,'device':'cuda','checkpoint_sha256':sha(args.out/'checkpoint.pt'),
             'state_verified_unchanged':True,'updates_completed':cp['updates_completed'],
             'scope':'Only GAE lambda config and explicit fork history change. Weights bytes, actor/critic/Adam/RNG/consumed rollouts unchanged. No neural CPU inference or training.'}
    durable_json(args.out/'report.json',receipt)
    durable_json(args.out/'complete.json',dict(weights_sha256=sha(args.out/'weights.json'),checkpoint_sha256=sha(args.out/'checkpoint.pt'),report_sha256=sha(args.out/'report.json')))
    print(receipt,flush=True)


if __name__=='__main__':main()
