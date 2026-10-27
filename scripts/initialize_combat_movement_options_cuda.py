"""CUDA initialization of a separate option head; source policy stays frozen."""
import argparse
import copy
import json
import math
from pathlib import Path

from ppo_combat import torch, nn, layers, training_devices
from process_combat_architecture_pool import read, save, sha
from combat_movement_options import VERSION, SAMPLER, DISTRIBUTION, SHIFTS, WIDTH, validate, MovementOptions
from combat_joint_aim_policy import validate as validate_joint


def run(source,out):
    assert not out.exists()
    training_devices();torch.set_num_threads(2);torch.manual_seed(20261027)
    model=read(source);validate_joint(model,allow_shared=True)
    assert model['feature_version']=='combat_features_v13_memory_static_routes' and not model.get('movement_options')
    assert not model.get('wide_aim') and not any(k.startswith('diagnostic_') for k in model)
    header=dict(version=VERSION,sampler=SAMPLER,enabled=False,duration_frames=5,mean_shifts=SHIFTS)
    for name,n in (('actor',5),('value',1)):
        net=nn.Sequential(nn.Linear(WIDTH,32,device='cuda'),nn.ReLU(),nn.Linear(32,16,device='cuda'),nn.ReLU(),nn.Linear(16,n,device='cuda'))
        with torch.no_grad():
            net[-1].weight.zero_();net[-1].bias.zero_()
            if name=='actor':net[-1].bias.copy_(torch.tensor([math.log(.8),*[math.log(.05)]*4],device='cuda'))
        header[name]=layers(net)
    out.mkdir(parents=True)
    for enabled,name in ((False,'disabled'),(True,'enabled')):
        result=copy.deepcopy(model);result['action_distribution']=DISTRIBUTION
        result['movement_options']={**copy.deepcopy(header),'enabled':enabled}
        validate(result);validate_joint(result,allow_shared=True,allow_options=True)
        restored=copy.deepcopy(result);restored.pop('movement_options');restored['action_distribution']=model['action_distribution']
        assert restored==model,'Source parameters changed'
        branch=MovementOptions(result['movement_options'])
        with torch.no_grad():
            logits,values=branch(torch.randn((7,1554),device='cuda'),torch.randn((7,81),device='cuda'))
            expected=torch.tensor([.8,.05,.05,.05,.05],device='cuda').expand(7,-1)
            assert torch.allclose(logits.softmax(-1),expected,atol=1e-7,rtol=0) and torch.equal(values,torch.zeros_like(values))
        folder=out/name;folder.mkdir()
        # Compact JSON preserves every numeric value and avoids making the
        # immutable source + two small heads exceed the runtime's 16MiB limit.
        (folder/'weights.json').write_text(json.dumps(result,separators=(',',':'),allow_nan=False),encoding='utf-8')
        save(folder/'report.json',dict(state='initialized',device='cuda',source_sha256=sha(source),weights_sha256=sha(folder/'weights.json'),
            enabled=enabled,actor_parameters=sum(p.numel() for p in branch.actor.parameters()),
            option_value_parameters=sum(p.numel() for p in branch.value.parameters()),source_parameters_exact=True,
            training_performed=False,quality_evaluated=False,promotion=False,
            scope='New separate option branch. Source actor/critic/std/heads frozen byte-value exact; source optimizer remains in original checkpoint, not reinterpreted as a new optimizer.'))
    save(out/'complete.json',dict(state='complete',device='cuda',source_sha256=sha(source),training_performed=False,promotion=False))
    print(dict(state='initialized',out=str(out),source_unchanged=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();run(a.source.resolve(),a.out.resolve())
