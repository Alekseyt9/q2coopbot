"""Real-TRAIN-output CUDA projection checks; no optimizer, runtime or new games."""
import argparse
import json
import shutil
from pathlib import Path
import torch
import numpy as np
from evaluate_registered_combat_models import read,save,sha
from combat_action_inertia import VERSION,EPS,apply,movement_log_std,validate

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('protocol','cache','out'):ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();assert torch.cuda.is_available();torch.set_default_device('cuda');torch.set_num_threads(2);pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();h=sha(path);assert expected is None or h==expected,str(path);pins[str(path)]=h;return path
    p=read(pin(a.protocol));assert p['version']=='combat_action_inertia_experiment_v1' and p['mean_clamp_epsilon']==EPS
    seal=read(pin(a.cache/'complete.json'));cache=read(pin(a.cache/'report.json',seal['report_sha256']))
    assert cache['version']=='combat_action_inertia_source_cache_v1' and cache['source_parameters_exact'] and cache['device']=='cuda'
    assert cache['protocol_sha256']==sha(a.protocol) and len(cache['chunks'])==24
    weights=pin(Path(p['source'])/'weights.json',p['source_weights_sha256']);model=read(weights)
    std=torch.tensor(model['log_std'],device='cuda');zero=dict(version=VERSION,rho=0.);header=dict(version=VERSION,rho=p['rho'])
    assert movement_log_std(std,zero) is std
    newstd=movement_log_std(std,header);assert torch.equal(newstd[2:],std[2:]) and bool((newstd[:2]<std[:2]).all())
    rows=changed=nonzero_grad=0
    for item in cache['chunks']:
        assert item['split']=='train'
        with np.load(pin(item['path'],item['sha256'])) as data:
            x=torch.tensor(data['features'],device='cuda');raw=torch.tensor(data['raw_actor'],device='cuda')
        before=raw.clone();assert apply(x,raw,zero) is raw
        out=apply(x,raw,header);assert torch.equal(raw,before) and torch.equal(out[:,2:],raw[:,2:]) and bool(torch.isfinite(out).all())
        changed+=int(((out[:,:2]-raw[:,:2]).abs().amax(-1)>1e-6).sum());rows+=len(x)
        variable=raw.detach().clone().requires_grad_(True);apply(x,variable,header)[:,:2].sum().backward()
        assert bool(torch.isfinite(variable.grad).all()) and bool((variable.grad[:,2:]==0).all());nonzero_grad+=int((variable.grad[:,:2].abs()>0).sum())
    assert rows==cache['rows'] and changed>0 and nonzero_grad>0
    x=torch.zeros((2,1147),device='cuda');raw=torch.zeros((2,81),device='cuda');x[0,13:15]=1;x[1,13:15]=-1;raw[0,:2]=20;raw[1,:2]=-20
    bounded=apply(x,raw,header)[:,:2].tanh();assert torch.allclose(bounded,torch.tensor([[1-EPS,1-EPS],[-1+EPS,-1+EPS]],device='cuda'),atol=1e-7,rtol=0)
    for rho in (-.1,1.,float('nan'),True):
        try:validate(dict(version=VERSION,rho=rho))
        except AssertionError:pass
        else:raise AssertionError('Invalid rho accepted')
    for path,h in pins.items():assert sha(path)==h,path
    a.out.mkdir(parents=True);sources=a.out/'source';sources.mkdir()
    for name in ('combat_action_inertia.py','probe_combat_action_inertia_cuda.py'):
        src=Path(__file__).with_name(name);target=sources/name;shutil.copy2(src,target);pin(target,sha(src))
    report=dict(state='complete',version='combat_action_inertia_cuda_core_probe_v1',device='cuda',rows=rows,changed_movement_rows=changed,nonzero_input_gradient_components=nonzero_grad,
        zero_identity_exact=True,other_raw_outputs_exact=True,source_raw_unchanged=True,angle_std_exact=True,boundary_clamp_passed=True,inputs=pins,
        neural_parameters_added=0,training_performed=False,source_model_training=False,new_games=False,runtime_integration_proven=False,ppo_checkpoint_written=False,promotion=False,
        scope='Only CUDA output transformation and Gaussian std projection on real24TRAIN source outputs. Not actual new runtime trajectories, likelihood parity, checkpoint migration or policy quality.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')));print(dict(rows=rows,changed=changed,core_passed=True,runtime_proven=False))

if __name__=='__main__':main()
