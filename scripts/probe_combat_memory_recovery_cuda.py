"""Test zero identity, state gating, output scope and gradients of the new core."""
import argparse
import copy
from pathlib import Path
import numpy as np
from evaluate_registered_combat_models import read,save,sha
from ppo_combat import torch,training_devices
from combat_memory_recovery import VERSION,PARAMETERS,active_mask,build_zero,project


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();h=sha(path);assert expected is None or expected==h,str(path);pins[str(path)]=h;return path
    pin(__file__);pin(Path(__file__).with_name('combat_memory_recovery.py'))
    seal=read(pin(a.dataset/'complete.json'));ds=read(pin(a.dataset/'report.json',seal['report_sha256']))
    assert ds['version']=='combat_target_lifecycle_dataset_v1'
    arrays=[]
    for chunk in ds['chunks']:
        with np.load(pin(chunk['path'],chunk['sha256'])) as data:arrays.append(data['features'])
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.manual_seed(2026102409);torch.cuda.manual_seed_all(2026102409)
    features=torch.tensor(np.concatenate(arrays),device='cuda');raw=torch.randn((len(features),81),device='cuda')
    active=active_mask(features);assert bool(active.any()) and bool((~active).any())
    head=build_zero();initial=copy.deepcopy(head.state_dict())
    with torch.no_grad():assert torch.equal(project(head,features,raw),raw)
    with torch.no_grad():head[-1].weight.uniform_(-.02,.02);head[-1].bias.copy_(torch.tensor([.1,-.1,.2],device='cuda'))
    projected=project(head,features,raw)
    untouched=torch.tensor([0,1,*range(5,81)],device='cuda',dtype=torch.long)
    assert torch.equal(projected[:,untouched],raw[:,untouched])
    assert torch.equal(projected[~active],raw[~active])
    assert bool((projected[active,2:5]!=raw[active,2:5]).any())
    loss=projected[active,2:5].square().mean();loss.backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters())
    assert sum(float(p.grad.abs().sum()) for p in head.parameters())>0
    head.load_state_dict(initial);head.zero_grad(set_to_none=True)
    a.out.mkdir(parents=True);torch.save(dict(version=VERSION,device='cuda',state_dict=head.state_dict(),parameters=PARAMETERS,
        role='Prototype module only; not PPO/runtime weights',training_performed=False),a.out/'checkpoint.pt')
    reloaded=torch.load(a.out/'checkpoint.pt',map_location='cuda',weights_only=True)
    assert all(torch.equal(v,reloaded['state_dict'][k]) for k,v in initial.items())
    with torch.no_grad():assert torch.equal(project(head,features,raw),raw)
    for path,h in pins.items():assert sha(path)==h,path
    report=dict(state='passed',version=VERSION,device='cuda',rows=len(features),active_rows=int(active.sum()),parameters=PARAMETERS,
        zero_identity=True,nonzero_outside_gate_identity=True,unmodified_outputs_exact=True,gradient_verified=True,gpu_reload_exact=True,
        input='Actual source client feature rows; synthetic raw actor outputs for isolated module contract checks.',
        inputs=pins,checkpoint_sha256=sha(a.out/'checkpoint.pt'),training_performed=False,source_model_execution=False,
        policy_weights_written=False,runtime_compatibility_proven=False,promotion=False,
        scope='Isolated CUDA prototype only. No source actor runtime/checkpoint integration, no claimed gaze quality, fire calibration or combat improvement.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='passed',report_sha256=sha(a.out/'report.json')))
    print(dict(state='passed',device='cuda',rows=len(features),active_rows=int(active.sum()),parameters=PARAMETERS))


if __name__=='__main__':main()
