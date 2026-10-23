"""Export a passed shared prototype and matched constant for native evaluation."""
import argparse
import copy
import json
import math
from pathlib import Path
from evaluate_registered_combat_models import read, save, sha
from ppo_combat import torch, training_devices
from ppo_recurrent import durable_write
from combat_joint_aim_policy import JointAim, base_model,GEOMETRY_KIND,GEOMETRY_SAMPLER


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('source','prototype','dataset','out'):ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(p,expected=None):
        p=Path(p).resolve();h=sha(p);assert expected is None or h==expected,str(p);pins[str(p)]=h;return p
    ss=read(pin(a.source/'complete.json'))
    source=read(pin(a.source/'weights.json',ss['weights_sha256']))
    pin(a.source/'checkpoint.pt',ss['checkpoint_sha256'])
    ps=read(pin(a.prototype/'complete.json'));report=read(pin(a.prototype/'report.json',ps['report_sha256']))
    proto=read(pin(a.prototype/'prototype.json',ps['prototype_sha256']))
    assert report['gate'] and report['variant'] in ('shared','geometry8') and report['device']=='cuda'
    if report['variant']=='geometry8':
        comparison=read(pin(a.prototype.parent/'report.json'))
        assert comparison['selected_for_runtime']=='geometry8' and proto['head']['kind']==GEOMETRY_KIND
    assert proto['source_weights_sha256']==ss['weights_sha256']
    ds=read(pin(a.dataset/'complete.json'));assert proto['dataset_sha256']==ds['labels_sha256']
    rows=list(map(json.loads,pin(a.dataset/'contexts.jsonl',ds['labels_sha256']).open(encoding='utf-8-sig')))
    expected=read(pin(a.prototype/'contexts.json'))
    assert [(r['seed'],r['frame']) for r in rows]==[(r['seed'],r['frame']) for r in expected]
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    x=torch.tensor([r['features'] for r in rows],device='cuda',dtype=torch.float32)
    z=torch.tensor([r['sample']['aim_candidate']['source_latent'] for r in rows],device='cuda',dtype=torch.float32)
    cat=lambda k:torch.tensor([r['sample'].get(k,0) for r in rows],device='cuda',dtype=torch.long)
    timing=torch.tensor([r['sample']['aim_candidate']['timing'] for r in rows],device='cuda',dtype=torch.float32)
    args=(x,z,cat('target'),cat('aim_mode'),cat('vertical'),cat('weapon'),timing)
    updated=copy.deepcopy(source);updated['aim_candidate_head']['shared_scorer']=proto['head']
    if proto['head']['kind']==GEOMETRY_KIND:updated['aim_candidate_head']['sampler']=GEOMETRY_SAMPLER
    assert base_model(updated,allow_shared=True)==base_model(source)
    try:JointAim(updated)
    except AssertionError:pass
    else:raise AssertionError('Unintegrated training/resume must reject shared scorer')
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):joint=JointAim(updated,allow_shared=True).requires_grad_(False)
    with torch.inference_mode():logits,active=joint(*args);probs=logits.softmax(-1)
    wanted=torch.tensor([r['probabilities'] for r in expected],device='cuda')
    error=float((probs-wanted).abs().max());assert bool(active.all()) and error<1e-6,error
    constant=copy.deepcopy(source);q=expected[0]['constant_probabilities'];assert len(q)==25 and all(v>0 for v in q)
    assert all(r['constant_probabilities']==q for r in expected)
    last=constant['aim_candidate_head']['layers'][-1]
    last['weight']=[[0.]*len(row) for row in last['weight']];last['bias']=[math.log(v) for v in q]
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):control=JointAim(constant).requires_grad_(False)
    with torch.inference_mode():cp,ca=control(*args);control_error=float((cp.softmax(-1)-torch.tensor(q,device='cuda')).abs().max())
    assert bool(ca.all()) and control_error<1e-6
    a.out.mkdir(parents=True)
    for name,model in (('shared',updated),('constant',constant)):
        directory=a.out/name;directory.mkdir();save(directory/'weights.json',model)
        # This is an evaluation checkpoint, deliberately not an Adam/RNG resume.
        durable_write(directory/'checkpoint.pt',lambda f:torch.save(dict(kind='combat_candidate_evaluation_checkpoint_v1',
            variant=name,inference_only=True,source_checkpoint_sha256=ss['checkpoint_sha256'],weights_sha256=sha(directory/'weights.json')),f))
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):decoded=JointAim(read(directory/'weights.json'),allow_shared=True).requires_grad_(False)
        with torch.inference_mode():again,_=decoded(*args)
        assert torch.equal(again,logits if name=='shared' else cp)
        save(directory/'report.json',dict(state='complete',version='combat_candidate_evaluation_bundle_v1',variant=name,
             device='cuda',weights_sha256=sha(directory/'weights.json'),source_weights_sha256=ss['weights_sha256'],
             probability_error=error if name=='shared' else control_error,roundtrip_verified=True,
             training_resume_supported=False,training_performed=False,promotion=False,inputs=pins))
        save(directory/'complete.json',dict(state='complete',weights_sha256=sha(directory/'weights.json'),
             checkpoint_sha256=sha(directory/'checkpoint.pt'),report_sha256=sha(directory/'report.json')))
    for p,h in pins.items():assert sha(p)==h,p
    save(a.out/'report.json',dict(state='complete',device='cuda',contexts=len(rows),shared_error=error,constant_error=control_error,
         inputs=pins,script_sha256=sha(__file__),training_performed=False,promotion=False,training_resume_supported=False))
    save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(json.dumps(read(a.out/'report.json'),indent=2))


if __name__=='__main__':main()
