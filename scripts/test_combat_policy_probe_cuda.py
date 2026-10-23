"""Numerical V2 one-frame checks on CUDA; captured states are not new labels."""
import argparse
import copy
import json
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha
from ppo_combat import torch,training_devices
from combat_joint_aim_policy import JointAim,POLICY_PROBE_VERSION
from fork_conditional_weapon_exploration_cuda import equal_tree


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('model','dataset','out'):ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();model=read(a.model);source_sha=sha(a.model)
    seal=read(a.dataset/'complete.json');assert sha(a.dataset/'contexts.jsonl')==seal['labels_sha256']
    rows=list(map(json.loads,(a.dataset/'contexts.jsonl').open(encoding='utf-8-sig')))
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):source=JointAim(model,allow_shared=True).requires_grad_(False)
    frozen=copy.deepcopy(source.state_dict());x=torch.tensor([r['features'] for r in rows],device='cuda',dtype=torch.float32)
    cat=lambda n:torch.tensor([r['sample'].get(n,0) for r in rows],device='cuda',dtype=torch.long)
    z=torch.tensor([r['sample']['aim_candidate']['source_latent'] for r in rows],device='cuda',dtype=torch.float32)
    timing=torch.tensor([r['sample']['aim_candidate']['timing'] for r in rows],device='cuda',dtype=torch.float32)
    args=(x,z,cat('target'),cat('aim_mode'),cat('vertical'),cat('weapon'),timing)
    frames=torch.arange(len(rows),device='cuda',dtype=torch.long)+100;frame=101
    override=[-18.]*25;override[4]=0.
    changed=copy.deepcopy(model);changed['diagnostic_joint_aim_probe']=dict(version=POLICY_PROBE_VERSION,frame=frame,logits=override)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):probe=JointAim(changed,allow_probe=True,allow_shared=True).requires_grad_(False)
    try:JointAim(changed,allow_shared=True)
    except AssertionError:pass
    else:raise AssertionError('Probe entered ordinary scoring')
    with torch.inference_mode():
        before,active=source(*args);after,other=probe(*args,frames=frames)
        assert torch.equal(active,other) and torch.equal(after[frames!=frame],before[frames!=frame])
        assert torch.equal(after[frames==frame],torch.tensor(override,device='cuda')[None,:])
        assert bool((after.softmax(-1)>0).all())
    assert equal_tree(frozen,source.state_dict()) and equal_tree(frozen,{k:v for k,v in probe.state_dict().items() if k!='probe_logits'})
    assert sha(a.model)==source_sha;a.out.mkdir(parents=True)
    save(a.out/'report.json',dict(state='passed',device='cuda',version=POLICY_PROBE_VERSION,rows=len(rows),
         outside_frame_logits_exact=True,source_neural_state_unchanged=True,ordinary_probe_rejected=True,training_performed=False,
         model_sha256=source_sha,dataset_sha256=seal['labels_sha256'],script_sha256=sha(__file__),scope='Numerical contract only. No new native labels or quality games.'))
    save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(json.dumps(read(a.out/'report.json')))


if __name__=='__main__':main()
