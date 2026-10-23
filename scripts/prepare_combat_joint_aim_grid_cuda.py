"""Seal full-support diagnostic constant heads for one TRAIN common-state grid."""
import argparse
import copy
import json
from pathlib import Path
from evaluate_registered_combat_models import read, save, sha
from ppo_combat import torch, training_devices, layers
from combat_joint_aim_policy import JointAim, events, validate
from combat_aim_candidates import choose, frame_uniforms


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('reference','audit','protocol','out'):ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(p,expected=None):
        p=Path(p).resolve();h=sha(p);assert expected is None or h==expected,str(p);pins[str(p)]=h;return p
    closed=read(pin(a.reference/'complete.json'));assert closed['native_cases']==closed['cuda_cases']==2
    spec=read(pin(a.reference/'spec.json',closed['spec_sha256']))
    assert spec['split']=='train' and spec['selected'][0]['seed']==1252025109 and len(spec['selected'])==1
    source=Path(spec['variants'][0]['update']);seal=read(pin(source/'complete.json'))
    model=read(pin(source/'weights.json',seal['weights_sha256']));pin(source/'checkpoint.pt',seal['checkpoint_sha256'])
    audit_seal=read(pin(a.audit/'complete.json'));audit=read(pin(a.audit/'report.json',audit_seal['report_sha256']))
    assert audit['model_sha256']==seal['weights_sha256'] and not audit['validation_used'] and not audit['training_performed']
    pool=read(pin(a.reference/'pool/report.json',closed['pool_sha256']));job=next(j for j in pool['jobs'] if j['plan_index']==0)
    folder=a.reference/'processing-source'/f"case-{job['job']:04d}"/'cuda';meta=read(pin(folder/'report.json'))
    rows=list(map(json.loads,pin(folder/'rollout.jsonl',meta['rollout_sha256']).open(encoding='utf-8-sig')))
    rows=[r for r in rows if r['frame']<=146];assert rows[-1]['frame']==146
    assert all(r['seed']==1252025109 for r in rows);pin(a.protocol)
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    joint=JointAim(model).requires_grad_(False)
    x=torch.tensor([r['features'] for r in rows],device='cuda',dtype=torch.float32)
    cats=lambda n:torch.tensor([r['sample'].get(n,0) for r in rows],device='cuda',dtype=torch.long)
    z,old,timing,active=events(rows,x,cats('target'),cats('aim_mode'),cats('vertical'),cats('weapon'))
    assert bool(active[-1]) and int(old[-1])==15
    uniforms=torch.tensor([frame_uniforms(r['sample']['sampling_seed'],r['frame'],1) for r in rows],device='cuda',dtype=torch.float64)
    a.out.mkdir(parents=True);variants=[]
    for k in range(25):
        p0=.999999 if k==0 else audit['calibrated_marginal_probabilities'][0]
        q=torch.full((25,),(1-p0)/24 if k==0 else (1-p0)*1e-8,device='cuda',dtype=torch.float64)
        q[0]=p0
        if k:q[k]=(1-p0)*(1-23e-8)
        assert bool((q>0).all()) and abs(float(q.sum())-1)<1e-10
        with torch.no_grad():joint.head.layers[-1].weight.zero_();joint.head.layers[-1].bias.copy_(q.log().float())
        updated=copy.deepcopy(model);updated['aim_candidate_head']['layers']=layers(joint.head.layers);validate(updated)
        before=copy.deepcopy(model);after=copy.deepcopy(updated)
        before['aim_candidate_head']['layers']=None;after['aim_candidate_head']['layers']=None;assert before==after
        assert updated['aim_candidate_head']['layers'][0]==model['aim_candidate_head']['layers'][0]
        exported=JointAim(updated).requires_grad_(False)
        with torch.no_grad():
            logits=exported.head(torch.zeros((len(rows),1359),device='cuda')).double()
            picked=choose(logits,uniforms[:,0],uniforms[:,1],active)
            assert bool((picked[:-1]==0).all()) and int(picked[-1])==k,(k,picked.tolist())
        root=a.out/'models'/f'category-{k:02d}';root.mkdir(parents=True);save(root/'weights.json',updated)
        variants.append(dict(category=k,model=str((root/'weights.json').resolve()),model_sha256=sha(root/'weights.json'),
            probabilities=logits[-1].softmax(-1).tolist(),prefix_choices_zero=True,intervention_frame=146,
            offset_yaw_degrees=(0,-5,-2.5,2.5,5)[k//5],offset_pitch_degrees=(0,-5,-2.5,2.5,5)[k%5]))
    for p,h in pins.items():assert sha(Path(p))==h,p
    save(a.out/'report.json',dict(state='complete',version='combat_joint_aim_grid_preparation_v1',device='cuda',
        source=str(source.resolve()),reference=str(a.reference.resolve()),source_weights_sha256=seal['weights_sha256'],
        seed=1252025109,episode=spec['selected'][0]['episode'],offset=109,frame=146,variants=variants,
        protocol=str(a.protocol.resolve()),protocol_sha256=sha(a.protocol),inputs=pins,script_sha256=sha(__file__),
        diagnostic_only=True,training_performed=False,validation_used=False,promotion=False,
        scope='Only last head layer differs. No training checkpoints created. Prefix choices verified on fixed historical TRAIN inputs; native reproduction pending.'))
    save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print('25 diagnostic heads sealed on CUDA; historical category prefixes verified')


if __name__=='__main__':main()
