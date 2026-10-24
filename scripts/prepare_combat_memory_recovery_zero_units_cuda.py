"""Select observable memory-active TRAIN fixtures and bind CUDA zero-core proofs."""
import argparse
from pathlib import Path
import numpy as np
from evaluate_registered_combat_models import read,save,sha
from ppo_combat import torch,training_devices
from combat_memory_recovery_v2 import active_mask
from combat_memory_recovery_runtime import build,apply


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();h=sha(path);assert expected is None or h==expected,str(path);pins[str(path)]=h;return path
    p=read(pin(a.protocol));assert p['version']=='combat_memory_recovery_zero_compatibility_v1'
    models=Path(p['models']);ms=read(pin(models/'complete.json'))
    mr=read(pin(models/'report.json',ms['report_sha256']));assert mr['device']=='cuda' and mr['zero_source_raw_exact'] and mr['source_parameters_exact']
    cache=Path(p['source_cache']);cs=read(pin(cache/'complete.json'))
    cr=read(pin(cache/'report.json',cs['report_sha256']));assert cr['source_weights_sha256']==p['source_weights_sha256']
    corpus=Path(p['corpus']).resolve();closed=read(pin(corpus/'complete.json'))
    cp=read(pin(corpus/'protocol.json',closed['protocol_sha256']));assignment=cp['assignments'][0]
    plan=read(pin(assignment['plan'],assignment['plan_sha256']));pool=read(pin(corpus/'pool/report.json',closed['pool_sha256']))
    jobs={j['seed']:j for j in pool['jobs']};pin(p['registry'])
    training_devices();torch.set_default_device('cuda');torch.set_num_threads(2)
    candidates=[]
    for chunk in cr['chunks']:
        if chunk['split']!='train':continue
        with np.load(pin(chunk['path'],chunk['sha256'])) as d:
            x=torch.tensor(d['features'],device='cuda');active=active_mask(x)
            if not bool(active.any()):continue
            f=int(torch.tensor(d['frame'],device='cuda')[active].min())
            candidates.append(dict(chunk=chunk,frame=f,seed=chunk['case']['seed'],map=chunk['case']['map']))
    selected=[min((c for c in candidates if c['map']==mp),key=lambda c:c['seed']) for mp in sorted({c['map'] for c in candidates})]
    assert len(selected)<=4
    chosen={c['seed'] for c in selected}
    for c in sorted(candidates,key=lambda c:c['seed']):
        if len(selected)==4:break
        if c['seed'] not in chosen:selected.append(c);chosen.add(c['seed'])
    assert len(selected)==4
    zero_info=mr['models']['zero'];zero_model=read(pin(zero_info['path'],zero_info['sha256']));head=build(zero_model)
    source=Path(mr['source']);pin(source/'weights.json',p['source_weights_sha256'])
    a.out.mkdir(parents=True);units=[]
    for c in selected:
        job=jobs[c['seed']];task=plan['tasks'][job['task_index']];slot=task['seeds'].index(c['seed']);chunk=c['chunk']
        context=dict(context_id=f"memory-zero-{c['seed']}-{c['frame']}",seed=c['seed'],frame=c['frame'],job=job['job'],split='train',
            role='technical_zero_compatibility',episode=task['episode']['id'],instance=task['instances'][slot],offset=assignment['offset']+slot,directory=chunk['case']['folder'])
        with np.load(chunk['path']) as d:
            x=torch.tensor(d['features'],device='cuda');raw=torch.tensor(d['raw_actor'],device='cuda')
            with torch.no_grad():assert torch.equal(apply(head,x,raw),raw)
            active_rows=int(active_mask(x).sum());assert active_rows>0
        variants=[dict(index=0,label='control',model=str((source/'weights.json').resolve()),model_sha256=p['source_weights_sha256']),
            dict(index=1,label='zero',model=zero_info['path'],model_sha256=zero_info['sha256'])]
        target=a.out/context['context_id'];target.mkdir()
        report=dict(state='complete',version='combat_memory_recovery_zero_unit_v1',device='cuda',context=context,variants=variants,
            source=str(source),source_weights_sha256=p['source_weights_sha256'],baseline=str(corpus),registry=str(Path(p['registry']).resolve()),
            inputs=dict(pins),neural_parameters_exact=True,zero_source_raw_exact=True,cuda_checked_rows=len(x),cuda_active_rows=active_rows,
            actor_integration_proof=str((models/'report.json').resolve()),actor_integration_proof_sha256=sha(models/'report.json'),
            training_performed=False,promotion=False)
        save(target/'report.json',report);save(target/'complete.json',dict(state='complete',report_sha256=sha(target/'report.json')))
        units.append(dict(unit_id=context['context_id'],context=context,prepared=str(target.resolve())))
    for path,h in pins.items():assert sha(path)==h,path
    report=dict(state='complete',version='combat_memory_recovery_units_preparation_v1',device='cuda',units=units,protocol_sha256=sha(a.protocol),
        maps=sorted({c['map'] for c in selected}),native_cases=8,cuda_cases=8,inputs=pins,training_performed=False,promotion=False)
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(dict(contexts=4,maps=report['maps'],cases=8))


if __name__=='__main__':main()
