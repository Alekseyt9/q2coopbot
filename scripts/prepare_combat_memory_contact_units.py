"""Select observable TRAIN contact-loss roots, then prepare fixed CUDA movement windows."""
import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from evaluate_registered_combat_models import read,save,sha

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();h=sha(path);assert expected is None or h==expected,str(path);pins[str(path)]=h;return path
    pin(__file__);p=read(pin(a.protocol));assert p['version']=='combat_memory_contact_movement_units_v1'
    ds=Path(p['dataset']);seal=read(pin(ds/'complete.json'));dataset=read(pin(ds/'report.json',seal['report_sha256']))
    assert dataset['version']=='combat_target_lifecycle_dataset_v1'
    corpus=Path(p['corpus']).resolve();closed=read(pin(corpus/'complete.json'));assert closed['native_cases']==closed['cuda_cases']==32
    collection=read(pin(corpus/'protocol.json',closed['protocol_sha256']));assignment=collection['assignments'][0]
    plan=read(pin(assignment['plan'],assignment['plan_sha256']));pool=read(pin(corpus/'pool/report.json',closed['pool_sha256']))
    assert pool['source_unchanged'];source=Path(p['source']).resolve();pin(source/'weights.json',p['source_weights_sha256']);pin(p['registry'])
    contexts=[];coverage=[]
    for chunk in sorted((c for c in dataset['chunks'] if c['split']=='train'),key=lambda c:c['case']['seed']):
        with np.load(pin(chunk['path'],chunk['sha256'])) as data:ids=set(map(int,data['step'][data['phase']==1]))
        folder=Path(chunk['case']['folder']);meta=read(pin(folder/'cuda/report.json'));assert meta['model_sha256']==p['source_weights_sha256']
        sp=next((folder/'native').glob('replay-*/steps.jsonl'));pin(sp,meta['source_sha256'][str(sp.resolve())])
        steps=list(map(json.loads,sp.open(encoding='utf-8-sig')));choices=[]
        for s in steps:
            if s['index'] not in ids:continue
            o=s['observation'];mem=sorted(o.get('remembered_threats') or [],key=lambda t:(t['age_frames'],t['id']))
            if not o['on_ground'] or o['enemies'] or not mem or not 1<=mem[0]['age_frames']<=10:continue
            assert s['owner']=='provider' and s.get('sample') and o['health']>0 and o['identity']['life']==1
            choices.append((o['identity']['frame'],mem[0]['age_frames'],s['index']))
        seed=chunk['case']['seed'];coverage.append(dict(seed=seed,map=chunk['case']['map'],eligible_roots=len(choices)))
        if not choices:continue
        frame,age,index=min(choices);job=next(j for j in pool['jobs'] if j['seed']==seed);task=plan['tasks'][job['task_index']];slot=task['seeds'].index(seed)
        assert slot!=3 and task['split']=='train'
        contexts.append(dict(frame=frame,observed_enemies=0,memory_age_frames=age,source_step=index,context_id=f'contact-{seed}-{frame}',seed=seed,job=job['job'],
            split='train',role='auxiliary_train',episode=task['episode']['id'],instance=task['instances'][slot],offset=assignment['offset']+slot,directory=str(folder.resolve())))
    chosen=[min((c for c in contexts if c['instance']['map']==mp),key=lambda c:c['seed']) for mp in p['maps']]
    assert len(chosen)==len({c['seed'] for c in chosen})==4
    helper=pin(Path(__file__).with_name('prepare_combat_memory_contact_window_cuda.py'));a.out.mkdir(parents=True);units=[]
    for c in chosen:
        folder=a.out/c['context_id'];folder.mkdir();metadata=folder/'context';metadata.mkdir()
        save(metadata/'report.json',dict(state='complete',version='combat_observed_memory_contact_context_v1',device='metadata_only',context=c,
            source=str(source),baseline=str(corpus),registry=str(Path(p['registry']).resolve()),inputs=copy.deepcopy(pins),training_performed=False,neural_execution=False))
        save(metadata/'complete.json',dict(state='complete',report_sha256=sha(metadata/'report.json')))
        protocol=dict(version='combat_movement_window_smoke_v1',context_id=c['context_id'],variants=p['variants'],source_weights_sha256=p['source_weights_sha256'],
            native_cases=len(p['variants']),cuda_cases=len(p['variants']),fresh_ordinary_reference=False,max_instances=16,timescale=2)
        save(folder/'protocol.json',protocol);target=folder/'cuda'
        with (folder/'prepare.log').open('w',encoding='utf-8') as log:
            done=subprocess.run([sys.executable,str(helper),'--prepared',str(metadata),'--protocol',str(folder/'protocol.json'),'--out',str(target)],stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert done.returncode==0,str(folder/'prepare.log')
        seal=read(pin(target/'complete.json'));r=read(pin(target/'report.json',seal['report_sha256']))
        assert r['device']=='cuda' and r['neural_parameters_exact'] and r['zero_control_exact']
        units.append(dict(context=c,prepared=str(target.resolve()),unit_id=c['context_id']))
        save(a.out/'progress.json',dict(stage='cuda-preparing',contexts_done=len(units),contexts=4))
    for path,h in pins.items():assert sha(path)==h,path
    save(a.out/'report.json',dict(state='complete',version='combat_movement_units_preparation_v1',device='cuda',units=units,coverage=coverage,
        protocol_sha256=sha(a.protocol),source_weights_sha256=p['source_weights_sha256'],inputs=pins,training_performed=False,promotion=False))
    save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')));save(a.out/'progress.json',dict(stage='complete',contexts=4));print(chosen)

if __name__=='__main__':main()
