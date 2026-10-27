"""Native disabled parity, held-option sampling and CUDA joint-density proof."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from audit_combat_checked_ground_native_probe import normalized, same
from process_combat_architecture_pool import read, save, sha


def run(protocol,root,out):
    assert not out.exists()
    spec,pool=read(protocol),read(root/'report.json')
    assert spec['version']=='combat_movement_options_native_smoke_v1'
    assert pool['state']=='complete' and pool['source_unchanged'] and len(pool['jobs'])==6
    assert pool['protocol_sha256']==sha(protocol)
    pins={**spec['source_files'],str(protocol):sha(protocol),str(Path(__file__).resolve()):sha(Path(__file__))}
    assert all(sha(Path(p))==digest for p,digest in pins.items())
    out.mkdir(parents=True);shutil.copyfile(__file__,out/Path(__file__).name)
    env=dict(os.environ,GOCACHE=str(Path.cwd()/'workspace/build/gocache'))
    def command(label,cmd):
        with (out/(label+'.log')).open('w',encoding='utf-8') as log:
            p=subprocess.run(list(map(str,cmd)),env=env,stdout=log,stderr=log)
        assert p.returncode==0,label
    exporter=(out/'q2ppo-data.exe').resolve();command('build',['go','build','-o',exporter,'./cmd/q2ppo-data'])
    groups,tasks={},[]
    for job in sorted(pool['jobs'],key=lambda j:j['job']):
        assert not job['error']
        cap=read(Path(job['root'])/'report.json');row=cap['results'][0]
        assert cap['capture_complete'] and cap['provenance_valid'] and row['capture_valid'] and row['dispatch_valid'] and row['seed_confirmed']
        target=(out/f"case-{job['job']}").resolve();target.mkdir();model=Path(spec['variants'][job['label']]['path'])
        assert sha(model)==spec['variants'][job['label']]['sha256']
        command(f"export-{job['job']}",[exporter,'--batch',job['root'],'--model',model,'--cuda-only','--out',target/'native'])
        path=target/'native/replay-0/steps.jsonl';pins[str(path)]=sha(path)
        steps=[json.loads(line) for line in path.open(encoding='utf-8-sig')]
        groups[job['label'],job['repeat']]=steps
        tasks.append(dict(model=str(model),model_sha256=sha(model),data=str(target/'native'),out=str(target/'cuda'),native_report_sha256=sha(target/'native/report.json')))
    def signature(s):
        sample={k:v for k,v in (s.get('sample') or {}).items() if k!='version'}
        return normalized(dict(observation=s['observation'],action=s['action'],applied_action=s['applied_action'],
            sent_command=s['sent_command'],sample=sample,next_observation=s.get('next_observation')))
    checks={}
    for label in ('source','disabled','enabled'):
        a,b=groups[label,0],groups[label,1]
        checks[label+'/cold_repeat']=len(a)==len(b) and all(same(signature(x),signature(y),1e-9) for x,y in zip(a,b))
    for repeat in (0,1):
        a,b=groups['source',repeat],groups['disabled',repeat]
        checks[f'disabled/{repeat}/full_native_parity']=len(a)==len(b) and all(same(signature(x),signature(y),1e-9) for x,y in zip(a,b))
    enabled=groups['enabled',0]
    events=[s['sample']['movement_option'] for s in enabled if s.get('sample') and s['sample'].get('movement_option')]
    checks['enabled_events']=bool(events) and sum(r['decision'] for r in events)>1
    checks['enabled_nonzero_choices']=any(r['choice']!=0 for r in events)
    source_by_frame={s['observation']['identity']['frame']:s for s in groups['source',0]}
    differences=sum(s['sent_command'][key]!=source_by_frame[s['observation']['identity']['frame']]['sent_command'][key]
        for s in enabled if s['observation']['identity']['frame'] in source_by_frame for key in ('Forward','Side'))
    checks['enabled_native_movement_changed']=differences>0
    save(out/'native-checks.json',dict(checks=checks,events=len(events),decisions=sum(r['decision'] for r in events),native_movement_component_differences=differences))
    save(out/'cuda-request.json',dict(version='combat_cuda_finalize_batch_v1',compact_finalized_streams=True,tasks=tasks))
    command('cuda',[sys.executable,'scripts/finalize_combat_cuda_batch.py','--request',out/'cuda-request.json','--receipt-root',out/'cuda-batch'])
    for index,task in enumerate(tasks):
        meta,proof=read(Path(task['out'])/'report.json'),read(Path(task['out'])/'cuda-verification.json')
        assert proof['state']=='passed' and proof['device']=='cuda'
        if index>=2:
            assert meta['version']=='combat_movement_options_rollout_v1' and not meta['ppo_training_ready']
            assert proof['movement_options_version']=='combat_five_frame_movement_options_v1' and not proof['ordinary_training_eligible']
            assert proof['movement_options']['enabled']==(index>=4)
    assert all(checks.values()),checks
    assert all(sha(Path(p))==digest for p,digest in pins.items())
    save(out/'report.json',dict(state='complete',checks=checks,events=len(events),decisions=sum(r['decision'] for r in events),
        native_movement_component_differences=differences,native_cases=6,cuda_cases=6,inputs=pins,
        training_performed=False,promotion=False,scope=spec['scope']))
    save(out/'complete.json',dict(state='complete',report_sha256=sha(out/'report.json')))
    print(json.dumps(dict(checks=checks,events=len(events),decisions=sum(r['decision'] for r in events)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('protocol','root','out'):p.add_argument('--'+key,required=True,type=Path)
    a=p.parse_args();run(a.protocol.resolve(),a.root.resolve(),a.out.resolve())
