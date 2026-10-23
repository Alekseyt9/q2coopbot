"""Four intentional TRAIN replays, selected by head disagreement, not outcome."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from evaluate_registered_combat_models import read, save, sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('source','control','corpus','audit','registry','protocol','out'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--port',type=int,default=44600)
    ap.add_argument('--seed',type=int,action='append',help='Explicit subset of the original two diagnostic TRAIN seeds')
    a=ap.parse_args();repo=Path(__file__).resolve().parents[1];out=a.out.resolve()
    assert not out.exists() and shutil.disk_usage('F:/').free>=32*1024**3
    os.environ['GOCACHE']=str(repo/'workspace/build/gocache');os.environ['GOTOOLCHAIN']='auto'
    pins={}
    def pin(p,expected=None):
        p=Path(p).resolve();h=sha(p);assert expected is None or h==expected,str(p);pins[str(p)]=h;return p
    audit_seal=read(pin(a.audit/'complete.json'));audit=read(pin(a.audit/'report.json',audit_seal['report_sha256']))
    assert audit['device']=='cuda' and not audit['validation_used'] and not audit['training_performed']
    old_closed=read(pin(a.corpus/'complete.json'));old_protocol=read(pin(a.corpus/'protocol.json',old_closed['protocol_sha256']))
    assert old_protocol['train_only'] and len(old_protocol['assignments'])==1
    assignment=old_protocol['assignments'][0];assert assignment['split']=='train'
    original=read(pin(assignment['plan'],assignment['plan_sha256']))
    registry=a.registry.resolve();pin(registry);pin(a.protocol)
    descriptors={read(registry.parent/f)['id']:read(registry.parent/f) for f in read(registry)['files']}
    instances={i['engine_seed']:(t['episode']['id'],i) for t in original['tasks'] for i in t['instances']}
    candidates=[]
    for c in audit['datasets']['round4']['cases']:
        if not c['marginal_control_different_categories']:continue
        episode,instance=instances[c['seed']];offset=c['seed']-descriptors[episode]['splits']['train']['start']
        candidates.append((offset,episode,c['seed']))
    assert candidates
    minimum=min(c[0] for c in candidates);selected=sorted((c for c in candidates if c[0]==minimum),key=lambda c:c[1])[:2]
    assert len(selected)==2 and len({c[1] for c in selected})==2
    assert [c[2] for c in selected]==[1172065109,1252025109], 'Selection differs from frozen protocol'
    if a.seed is not None:
        assert len(a.seed)==len(set(a.seed)) and set(a.seed)<={c[2] for c in selected}
        selected=[c for c in selected if c[2] in a.seed]
    case_count=2*len(selected)
    roots=dict(source=a.source.resolve(),marginal=a.control.resolve());models={}
    for label,root in roots.items():
        seal=read(pin(root/'complete.json'))
        for name,key in (('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')):pin(root/name,seal[key])
        models[label]=read(root/'weights.json');assert read(root/'report.json')['device']=='cuda'
    assert sha(roots['source']/'weights.json')==audit['model_sha256']==original['model_sha256']
    before=read(roots['source']/'weights.json');after=read(roots['marginal']/'weights.json')
    before['aim_candidate_head']['layers']=None;after['aim_candidate_head']['layers']=None;assert before==after
    assert read(roots['marginal']/'report.json')['diagnostic_only']
    out.mkdir(parents=True)
    def run(label,command):
        with (out/(label+'.log')).open('w',encoding='utf-8') as log:
            r=subprocess.run(list(map(str,command)),cwd=repo,stdout=log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert r.returncode==0,str(out/(label+'.log'))
    try:
        save(out/'progress.json',dict(stage='preparing'))
        planner=repo/'workspace/build/q2episode-joint-aim.exe';assert planner.is_file()
        plans=[];fixtures=[]
        for label,root in roots.items():
            plan=out/(label+'-plan.json')
            run('plan-'+label,[planner,'--registry',registry,'--root',repo,'--episodes',','.join(c[1] for c in selected),
                '--split','train','--mode','learned','--model',root/'weights.json','--count',1,'--seed-offset',minimum,
                '--checked-wall-slide','--out',plan,'--artifacts',out/('captures-'+label)])
            value=read(plan);assert len(value['tasks'])==len(selected)
            for task in value['tasks']:
                assert task['split']=='train' and task['modes']==['learned'] and len(task['seeds'])==1
                seed=task['seeds'][0];assert task['instances']==[instances[seed][1]],'Fixture changed since TRAIN capture'
            current={t['episode']['id']:t['instances'] for t in value['tasks']}
            if fixtures:assert fixtures[0]==current
            fixtures.append(current);plans.append(plan)
        spec=dict(version='combat_joint_aim_context_probe_v1',split='train',registry=str(registry),
            variants=[dict(label=label,mode='learned',update=str(root),deterministic=False,cuda_replay=True) for label,root in roots.items()],
            selected=[dict(episode=e,seed=s,offset=o) for o,e,s in selected],seed_reuse_intentional=True,
            selection='Lowest TRAIN offset with marginal disagreement; episode ID order; optional explicit subset fixed by diagnostic protocol',
            explicit_seed_subset=a.seed,
            source_corpus=str(a.corpus.resolve()),source_audit=str(a.audit.resolve()),max_instances=16,timescale=2,
            protocol_sha256=sha(a.protocol),inputs=pins,training_performed=False,promotion=False)
        save(out/'spec.json',spec)
        shell=shutil.which('pwsh');assert shell
        quote=lambda p:"'"+str(p).replace("'","''")+"'"
        command='& '+quote(repo/'scripts/run_registered_combat_pool.ps1')+' -Plans @('+','.join(map(quote,plans))+') -MaxInstances 16 -Port '+str(a.port)+' -OutputRoot '+quote(out/'pool')
        save(out/'progress.json',dict(stage='native-probe',cases=case_count))
        run('native-pool',[shell,'-NoProfile','-Command',command])
        pool=read(out/'pool/report.json');assert pool['state']=='complete' and pool['source_unchanged'] and pool['usable_captures']==case_count
        run('finalize',[sys.executable,repo/'scripts/finish_combat_joint_aim_context_probe.py','--root',out])
        for p,h in pins.items():assert sha(Path(p))==h,'Input changed: '+p
        assert read(out/'complete.json')['state']=='complete'
        save(out/'progress.json',dict(stage='complete',native_cases=case_count,cuda_cases=case_count));print(out/'complete.json')
    except BaseException as e:
        save(out/'progress.json',dict(stage='failed',error=str(e),exception=type(e).__name__));raise


if __name__=='__main__':main()
