"""Fresh TRAIN diagnostic baselines for later common-state aim alternatives."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from evaluate_registered_combat_models import read, save, sha
from collect_combat_joint_aim_budget_round import historical_plans


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('source','registry','protocol','out'):ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--port',type=int,default=44900);a=ap.parse_args()
    repo=Path(__file__).resolve().parents[1];out=a.out.resolve();assert not out.exists()
    assert shutil.disk_usage('F:/').free>=32*1024**3;os.environ['GOCACHE']=str(repo/'workspace/build/gocache');os.environ['GOTOOLCHAIN']='auto';pins={}
    def pin(p,expected=None):
        p=Path(p).resolve();h=sha(p);assert expected is None or h==expected,str(p);pins[str(p)]=h;return p
    seal=read(pin(a.source/'complete.json'))
    for n,k in (('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')):pin(a.source/n,seal[k])
    report=read(a.source/'report.json');assert report['device']=='cuda' and report['diagnostic_only'] and not report['training_performed']
    assert report['constant_distribution_mode']=='explicit_baseline_control' and report['requested_baseline_probability']==.999999
    registry=pin(a.registry);pin(a.protocol)
    descriptors={read(registry.parent/f)['id']:read(pin(registry.parent/f)) for f in read(registry)['files']};assert len(descriptors)==8
    assert all(e['timescale']==2 and e['status']=='runnable' for e in descriptors.values())
    history=historical_plans(repo/'workspace/artifacts');used={s for _,p in history for t in p['tasks'] for s in t.get('seeds',[])}
    offset=112
    while True:
        assert all(offset+2<=e['splits']['train']['count'] for e in descriptors.values())
        seeds={e['splits']['train']['start']+offset+i for e in descriptors.values() for i in range(2)}
        if not seeds&used:break
        offset+=2
    assert len(seeds)==16;out.mkdir(parents=True)
    def run(label,args):
        with (out/(label+'.log')).open('w',encoding='utf-8') as log:
            r=subprocess.run(list(map(str,args)),cwd=repo,stdout=log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert r.returncode==0,str(out/(label+'.log'))
    try:
        save(out/'progress.json',dict(stage='planning'));planner=repo/'workspace/build/q2episode-joint-aim.exe';pin(planner)
        plan=out/'baseline-plan.json'
        run('plan',[planner,'--registry',registry,'--root',repo,'--episodes',','.join(sorted(descriptors)),
            '--split','train','--mode','learned','--model',a.source/'weights.json','--count',2,'--seed-offset',offset,
            '--checked-wall-slide','--out',plan,'--artifacts',out/'captures'])
        value=read(plan);assert value['model_sha256']==seal['weights_sha256']
        assert {s for t in value['tasks'] for s in t['seeds']}==seeds and all(t['split']=='train' for t in value['tasks'])
        save(out/'spec.json',dict(version='combat_aim_pair_baselines_v1',source=str(a.source.resolve()),registry=str(registry),
            protocol=str(a.protocol.resolve()),protocol_sha256=sha(a.protocol),inputs=pins,split='train',offset=offset,
            seeds=sorted(seeds),native_cases=16,max_instances=16,timescale=2,training_performed=False,promotion=False,
            historical_plans=[str(p) for p,_ in history],unavailable_historical_references=historical_plans.missing_references,
            scope='Fresh diagnostic baseline captures only; neither quality comparison nor an ordinary PPO training assignment.'))
        shell=shutil.which('pwsh');assert shell;quote=lambda p:"'"+str(p).replace("'","''")+"'"
        command='& '+quote(repo/'scripts/run_registered_combat_pool.ps1')+' -Plans @('+quote(plan)+') -MaxInstances 16 -Port '+str(a.port)+' -OutputRoot '+quote(out/'pool')
        save(out/'progress.json',dict(stage='native-baselines',cases=16))
        run('native-pool',[shell,'-NoProfile','-Command',command])
        pool=read(out/'pool/report.json');assert pool['state']=='complete' and pool['source_unchanged'] and pool['usable_captures']==16
        save(out/'progress.json',dict(stage='cuda-baselines',cases=16))
        run('processing',[sys.executable,repo/'scripts/process_registered_combat_pool_cuda.py','--pool',out/'pool','--plan-index',0,'--out',out/'processing'])
        meta=read(out/'processing/report.json');assert meta['state']=='complete' and meta['device']=='cuda' and len(meta['cases'])==16
        assert not meta['training_performed'] and {c['seed'] for c in meta['cases']}==seeds
        for p,h in pins.items():assert sha(Path(p))==h,p
        save(out/'complete.json',dict(state='complete',version='combat_aim_pair_baselines_v1',native_cases=16,cuda_cases=16,
            spec_sha256=sha(out/'spec.json'),pool_sha256=sha(out/'pool/report.json'),processing_sha256=sha(out/'processing/report.json'),
            training_performed=False,promotion=False,quality_confirmation=False))
        save(out/'progress.json',dict(stage='complete',native_cases=16,cuda_cases=16));print(out/'complete.json')
    except BaseException as e:
        save(out/'progress.json',dict(stage='failed',error=str(e),exception=type(e).__name__));raise


if __name__=='__main__':main()
