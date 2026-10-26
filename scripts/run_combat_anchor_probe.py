"""Two reused TRAIN states, baseline/identity/observed anchor, two repeats."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from process_combat_architecture_pool import read,save,sha
from combat_anchor_probe import VERSION


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--contexts',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--port',type=int,default=53800)
    a=ap.parse_args();repo=Path(__file__).resolve().parents[1];out=a.out.resolve()
    assert not out.exists() and shutil.disk_usage('F:/').free>=32*1024**3
    contexts=read(a.contexts);source=Path(contexts['source_model'])
    assert sha(source)==contexts['source_model_sha256']
    original_plan=read(repo/'workspace/artifacts/combat-stock-full-series-v3-20261026/update-10/plan.json')
    assert original_plan['model_sha256']==sha(source)
    out.mkdir(parents=True)
    os.environ['GOCACHE']=str(repo/'workspace/build/gocache')
    engine=repo/'workspace/artifacts/combat-objective-engine-v2-20261025'
    assert engine.is_dir();os.environ['Q2COOPBOT_DIAGNOSTIC_ENGINE_ROOT']=str(engine)
    def run(label,command):
        with (out/(label+'.log')).open('w',encoding='utf-8') as log:
            p=subprocess.run(list(map(str,command)),cwd=repo,stdout=log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert p.returncode==0,str(out/(label+'.log'))
    try:
        planner=out/'q2episode.exe'
        run('build-planner',['go','build','-o',planner,'./cmd/q2episode'])
        source_copy=out/'models/source.json';source_copy.parent.mkdir();shutil.copyfile(source,source_copy)
        assert sha(source_copy)==sha(source)
        plans=[];assignments=[]
        for ctx in contexts['contexts']:
            original=next(t for t in original_plan['tasks'] if t['episode']['id']==ctx['episode_id'])
            assert original['seeds']==[ctx['seed']] and original['split']=='train'
            assert original['episode']['timescale']==2
            models={'baseline':source_copy}
            for mode in ('identity','observed_mg'):
                m=read(source);m['diagnostic_anchor_probe']=dict(version=VERSION,frame=ctx['branch_frame'],mode=mode)
                path=out/'models'/(ctx['context_id']+'-'+mode+'.json');save(path,m)
                stripped=copy.deepcopy(m);stripped.pop('diagnostic_anchor_probe');assert stripped==read(source)
                models[mode]=path
            for repeat in (0,1):
                for mode,path in models.items():
                    label=ctx['context_id']+'-'+mode+'-r'+str(repeat)
                    plan=out/(label+'-plan.json')
                    run('plan-'+label,[planner,'--registry',original_plan['registry_path'],'--root',repo,
                        '--episodes',ctx['episode_id'],'--split','train','--mode','learned','--model',path,
                        '--count',1,'--seed-offset',ctx['offset'],'--checked-wall-slide',
                        '--execution-contract','stock_weapons_fire_v1','--out',plan,'--artifacts',out/('captures-'+label)])
                    p=read(plan);assert len(p['tasks'])==1
                    assert p['tasks'][0]['instances']==original['instances'],'Original TRAIN fixture changed'
                    plans.append(plan)
                    assignments.append(dict(plan_index=len(plans)-1,context_id=ctx['context_id'],mode=mode,repeat=repeat,
                        seed=ctx['seed'],branch_frame=ctx['branch_frame'],model=str(path),model_sha256=sha(path)))
        pins={str(a.contexts.resolve()):sha(a.contexts),str(source):sha(source)}
        pins.update({str(p):sha(p) for p in plans})
        for path in [*sorted((repo/'internal/policy').glob('*.go')),repo/'internal/bot/run.go',
                     *[repo/'scripts'/n for n in ('combat_anchor_probe.py','combat_joint_aim_policy.py',
                        'finalize_combat_cuda_rollout.py','finish_combat_anchor_probe.py','run_combat_anchor_probe.py')]]:
            pins[str(path)]=sha(path)
        save(out/'spec.json',dict(version=VERSION,state='prepared',contexts=contexts,assignments=assignments,
            pins=pins,max_instances=16,timescale=2,training_performed=False,promotion=False,
            scope='Selected reused TRAIN mechanism probe. Repeats test reproducibility, not independent sample size.'))
        save(out/'progress.json',dict(stage='native',cases=len(assignments)))
        quote=lambda p:"'"+str(p).replace("'","''")+"'"
        command='& '+quote(repo/'scripts/run_registered_combat_pool.ps1')+' -Plans @('+','.join(map(quote,plans))+') -MaxInstances 16 -Port '+str(a.port)+' -OutputRoot '+quote(out/'pool')
        run('native-pool',[shutil.which('pwsh'),'-NoProfile','-Command',command])
        assert read(out/'pool/report.json')['state']=='complete'
        run('finish',[sys.executable,repo/'scripts/finish_combat_anchor_probe.py','--root',out])
        assert read(out/'complete.json')['state']=='complete'
        save(out/'progress.json',dict(stage='complete',report_sha256=sha(out/'complete.json')))
    except BaseException as e:
        save(out/'progress.json',dict(stage='failed',error=str(e),exception=type(e).__name__));raise


if __name__=='__main__':main()
