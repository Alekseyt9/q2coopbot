"""Nine native runs for legacy/disabled/enabled event compatibility, not quality."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from process_combat_architecture_pool import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('plan','source','migration','audit','out'):ap.add_argument('--'+k,type=Path,required=True)
    ap.add_argument('--port',type=int,default=54000);a=ap.parse_args()
    repo=Path(__file__).resolve().parents[1];out=a.out.resolve();assert not out.exists() and shutil.disk_usage('F:/').free>=32*1024**3
    audit=read(a.audit/'complete.json');assert audit['state']=='complete' and audit['device']=='cuda'
    original=read(a.plan);assert sha(a.source/'weights.json')==original['model_sha256']
    models=dict(legacy=(a.source/'weights.json').resolve(),disabled=(a.migration/'disabled/weights.json').resolve(),enabled=(a.migration/'enabled/weights.json').resolve())
    selected=[t for t in original['tasks'] if t['seeds'][0] in (1844000142,1865000142,1892000142)]
    assert len(selected)==3 and all(t['split']=='train' and len(t['seeds'])==1 for t in selected)
    out.mkdir(parents=True);(out/'frozen-scripts').mkdir()
    os.environ['GOCACHE']=str(repo/'workspace/build/gocache')
    os.environ['Q2COOPBOT_DIAGNOSTIC_ENGINE_ROOT']=str(repo/'workspace/artifacts/combat-objective-engine-v2-20261025')
    def run(label,cmd):
        with (out/(label+'.log')).open('w',encoding='utf-8') as log:
            p=subprocess.run(list(map(str,cmd)),cwd=repo,stdout=log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert p.returncode==0,str(out/(label+'.log'))
    try:
        planner=out/'q2episode.exe';run('build-planner',['go','build','-o',planner,'./cmd/q2episode'])
        assignments=[];plans=[];pins={str(a.plan.resolve()):sha(a.plan),str((a.audit/'report.json').resolve()):sha(a.audit/'report.json')}
        for t in selected:
            seed=t['seeds'][0];episode=t['episode']['id'];offset=seed-t['episode']['splits']['train']['start']
            for name,model in models.items():
                label=str(seed)+'-'+name;plan=out/(label+'-plan.json')
                run('plan-'+label,[planner,'--registry',original['registry_path'],'--root',repo,'--episodes',episode,
                    '--split','train','--mode','learned','--model',model,'--count',1,'--seed-offset',offset,'--checked-wall-slide',
                    '--execution-contract','stock_weapons_fire_v1','--out',plan,'--artifacts',out/('captures-'+label)])
                current=read(plan);assert len(current['tasks'])==1 and current['tasks'][0]['instances']==t['instances']
                plans.append(plan);pins[str(plan)]=sha(plan);pins[str(model)]=sha(model)
                assignments.append(dict(plan_index=len(plans)-1,seed=seed,map=t['episode']['map'],variant=name,model=str(model),model_sha256=sha(model)))
        for name in ('combat_wide_aim.py','combat_joint_aim_policy.py','combat_full_joint_ppo.py','finalize_combat_cuda_rollout.py',
                     'finish_combat_wide_aim_compatibility.py','run_combat_wide_aim_compatibility.py','recover_combat_anchor_screen.py'):
            path=repo/'scripts'/name;pins[str(path)]=sha(path);shutil.copyfile(path,out/'frozen-scripts'/name)
        save(out/'spec.json',dict(version='combat_wide_aim_native_compatibility_v1',assignments=assignments,pins=pins,
            max_instances=16,timescale=2,seed_reuse_intentional=True,purpose='diagnostic_compatibility_only',
            training_performed=False,promotion=False,quality_confirmation=False))
        quote=lambda p:"'"+str(p).replace("'","''")+"'"
        command='& '+quote(repo/'scripts/run_registered_combat_pool.ps1')+' -Plans @('+','.join(map(quote,plans))+') -MaxInstances 16 -Port '+str(a.port)+' -OutputRoot '+quote(out/'pool')
        save(out/'progress.json',dict(stage='native',cases=9))
        run('native-pool',[shutil.which('pwsh'),'-NoProfile','-Command',command])
        run('finish',[sys.executable,repo/'scripts/finish_combat_wide_aim_compatibility.py','--root',out])
        assert read(out/'complete.json')['state']=='complete'
        save(out/'progress.json',dict(stage='complete',complete_sha256=sha(out/'complete.json')))
    except BaseException as error:
        save(out/'progress.json',dict(stage='failed',error=str(error),exception=type(error).__name__));raise


if __name__=='__main__':main()
