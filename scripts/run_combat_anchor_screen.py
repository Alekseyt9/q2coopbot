"""Frozen multi-parent TRAIN mechanism screen; no fitting or quality promotion."""
import argparse
import copy
import os
from pathlib import Path
import shutil
import subprocess
import sys
from combat_anchor_probe import VERSION
from process_combat_architecture_pool import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--contexts',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--port',type=int,default=53900)
    a=ap.parse_args();repo=Path(__file__).resolve().parents[1];out=a.out.resolve()
    assert not out.exists() and shutil.disk_usage('F:/').free>=32*1024**3
    selected=read(a.contexts);assert selected['state']=='prepared' and len(selected['contexts'])==8
    for p,h in selected['pins'].items():assert sha(p)==h
    out.mkdir(parents=True);(out/'models').mkdir();(out/'frozen-scripts').mkdir()
    os.environ['GOCACHE']=str(repo/'workspace/build/gocache')
    os.environ['Q2COOPBOT_DIAGNOSTIC_ENGINE_ROOT']=str(repo/'workspace/artifacts/combat-objective-engine-v2-20261025')
    def run(label,command):
        with (out/(label+'.log')).open('w',encoding='utf-8') as log:
            p=subprocess.run(list(map(str,command)),cwd=repo,stdout=log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert p.returncode==0,str(out/(label+'.log'))
    try:
        planner=out/'q2episode.exe';run('build-planner',['go','build','-o',planner,'./cmd/q2episode'])
        plans=[];assignments=[];pins={str(a.contexts.resolve()):sha(a.contexts)}
        for ctx in selected['contexts']:
            old_plan=Path(ctx['original_plan']);assert sha(old_plan)==ctx['original_plan_sha256']
            old=read(old_plan);task=next(t for t in old['tasks'] if t['episode']['id']==ctx['episode_id'])
            assert task['split']=='train' and task['seeds']==[ctx['seed']]
            source=Path(ctx['source_model']);assert sha(source)==ctx['source_model_sha256']
            source_copy=out/'models'/(sha(source)+'.json')
            if not source_copy.exists():shutil.copyfile(source,source_copy)
            for mode in ('baseline','identity','observed_mg'):
                model=source_copy
                if mode!='baseline':
                    v=read(source);v['diagnostic_anchor_probe']=dict(version=VERSION,frame=ctx['branch_frame'],mode=mode)
                    model=out/'models'/(ctx['context_id']+'-'+mode+'.json');save(model,v)
                    v.pop('diagnostic_anchor_probe');assert v==read(source)
                label=ctx['context_id']+'-'+mode;plan=out/(label+'-plan.json')
                run('plan-'+label,[planner,'--registry',old['registry_path'],'--root',repo,'--episodes',ctx['episode_id'],
                    '--split','train','--mode','learned','--model',model,'--count',1,'--seed-offset',ctx['offset'],
                    '--checked-wall-slide','--execution-contract','stock_weapons_fire_v1','--out',plan,'--artifacts',out/('captures-'+label)])
                v=read(plan);assert len(v['tasks'])==1 and v['tasks'][0]['instances']==task['instances']
                assert v['registry_sha256']==old['registry_sha256']
                plans.append(plan);pins[str(plan)]=sha(plan);pins[str(model)]=sha(model)
                assignments.append(dict(plan_index=len(plans)-1,context_id=ctx['context_id'],mode=mode,repeat=0,
                    seed=ctx['seed'],branch_frame=ctx['branch_frame'],model=str(model),model_sha256=sha(model)))
        scripts=('run_combat_anchor_screen.py','finish_combat_anchor_screen.py','finish_combat_anchor_probe.py',
                 'combat_anchor_probe.py','combat_joint_aim_policy.py','finalize_combat_cuda_rollout.py',
                 'report_combat_anchor_probe.py','report_combat_anchor_continuation.py')
        for name in scripts:
            path=repo/'scripts'/name;pins[str(path)]=sha(path);shutil.copyfile(path,out/'frozen-scripts'/name)
        save(out/'spec.json',dict(version='combat_observed_anchor_other_maps_screen_v1',contexts=selected,
            assignments=assignments,pins=pins,max_instances=16,timescale=2,training_performed=False,promotion=False,
            source_event_likelihood_only=True,quality_confirmation=False,
            scope='Eight outcome-blind reused TRAIN roots on other maps, each with its own frozen source. Within-context intervention comparisons only; no inter-model ranking or independent final quality claim.'))
        quote=lambda p:"'"+str(p).replace("'","''")+"'"
        command='& '+quote(repo/'scripts/run_registered_combat_pool.ps1')+' -Plans @('+','.join(map(quote,plans))+') -MaxInstances 16 -Port '+str(a.port)+' -OutputRoot '+quote(out/'pool')
        save(out/'progress.json',dict(stage='native',cases=len(plans)))
        run('native-pool',[shutil.which('pwsh'),'-NoProfile','-Command',command])
        run('finish',[sys.executable,repo/'scripts/finish_combat_anchor_screen.py','--root',out])
        assert read(out/'complete.json')['state']=='complete'
        save(out/'progress.json',dict(stage='complete',complete_sha256=sha(out/'complete.json')))
    except BaseException as error:
        save(out/'progress.json',dict(stage='failed',error=str(error),exception=type(error).__name__));raise


if __name__=='__main__':main()
