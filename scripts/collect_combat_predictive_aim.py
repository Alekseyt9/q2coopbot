"""Fresh registered TRAIN captures for randomized-aim prediction, never PPO replay."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('source','registry','out'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--episodes',help='Comma separated registered IDs; default all in registry')
    ap.add_argument('--count',type=int,default=1)
    ap.add_argument('--train-offset',type=int,default=0)
    ap.add_argument('--port',type=int,default=48800)
    args=ap.parse_args();repo=Path(__file__).resolve().parents[1]
    source,registry,out=[p.resolve() for p in (args.source,args.registry,args.out)]
    assert not out.exists() and 1<=args.count<=16 and args.train_offset>=0
    assert shutil.disk_usage('F:/').free>32*1024**3
    seal=read(source/'complete.json');pins={str(registry):sha(registry)}
    for name,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:
        assert sha(source/name)==seal[key];pins[str(source/name)]=seal[key]
    model=read(source/'weights.json');receipt=read(source/'report.json')
    assert model['execution_mode']=='geometry_aim_explore_v1' and not model['deterministic']
    assert receipt['device']=='cuda' and receipt['current_frame_fire_independent_of_intervention']
    descriptors={read(registry.parent/f)['id']:read(registry.parent/f) for f in read(registry)['files']}
    names=args.episodes.split(',') if args.episodes else sorted(descriptors)
    assert names and len(names)==len(set(names)) and set(names)<=descriptors.keys()
    for name in names:
        e=descriptors[name];assert e['status']=='runnable' and e['timescale']==2
    artifacts=repo/'workspace/artifacts';history=[];used=set()
    paths=set()
    for pattern in ('*plan*.json','*/*plan*.json','*/*/*plan*.json','*/corpus/*plan*.json'):
        paths.update(artifacts.glob(pattern))
    for path in sorted(paths):
        if path.parent.name.startswith('disk-cleanup'):continue
        value=read(path)
        if not isinstance(value,dict) or not isinstance(value.get('tasks'),list):continue
        history.append(dict(path=str(path),sha256=sha(path)))
        for task in value['tasks']:used.update(task.get('seeds',[]))
    offset=args.train_offset
    while True:
        assert all(offset+args.count<=descriptors[n]['splits']['train']['count'] for n in names),'TRAIN seed range exhausted'
        seeds=[descriptors[n]['splits']['train']['start']+offset+i for n in names for i in range(args.count)]
        assert len(seeds)==len(set(seeds))
        if not used.intersection(seeds):break
        offset+=args.count
    out.mkdir(parents=True)
    def run(label,command):
        with (out/(label+'.log')).open('w',encoding='utf-8') as f:
            result=subprocess.run(list(map(str,command)),cwd=repo,stdout=f,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert result.returncode==0,str(out/(label+'.log'))
    try:
        planner=out/'q2episode.exe';exporter=out/'q2action-geometry.exe'
        run('build-planner',['go','build','-o',planner,'./cmd/q2episode'])
        run('build-feature-exporter',['go','build','-o',exporter,'./cmd/q2action-geometry'])
        plan=out/'train-plan.json'
        run('plan',['%s'%planner,'--registry',registry,'--root',repo,'--episodes',','.join(names),'--split','train',
            '--mode','learned','--model',source/'weights.json','--count',args.count,'--seed-offset',offset,
            '--checked-wall-slide','--out',plan,'--artifacts',out/'captures-train'])
        frozen=read(plan);actual=[s for t in frozen['tasks'] for s in t['seeds']]
        assert sorted(actual)==sorted(seeds) and not used.intersection(actual)
        assert all(t['split']=='train' and t['modes']==['learned'] for t in frozen['tasks'])
        recipe=dict(version='combat_predictive_aim_capture_recipe_v1',purpose='shot_predictive_train_only',
            split='train',source=str(source),behavior_sampler=model['execution_mode'],registry=str(registry),episodes=names,
            count=args.count,offset=offset,seeds=seeds,max_instances=16,timescale=2,
            plan=str(plan),plan_sha256=sha(plan),inputs=pins,historical_plans=history,
            ppo_ready=False,policy_training=False,promotion=False,
            scope='Fresh randomized aim interventions, source actor parameters unchanged. Actual native MG outcomes are labels. Never standard PPO data or production-policy evaluation.')
        save(out/'recipe.json',recipe)
        save(out/'progress.json',dict(stage='native-collection',cases=len(seeds)))
        literal=lambda p:"'"+str(p).replace("'","''")+"'"
        command='& '+literal(repo/'scripts/run_registered_combat_pool.ps1')+' -Plans @('+literal(plan)+') -MaxInstances 16 -Port '+str(args.port)+' -OutputRoot '+literal(out/'pool')
        shell=shutil.which('pwsh');assert shell
        run('native-pool',[shell,'-NoProfile','-Command',command])
        pool=read(out/'pool/report.json')
        assert pool['state']=='complete' and pool['source_unchanged'] and pool['usable_captures']==len(pool['jobs'])==len(seeds)
        assert not any(j['error'] for j in pool['jobs'])
        for script,label in [('audit_registered_combat_ownership.py','ownership'),('audit_registered_execution_actions.py','execution')]:
            run(label,[sys.executable,repo/'scripts'/script,'--pool',out/'pool','--plan-index',0,'--out',out/(label+'.json')])
        save(out/'native-complete.json',dict(state='complete',recipe_sha256=sha(out/'recipe.json'),pool_sha256=sha(out/'pool/report.json'),
            ownership_sha256=sha(out/'ownership.json'),execution_sha256=sha(out/'execution.json'),native_cases=len(seeds),ppo_ready=False))
        save(out/'progress.json',dict(stage='building-predictive-dataset',native_cases=len(seeds)))
        run('dataset',[sys.executable,repo/'scripts/build_combat_predictive_aim_dataset_cuda.py',
            '--capture',out,'--exporter',exporter,'--out',out/'dataset'])
        complete=read(out/'dataset/complete.json');assert complete['state']=='complete' and not complete['ppo_ready']
        for path,digest in pins.items():assert sha(Path(path))==digest,path
        save(out/'complete.json',dict(state='complete',version='combat_predictive_aim_capture_complete_v1',native_cases=len(seeds),
            recipe_sha256=sha(out/'recipe.json'),native_complete_sha256=sha(out/'native-complete.json'),
            dataset_complete_sha256=sha(out/'dataset/complete.json'),ppo_ready=False,policy_training=False,promotion=False))
        save(out/'progress.json',dict(stage='complete',native_cases=len(seeds)))
        print(out/'complete.json')
    except BaseException as error:
        save(out/'progress.json',dict(stage='failed',error=str(error)))
        raise


if __name__=='__main__':main()
