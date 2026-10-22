"""Fresh native rules demonstrations for joint imitation; no neural execution.

Uses the existing frozen episode planner and 16-slot native pool. Records raw
pre-command observations, explicit teacher aim entity and applied commands.
Feature/history adaptation and GPU student training are separate later phases.
"""
import argparse
import collections
import json
import os
import pathlib
import shutil
import subprocess
import sys
from process_combat_architecture_pool import read,save,sha
from native_combat_release import release_boundary


def main():
    ap=argparse.ArgumentParser()
    for name in ('reference','registry','out'):ap.add_argument('--'+name,type=pathlib.Path,required=True)
    ap.add_argument('--episodes',required=True);ap.add_argument('--train-offset',type=int,default=416)
    ap.add_argument('--development-offset',type=int,default=224);ap.add_argument('--port',type=int,default=44700)
    ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--run-prepared',action='store_true')
    args=ap.parse_args();assert not (args.prepare_only and args.run_prepared)
    repo=pathlib.Path(__file__).resolve().parents[1];root=args.out.resolve();reference=args.reference.resolve();registry=args.registry.resolve()
    assert root.is_relative_to(repo/'workspace/artifacts') and 1024<=args.port<=65520
    assert shutil.disk_usage(root.anchor).free>=32*1024**3
    os.environ['GOCACHE']=str(repo/'workspace/build/go-cache');os.environ['GOTOOLCHAIN']='auto'
    def run(label,command):
        with (root/(label+'.log')).open('w',encoding='utf-8') as output:
            result=subprocess.run(list(map(str,command)),cwd=repo,stdout=output,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert result.returncode==0,'Inspect '+str(root/(label+'.log'))
    if not args.run_prepared:
        assert not root.exists();root.mkdir(parents=True)
        source_seal=read(reference/'complete.json');pins={str(reference/'complete.json'):sha(reference/'complete.json'),str(registry):sha(registry)}
        for file,key in [('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')]:
            assert sha(reference/file)==source_seal[key];pins[str(reference/file)]=source_seal[key]
        model=read(reference/'weights.json');assert model['feature_version']=='combat_features_v10'
        names=args.episodes.split(',');assert len(names)==len(set(names)) and names
        descriptors={read(registry.parent/f)['id']:(registry.parent/f,read(registry.parent/f)) for f in read(registry)['files']}
        for name in names:
            path,e=descriptors[name];assert e['status']=='runnable' and e['recipe']['runner']=='combat-baseline' and e['timescale']==2
            pins[str(path)]=sha(path);reward=(repo/e['recipe']['reward_config']).resolve();pins[str(reward)]=sha(reward)
        history=set();historical=[];artifacts=repo/'workspace/artifacts'
        for path in sorted(set(artifacts.glob('*plan*.json'))|set(artifacts.glob('*/*plan*.json'))|set(artifacts.glob('*/round-*/*plan*.json'))):
            value=read(path)
            if not isinstance(value,dict) or not isinstance(value.get('tasks'),list):continue
            historical.append(dict(path=str(path.resolve()),sha256=sha(path)))
            for task in value['tasks']:history.update(task.get('seeds',[]))
        planner=root/'q2episode.exe';run('build-planner',['go','build','-o',planner,'./cmd/q2episode'])
        assignments=[]
        for label,split,count,offset in [('train','train',4,args.train_offset),('development','validation',2,args.development_offset)]:
            while True:
                assert all(offset+count<=descriptors[n][1]['splits'][split]['count'] for n in names)
                seeds={descriptors[n][1]['splits'][split]['start']+offset+k for n in names for k in range(count)}
                assert len(seeds)==len(names)*count
                if not seeds.intersection(history):break
                offset+=count
            plan=root/(label+'-plan.json')
            run('plan-'+label,[planner,'--registry',registry,'--root',repo,'--episodes',','.join(names),
                '--split',split,'--mode','rules','--count',count,'--seed-offset',offset,'--out',plan,'--artifacts',root/('captures-'+label)])
            frozen=read(plan);actual=[s for t in frozen['tasks'] for s in t['seeds']]
            assert set(actual)==seeds and len(actual)==len(seeds) and all(t['modes']==['rules'] and t['split']==split for t in frozen['tasks'])
            history.update(seeds)
            assignments.append(dict(label=label,split=split,plan=str(plan),plan_sha256=sha(plan),seeds=sorted(seeds),count=count,offset=offset))
        save(root/'protocol.json',dict(state='prepared',version='registered_rules_teacher_corpus_v1',student_reference=str(reference),
            reference_weights_sha256=source_seal['weights_sha256'],feature_version=model['feature_version'],registry=str(registry),
            episodes=names,assignments=assignments,inputs=[dict(path=p,sha256=h) for p,h in pins.items()],historical_plans=historical,
            port=args.port,max_instances=16,timescale=2,mode='rules',neural_execution=False,test_deferred=True,
            training_performed=False,promotion=False,script_sha256=sha(__file__),
            scope='Rules teacher demonstrations on fresh registered train/development seeds. Student reference is provenance only; its network does not execute. Explicit teacher targets and all applied controls preserved for later joint imitation. Rules include tactical guards and are not an optimal or human-reviewed teacher. No validation/test evaluation logs reused as training.'))
        save(root/'progress.json',dict(stage='prepared',native_cases=sum(len(a['seeds']) for a in assignments)))
    else:assert (root/'protocol.json').exists() and not (root/'pool').exists(), 'Prepared fresh native phase required'
    protocol=read(root/'protocol.json');assert protocol['version']=='registered_rules_teacher_corpus_v1' and protocol['mode']=='rules'
    assert protocol['script_sha256']==sha(__file__)
    for item in protocol['inputs']:assert sha(item['path'])==item['sha256']
    plans=[]
    for item in protocol['assignments']:
        assert sha(item['plan'])==item['plan_sha256'];plans.append(pathlib.Path(item['plan']))
    if args.prepare_only:
        print(dict(state='prepared',train_cases=len(protocol['assignments'][0]['seeds']),development_cases=len(protocol['assignments'][1]['seeds'])));return
    try:
        total=sum(len(a['seeds']) for a in protocol['assignments']);save(root/'progress.json',dict(stage='native-teacher-collection',native_cases=total))
        literal=lambda p:"'"+str(p).replace("'","''")+"'"
        command='& '+literal(repo/'scripts/run_registered_combat_pool.ps1')+' -Plans @('+','.join(map(literal,plans))+') -MaxInstances 16 -Port '+str(protocol['port'])+' -OutputRoot '+literal(root/'pool')
        run('native-pool',[shutil.which('pwsh'),'-NoProfile','-Command',command])
        pool=read(root/'pool/report.json');assert pool['state']=='complete' and pool['source_unchanged']
        assert pool['usable_captures']==len(pool['jobs'])==total and not any(j['error'] for j in pool['jobs'])
        cases=[];source_pins=[]
        for job in pool['jobs']:
            capture_path=pathlib.Path(job['root'])/'report.json';manifest_path=capture_path.with_name('manifest.json')
            capture,manifest=read(capture_path),read(manifest_path)
            assert capture['capture_complete'] and capture['provenance_valid'] and len(capture['results'])==1 and job['mode']=='rules'
            result=capture['results'][0];assert result['seed']==job['seed'] and result['capture_valid'] and result['dispatch_valid'] and result['seed_confirmed']
            worker=pathlib.Path(result['root']);log=worker/'server.log';trace=worker/'bot.jsonl'
            with log.open(encoding='utf-8-sig') as stream:generation,release,game_release=release_boundary(stream,manifest,job['seed'])
            counts=collections.Counter()
            with trace.open(encoding='utf-8-sig') as stream:
                for row in map(json.loads,stream):
                    p=row.get('combat_policy') or {};o=p.get('observation') or {};identity=o.get('identity') or {}
                    if identity.get('spawncount')!=generation or identity.get('frame',-1)<release or identity.get('life')!=1 or o.get('health',0)<=0:continue
                    selection=p.get('selection') or {};assert selection.get('owner')=='rules' and selection.get('mode')=='rules'
                    a=p['applied'];counts['released_frames']+=1
                    target=p.get('teacher_aim_entity',0)
                    if p.get('teacher_aim_source')=='enemy' and target:
                        counts['explicit_enemy_target_frames']+=1
                        counts['explicit_visible_target_frames']+=any(e['id']==target and e.get('clear_shot') for e in o['enemies'])
                    counts['movement_frames']+=bool(a['forward'] or a['side'])
                    counts['weapon_request_frames']+=bool(a.get('weapon'))
                    counts['vertical_nonrelease_frames']+=a.get('vertical','release')!='release'
            assert counts['released_frames']>0
            cases.append(dict(job=job['job'],seed=job['seed'],split=protocol['assignments'][job['plan_index']]['label'],
                root=str(capture_path.parent),goal=bool(result['goal_stop']),counts=dict(counts),network_release_frame=release,native_release_game_frame=game_release))
            source_pins.extend(dict(path=str(p),sha256=sha(p)) for p in [capture_path,manifest_path,trace,log])
        for item in protocol['inputs']:assert sha(item['path'])==item['sha256']
        save(root/'complete.json',dict(state='complete',version='registered_rules_teacher_corpus_v1',protocol_sha256=sha(root/'protocol.json'),
            pool_sha256=sha(root/'pool/report.json'),native_cases=total,cases=cases,inputs=source_pins,training_performed=False,
            neural_execution=False,promotion=False,feature_export_complete=False,
            scope='Closed rules demonstrations with canonical release and explicit target/control evidence. No student fitting or NN execution. Derived feature/history adapters and supervised masks still required before GPU BC.'))
        save(root/'progress.json',dict(stage='complete',native_cases=total,training_performed=False))
        print(dict(state='complete',native_cases=total,teacher_counts=dict(sum((collections.Counter(c['counts']) for c in cases),collections.Counter()))))
    except BaseException as error:
        save(root/'progress.json',dict(stage='failed',error=str(error),training_performed=False));raise


if __name__=='__main__':main()
