"""Fresh native quality screen of fixed learned-option prior initializations."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from process_combat_architecture_pool import read,save,sha
from combat_plan_history import historical_plans
from compare_registered_combat_pools import load
from combat_joint_aim_policy import base_model
from combat_teacher_ownership import rules_ownership
from evaluate_registered_combat_models import pool_command


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('pilot','parent','initial','half','high','out'):ap.add_argument('--'+k,type=Path,required=True)
    ap.add_argument('--seed-offset',type=int,default=384)
    ap.add_argument('--prepared-from',type=Path,help='Reuse eight verified worlds from a preparation failure before any games')
    ap.add_argument('--port',type=int,default=55600);a=ap.parse_args()
    repo=Path(__file__).resolve().parents[1];root=a.out.resolve();assert not root.exists() and shutil.disk_usage('F:/').free>=32*1024**3
    pilot=read(a.pilot/'complete.json');assert pilot['state']=='complete' and pilot['training_performed'] and not pilot['quality_evaluated']
    train=read(a.pilot/'train-plan.json');ids=sorted({t['episode']['id'] for t in train['tasks']});assert 1<=len(ids)<=53
    variants=dict(parent=a.parent.resolve(),p02=a.initial.resolve(),p05=a.half.resolve(),p09=a.high.resolve(),rules=None)
    model_pins={}
    for name,path in variants.items():
        if path is None:continue
        seal=read(path/'complete.json')
        for file,key in (('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')):
            assert sha(path/file)==seal[key];model_pins[str(path/file)]=seal[key]
    models={name:read(path/'weights.json') for name,path in variants.items() if path is not None}
    parent_core=base_model(models['parent'],allow_shared=True)
    for name in ('p02','p05','p09'):
        assert base_model(models[name],allow_shared=True,allow_wide=True)==parent_core
        assert models[name]['wide_aim']['layers'][:2]==models['p02']['wide_aim']['layers'][:2]
        assert all(v==0 for row in models[name]['wide_aim']['layers'][-1]['weight'] for v in row)
        assert not read(variants[name]/'report.json')['training_performed']
    assert [read(variants[n]/'report.json')['initial_probability'] for n in ('p02','p05','p09')]==[.2,.5,.9]
    root.mkdir(parents=True);(root/'frozen-scripts').mkdir()
    os.environ['GOCACHE']=str(repo/'workspace/build/gocache')
    os.environ['Q2COOPBOT_DIAGNOSTIC_ENGINE_ROOT']=str(repo/'workspace/artifacts/combat-objective-engine-v2-20261025')
    def run(label,cmd):
        save(root/'progress.json',dict(stage=label))
        with (root/(label+'.log')).open('w',encoding='utf-8') as log:
            p=subprocess.run(list(map(str,cmd)),cwd=repo,stdout=log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert p.returncode==0,str(root/(label+'.log'))
    try:
        planner=root/'q2episode.exe';exporter=root/'q2ppo-data.exe'
        run('build-planner',['go','build','-o',planner,'./cmd/q2episode'])
        run('build-exporter',['go','build','-o',exporter,'./cmd/q2ppo-data'])
        inventory_planner=root/'q2inventory-candidates.exe'
        run('build-inventory-planner',['go','build','-o',inventory_planner,'./cmd/q2inventory-candidates'])
        if a.prepared_from:
            previous=a.prepared_from.resolve()
            assert not (previous/'pool').exists(), 'Only pre-collection preparation may be reused'
            failure=read(previous/'progress.json')
            assert failure['stage']=='failed' and failure['error']==str(previous/'verify-rules.log')
            assert 'checked execution contract requires exclusive learned plan' in (previous/'verify-rules.log').read_text(encoding='utf-8-sig')
            candidate=read(previous/'p05-plan.json')
            assert candidate['model_sha256']==sha(variants['p05']/'weights.json')
            tasks=candidate['tasks'];names=[t['episode']['map'] for t in tasks]
            assert len(tasks)==len(set(names))==8 and all(t['split']=='validation' for t in tasks)
            assert all(len(t['seeds'])==len(t['instances'])==1 for t in tasks)
            save(root/'prepared-reuse.json',dict(source=str(previous),plan_sha256=sha(previous/'p05-plan.json'),
                reason='Correct rules-only execution metadata; preserve all eight admitted worlds before any native outcomes'))
            for file in ('history.json','geometry-preflight.json','inventory-candidates.json'):
                shutil.copyfile(previous/file,root/file)
        else:
            run('inventory-candidates',[inventory_planner,'--registry',train['registry_path'],'--episodes',','.join(ids),
                '--split','validation','--count',32,'--seed-offset',a.seed_offset,'--initial-weapon','Machinegun','--ammo','Bullets','--minimum-ammo',5,
                '--out',root/'inventory-candidates.json'])
            inventory_candidates=read(root/'inventory-candidates.json');used=set();history=[]
            save(root/'progress.json',dict(stage='historical-seed-admission'))
            for path,old in historical_plans(repo/'workspace/artifacts',exclude_roots=(root,)):
                for t in old['tasks']:used.update(t.get('seeds',[]))
                history.append(dict(path=str(path),sha256=sha(path)))
            save(root/'history.json',dict(plans=history,used_seeds=len(used)))
            maps={}
            for c in inventory_candidates['candidates']:
                if c['seed'] not in used:maps.setdefault(c['map'],[]).append(c)
            ordered=sorted(maps,key=lambda name:hashlib.sha256(('wide-prior-screen-20261026:'+name).encode()).hexdigest())
            tasks=[];names=[];preflight=[];candidate=None
            for name in ordered:
                for c in sorted(maps[name],key=lambda c:(c['episode_id'],c['seed'])):
                    label='preflight-'+name+'-'+str(c['seed']);path=root/(label+'-plan.json')
                    cmd=[planner,'--registry',train['registry_path'],'--root',repo,'--episodes',c['episode_id'],
                        '--split','validation','--mode','learned','--model',variants['p05']/'weights.json','--count',1,'--seed-offset',c['offset'],
                        '--checked-wall-slide','--execution-contract','stock_weapons_fire_v1','--out',path,'--artifacts',root/('unused-'+label)]
                    save(root/'progress.json',dict(stage=label))
                    log=root/(label+'.log')
                    with log.open('w',encoding='utf-8') as stream:p=subprocess.run(list(map(str,cmd)),cwd=repo,stdout=stream,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    preflight.append(dict(map=name,seed=c['seed'],exit_code=p.returncode,log=str(log),log_sha256=sha(log)))
                    save(root/'geometry-preflight.json',dict(cases=preflight,native_games=0))
                    if p.returncode:
                        assert 'generator exhausted attempts' in log.read_text(encoding='utf-8-sig'),'Unexpected metadata failure: '+str(log)
                        continue
                    value=read(path);one=value['tasks'][0];assert one['instances'][0]['inventory']==c['inventory']
                    candidate=value if candidate is None else candidate
                    tasks.append(one);names.append(name);break
                if len(tasks)==8:break
            assert len(tasks)==8 and candidate is not None,'Insufficient fresh physically valid validation MG coverage'
        plans=[];assignments=[];pins=dict(model_pins)
        for name,folder in variants.items():
            plan=copy.deepcopy(candidate);plan['tasks']=copy.deepcopy(tasks);plan['output_root']=str(root/('captures-'+name))
            if folder is None:
                plan.pop('model_path',None);plan.pop('model_sha256',None)
                plan.pop('combat_execution_contract',None);plan.pop('checked_wall_slide',None)
                for task in plan['tasks']:task['modes']=['rules']
            else:plan['model_path']=str(folder/'weights.json');plan['model_sha256']=sha(folder/'weights.json')
            path=root/(name+'-plan.json');save(path,plan)
            run('verify-'+name,[planner,'--verify-plan',path,'--root',repo]);plans.append(path)
            pins[str(path)]=sha(path);assignments.append(dict(label=name,plan_index=len(plans)-1,model=plan.get('model_path')))
        for name in ('run_combat_wide_prior_screen.py','combat_teacher_ownership.py','combat_wide_aim.py','combat_joint_aim_policy.py','finalize_combat_cuda_rollout.py','ppo_combat.py'):
            path=repo/'scripts'/name;pins[str(path)]=sha(path);shutil.copyfile(path,root/'frozen-scripts'/name)
        save(root/'protocol.json',dict(version='combat_wide_prior_screen_v1',split='validation',maps=names,
            seeds=[t['seeds'][0] for t in tasks],assignments=assignments,inputs=pins,history_sha256=sha(root/'history.json'),
            selection='Fresh validation split, inventory-only initialMG/atleast5bullets enumeration, then unmodified static geometry preflight. First physically valid episode/seed per map; fixed SHA map ordering, cap8. Every rejected geometry seed logged before games; no outcome filtering or relaxed overlap checks.',
            primary='Native goal without first death or rules-owned combat through last kill',
            advance='For each p05/p09: more autonomous goals than both parent and p02, with no more deaths than either. Rules reported separately. Fixed development priors, not trained improvement; larger fresh screen required, no promotion.',
            native_cases=40,cuda_cases=32,max_instances=16,timescale=2,training_performed=False,promotion=False,
            scope='Fresh seeds and disjoint validation placements within TRAIN episode families; not unseen-map or independent final confirmation.'))
        run('collect',[shutil.which('pwsh'),'-NoProfile','-Command',pool_command(repo,plans,16,a.port,root/'pool')])
        pool=read(root/'pool/report.json');assert pool['state']=='complete' and pool['source_unchanged'] and pool['usable_captures']==len(pool['jobs'])==40
        cuda_tasks=[]
        for j in sorted(pool['jobs'],key=lambda j:(j['plan_index'],j['task_index'])):
            assert not j['error']
            if j['mode']=='rules':continue
            i=j['plan_index']*8+j['task_index'];model=assignments[j['plan_index']]['model'];native=root/f'export-{i:02d}-native';cuda=root/f'export-{i:02d}-cuda'
            run('export-'+str(i),[exporter,'--batch',j['root'],'--model',model,'--cuda-only','--out',native])
            cuda_tasks.append(dict(model=model,model_sha256=sha(model),data=str(native),out=str(cuda),native_report_sha256=sha(native/'report.json')))
        save(root/'cuda-request.json',dict(version='combat_cuda_finalize_batch_v1',compact_finalized_streams=True,tasks=cuda_tasks))
        run('cuda',[sys.executable,repo/'scripts/finalize_combat_cuda_batch.py','--request',root/'cuda-request.json','--receipt-root',root/'cuda-batch'])
        assert read(root/'cuda-batch/report.json')['state']=='complete'
        # Existing comparator checks exact fixtures, engine binaries, source and
        # timing/environment for every paired world across all four arms.
        groups={v['label']:load(root/'pool',v['plan_index']) for v in assignments}
        keys=groups['parent'].keys();assert all(v.keys()==keys and len(v)==8 for v in groups.values())
        for name in ('p02','p05','p09','rules'):
            run('compare-'+name,[sys.executable,repo/'scripts/compare_registered_combat_pools.py','--before',root/'pool','--after',root/'pool',
                '--before-plan-index',0,'--after-plan-index',next(v['plan_index'] for v in assignments if v['label']==name),'--out',root/('paired-'+name)])
        results={v['label']:[] for v in assignments}
        commands={}
        for j in pool['jobs']:
            label=assignments[j['plan_index']]['label'];r=read(Path(j['root'])/'report.json')['results'][0]
            steps=Path(r['root'])/'dataset/steps.jsonl';rows=[json.loads(l) for l in steps.open(encoding='utf-8-sig')]
            stop=r['first_life']['end_frame'];goal=r.get('goal_stop');cutoff=goal['kill_frame'] if goal else stop
            active=[s for s in rows if s['observation']['identity']['life']==1 and s['observation']['health']>0 and s['observation']['identity']['frame']<=cutoff]
            ownership=rules_ownership(active,goal,stop);rules_before=ownership['rules_combat_rows']
            m=groups[label][(r['generated_fixture']['episode_id'],j['seed'])]
            record={k:m[k] for k in ('goal','death','kills','dealt','received')}
            record.update(seed=j['seed'],map=r['first_life']['map'],rules_combat_rows=rules_before,passive_goal_stop_frames=ownership['passive_goal_stop_frames'],
                autonomous_goal=bool(m['goal'] and not m['death'] and (j['mode']=='rules' or rules_before==0)),
                wide_events=sum(bool((s.get('sample') or {}).get('wide_aim')) for s in active),
                wide_choices=sum((s.get('sample') or {}).get('wide_aim',{}).get('choice',0) for s in active))
            results[label].append(record)
            commands[(label,j['seed'])]=[(s['observation']['identity']['frame'],s['sent_command']) for s in active]
        totals={label:{key:sum(r[key] for r in rows) for key in ('goal','autonomous_goal','death','kills','dealt','received','wide_events','wide_choices')} for label,rows in results.items()}
        controls=[totals[n] for n in ('parent','p02')]
        advances={name:totals[name]['autonomous_goal']>max(c['autonomous_goal'] for c in controls) and totals[name]['death']<=min(c['death'] for c in controls) for name in ('p05','p09')}
        action_differences={name:sum(commands[(name,seed)]!=commands[('p02',seed)] for seed in (t['seeds'][0] for t in tasks)) for name in ('p05','p09')}
        advance=any(advances.values());verdict='advance_to_larger_fresh_screen' if advance else 'not_advanced'
        for path,h in pins.items():assert sha(path)==h
        save(root/'results.json',dict(state='complete',totals=totals,cases=results,executed_difference_cases=action_differences,candidate_advances=advances,
            verdict=verdict,advance=advance,training_performed=False,promotion=False,quality_confirmation=False,initialization_only=True,
            scope='Eight fresh paired MG-start validation conditions. Rules ownership explicitly qualifies learned goals. This is a small development screen, not campaign or held-out-map acceptance.'))
        save(root/'complete.json',dict(state='complete',native_cases=40,cuda_cases=32,worlds=8,results_sha256=sha(root/'results.json'),
            protocol_sha256=sha(root/'protocol.json'),pool_sha256=sha(root/'pool/report.json'),cuda_sha256=sha(root/'cuda-batch/report.json'),
            training_performed=False,promotion=False,quality_confirmation=False))
        save(root/'progress.json',dict(stage='complete',complete_sha256=sha(root/'complete.json')))
        print(dict(totals=totals,verdict=verdict))
    except BaseException as error:
        save(root/'progress.json',dict(stage='failed',error=str(error),exception=type(error).__name__));raise


if __name__=='__main__':main()
