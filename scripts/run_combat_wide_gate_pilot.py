"""Fresh native TRAIN batch of the isolated learned wide-aim choice."""
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


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('source','reference-plan','compatibility','out'):ap.add_argument('--'+k,type=Path,required=True)
    ap.add_argument('--collect-only',action='store_true',help='Close verified source collection without actor/critic fitting')
    ap.add_argument('--config',type=Path,help='Exact existing checkpoint-compatible training config')
    ap.add_argument('--seed-offset',type=int,default=192)
    ap.add_argument('--cases',type=int,choices=(16,32),default=32)
    ap.add_argument('--port',type=int,default=54100);a=ap.parse_args()
    repo=Path(__file__).resolve().parents[1];out=a.out.resolve();source=a.source.resolve()
    assert not out.exists() and shutil.disk_usage('F:/').free>=32*1024**3
    compatibility=read(a.compatibility/'complete.json');assert compatibility['state']=='complete' and compatibility['native_cases']==compatibility['cuda_cases']==9
    seal=read(source/'complete.json');assert sha(source/'weights.json')==seal['weights_sha256']
    config_path=a.config.resolve() if a.config else source/'config.json'
    config=read(config_path);assert config['actor_scope']=='wide_only' and config['actor_steps']==config['value_steps']==32
    reference=read(a.reference_plan);ids=[t['episode']['id'] for t in reference['tasks']]
    assert len(ids)==53
    assert a.seed_offset>=0 and all(a.seed_offset+128<=t['episode']['splits']['train']['count'] for t in reference['tasks']), 'Inventory cohort exceeds a registered TRAIN split'
    out.mkdir(parents=True);(out/'frozen-scripts').mkdir()
    os.environ['GOCACHE']=str(repo/'workspace/build/gocache')
    os.environ['Q2COOPBOT_DIAGNOSTIC_ENGINE_ROOT']=str(repo/'workspace/artifacts/combat-objective-engine-v2-20261025')
    def run(label,cmd):
        save(out/'progress.json',dict(stage=label))
        with (out/(label+'.log')).open('w',encoding='utf-8') as log:
            p=subprocess.run(list(map(str,cmd)),cwd=repo,stdout=log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert p.returncode==0,str(out/(label+'.log'))
    try:
        planner=out/'q2episode.exe';exporter=out/'q2ppo-data.exe'
        run('build-planner',['go','build','-o',planner,'./cmd/q2episode'])
        run('build-exporter',['go','build','-o',exporter,'./cmd/q2ppo-data'])
        inventory_planner=out/'q2inventory-candidates.exe'
        run('build-inventory-planner',['go','build','-o',inventory_planner,'./cmd/q2inventory-candidates'])
        run('inventory-candidates',[inventory_planner,'--registry',reference['registry_path'],'--episodes',','.join(ids),
            '--split','train','--count',128,'--seed-offset',a.seed_offset,'--initial-weapon','Machinegun','--ammo','Bullets','--minimum-ammo',5,
            '--out',out/'inventory-candidates.json'])
        assert len({c['map'] for c in read(out/'inventory-candidates.json')['candidates']})>=a.cases,'Insufficient inventory-only map coverage'
        save(out/'progress.json',dict(stage='historical-seed-admission'))
        used=set();history=[]
        for path,old in historical_plans(repo/'workspace/artifacts',exclude_roots=(out,)):
            for t in old['tasks']:used.update(t.get('seeds',[]))
            history.append(dict(path=str(path),sha256=sha(path)))
        save(out/'history.json',dict(plans=history,used_seeds=len(used)))
        maps={}
        for c in read(out/'inventory-candidates.json')['candidates']:
            if c['seed'] not in used:maps.setdefault(c['map'],[]).append(c)
        ordered=sorted(maps,key=lambda name:hashlib.sha256(('wide-gate-pilot-20261026:'+name).encode()).hexdigest())
        selected={};preflight=[];template=None
        for name in ordered:
            chosen=[]
            for c in sorted(maps[name],key=lambda c:(c['episode_id'],c['seed'])):
                label='preflight-'+name+'-'+str(c['seed']);path=out/(label+'-plan.json');log=out/(label+'.log')
                cmd=[planner,'--registry',reference['registry_path'],'--root',repo,'--episodes',c['episode_id'],
                    '--split','train','--mode','learned','--model',source/'weights.json','--count',1,'--seed-offset',c['offset'],
                    '--checked-wall-slide','--execution-contract','stock_weapons_fire_v1','--out',path,'--artifacts',out/('unused-'+label)]
                save(out/'progress.json',dict(stage=label))
                with log.open('w',encoding='utf-8') as stream:
                    process=subprocess.run(list(map(str,cmd)),cwd=repo,stdout=stream,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                preflight.append(dict(map=name,seed=c['seed'],exit_code=process.returncode,log=str(log),log_sha256=sha(log)))
                save(out/'geometry-preflight.json',dict(cases=preflight,native_games=0))
                if process.returncode:
                    assert 'generator exhausted attempts' in log.read_text(encoding='utf-8-sig'),'Unexpected preparation failure: '+str(log)
                    continue
                value=read(path);one=value['tasks'][0]
                assert one['instances'][0]['inventory']==c['inventory']
                template=value if template is None else template
                chosen.append(one)
                if len(chosen)==1:break
            if len(chosen)==1:selected[name]=chosen
            if len(selected)==a.cases:break
        assert len(selected)==a.cases and template is not None,'Insufficient fresh physically valid MG-start coverage'
        names=list(selected)
        plan=copy.deepcopy(template);plan['tasks']=[t for name in names for t in selected[name]];plan['output_root']=str(out/'captures')
        assert len(plan['tasks'])==len({t['seeds'][0] for t in plan['tasks']})==a.cases
        save(out/'train-plan.json',plan)
        run('verify-plan',[planner,'--verify-plan',out/'train-plan.json','--root',repo])
        pins={str(p.resolve()):sha(p) for p in (source/'weights.json',source/'checkpoint.pt',config_path,out/'train-plan.json',out/'history.json')}
        helpers=('run_combat_wide_gate_pilot.py','train_combat_full_joint_ppo_cuda.py','combat_full_joint_ppo.py',
            'combat_wide_aim.py','combat_joint_aim_policy.py','finalize_combat_cuda_rollout.py','ppo_combat.py')
        for name in helpers:
            path=repo/'scripts'/name;pins[str(path)]=sha(path);shutil.copyfile(path,out/'frozen-scripts'/name)
        save(out/'protocol.json',dict(version='combat_wide_gate_train_pilot_v2',source=str(source),inputs=pins,
            maps=names,cases=a.cases,seed_offset=a.seed_offset,train_seeds=[t['seeds'][0] for t in plan['tasks']],selection='Inventory-only initialMG/atleast5bullets enumeration, historical seed exclusion, then normal static geometry admission. One fresh case per distinct SHA-ordered map; all native class and random remaining inventory constraints unchanged. No outcome filtering.',
            split='train',actor_scope='wide_only',actor_steps=0 if a.collect_only else 32,value_steps=0 if a.collect_only else 32,collection_only=a.collect_only,config=config,max_instances=16,timescale=2,
            initial_weights_native_compatibility=str(a.compatibility.resolve()),training_performed=False,promotion=False,
            scope='Isolated gate-learning batch; all planned cases required. Fresh seeds against physical plan history. No diagnostic-source replay or synthetic labels; no quality claim.'))
        run('collect',[shutil.which('pwsh'),'-NoProfile','-File',repo/'scripts/run_registered_combat_pool.ps1','-Plans',out/'train-plan.json',
            '-MaxInstances',16,'-Port',a.port,'-OutputRoot',out/'pool'])
        pool=read(out/'pool/report.json');assert pool['state']=='complete' and pool['source_unchanged'] and pool['usable_captures']==len(pool['jobs'])==a.cases
        tasks=[];cases=[]
        for j in sorted(pool['jobs'],key=lambda j:j['task_index']):
            assert not j['error'];i=j['task_index'];native=out/f'export-{i:02d}-native';cuda=out/f'export-{i:02d}-cuda'
            run('export-'+str(i),[exporter,'--batch',j['root'],'--model',source/'weights.json','--cuda-only','--out',native])
            tasks.append(dict(model=str(source/'weights.json'),model_sha256=seal['weights_sha256'],data=str(native),out=str(cuda),native_report_sha256=sha(native/'report.json')))
            cases.append(dict(seed=j['seed'],capture_root=j['root'],data=str(cuda)))
        save(out/'cuda-request.json',dict(version='combat_cuda_finalize_batch_v1',compact_finalized_streams=True,tasks=tasks))
        run('cuda',[sys.executable,repo/'scripts/finalize_combat_cuda_batch.py','--request',out/'cuda-request.json','--receipt-root',out/'cuda-batch'])
        batch=read(out/'cuda-batch/report.json');assert batch['state']=='complete' and len(batch['tasks'])==a.cases
        save(out/'train-request.json',dict(version='combat_full_joint_train_request_v1',split='train',plan=str(out/'train-plan.json'),plan_sha256=sha(out/'train-plan.json'),
            pool_report=str(out/'pool/report.json'),pool_report_sha256=sha(out/'pool/report.json'),cases=cases))
        for p,h in pins.items():assert sha(p)==h
        if a.collect_only:
            save(out/'complete.json',dict(state='complete',device='cuda',collection_only=True,train_cases=a.cases,train_rows=sum(read(Path(c['data'])/'report.json')['rows'] for c in cases),source_model_sha256=seal['weights_sha256'],native_cases=a.cases,cuda_cases=a.cases,training_performed=False,quality_evaluated=False,promotion=False))
            save(out/'progress.json',dict(stage='complete',complete_sha256=sha(out/'complete.json')))
            print(out/'complete.json');return
        run('fit',[sys.executable,repo/'scripts/train_combat_full_joint_ppo_cuda.py','--model',source/'weights.json','--resume',source/'checkpoint.pt',
            '--request',out/'train-request.json','--config',config_path,'--out',out/'fit'])
        fitted=read(out/'fit/report.json');assert fitted['state']=='passed' and fitted['actor_scope']=='wide_only'
        assert fitted['legacy_policy_parameters_unchanged'] and fitted['legacy_policy_moments_unchanged'] and len(fitted['train_seeds'])==a.cases
        save(out/'complete.json',dict(state='complete',device='cuda',train_cases=a.cases,train_rows=fitted['rows'],
            weights_sha256=sha(out/'fit/weights.json'),fit_seal_sha256=sha(out/'fit/complete.json'),
            actor_steps=fitted['actor_steps'],wide_event_rows=fitted['wide_event_rows'],wide_selected_rows=fitted['wide_selected_rows'],
            training_performed=True,quality_evaluated=False,promotion=False))
        save(out/'progress.json',dict(stage='complete',complete_sha256=sha(out/'complete.json')))
        print(out/'complete.json')
    except BaseException as error:
        save(out/'progress.json',dict(stage='failed',error=str(error),exception=type(error).__name__));raise


if __name__=='__main__':main()
