"""Fixed-budget own-policy CUDA continuation; native collection uses the existing pool.

Fresh output required. Restart a stopped experiment from its last sealed fit in
a new output directory; phase receipts expose child PIDs for authoritative audit.
"""
import argparse
import concurrent.futures
import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys

from process_combat_architecture_pool import read, save, sha


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def command(args, folder, label, env):
    receipt = folder/(label+'-process.json')
    record = dict(state='starting', started_utc=now(), command=list(map(str,args)))
    save(receipt,record)
    with (folder/(label+'.log')).open('w',encoding='utf-8') as log:
        process = subprocess.Popen(record['command'],stdout=log,stderr=subprocess.STDOUT,
            env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        record.update(state='running',pid=process.pid);save(receipt,record)
        code = process.wait()
    record.update(state='complete' if code==0 else 'failed',exit_code=code,finished_utc=now())
    save(receipt,record)
    if code:raise RuntimeError('Phase failed; inspect '+str(folder/(label+'.log')))


def verify_source(folder, torch):
    seal=read(folder/'complete.json')
    assert seal['state']=='complete'
    for name,key in (('weights.json','weights_sha256'),('checkpoint.pt','checkpoint_sha256'),('report.json','report_sha256')):
        assert sha(folder/name)==seal[key], 'Unsealed source: '+name
    cp=torch.load(folder/'checkpoint.pt',map_location='cuda',weights_only=True)
    assert cp['weights_sha256']==seal['weights_sha256']
    assert cp['checkpoint_training_scope']=='combat_full_joint_policy_ppo_v1'
    assert read(folder/'weights.json')['combat_execution_contract']=='stock_weapons_fire_v1'
    return seal,cp


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('source','curriculum','config','engine','reference-plan','reference-summary','out'):
        ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--last-update',type=int,default=10)
    ap.add_argument('--first-offset',type=int,default=133)
    ap.add_argument('--evaluate-at',default='5,10')
    ap.add_argument('--port',type=int,default=53800)
    ap.add_argument('--minimum-free-gib',type=float,default=32)
    a=ap.parse_args()
    repo=Path(__file__).resolve().parents[1];a.out=a.out.resolve()
    assert a.out.is_relative_to(repo/'workspace/artifacts') and not a.out.exists()
    assert a.minimum_free_gib>=16 and 1024<=a.port<=65500-16
    powershell=shutil.which('pwsh.exe');assert powershell,'PowerShell 7 required'
    from ppo_combat import torch,training_devices
    training_devices()
    source=a.source.resolve();seal,cp=verify_source(source,torch)
    config=read(a.config);assert cp['config']==config
    start=cp['updates_completed'];assert start<a.last_update<=start+80
    evaluations=sorted({int(x) for x in a.evaluate_at.split(',') if x})
    assert all(start<x<=a.last_update for x in evaluations)
    curriculum=read(a.curriculum)
    assert len(curriculum['train_ids'])==53 and len(curriculum['validation_ids'])==38
    assert config['combat_execution_contract']=='stock_weapons_fire_v1'
    a.out.mkdir(parents=True)
    env=os.environ.copy();env['GOCACHE']=str(repo/'workspace/build/gocache')
    env['Q2COOPBOT_DIAGNOSTIC_ENGINE_ROOT']=str(a.engine.resolve())
    pins={str(p.resolve()):sha(p) for p in (a.config,a.curriculum,a.engine/'engine.json',a.reference_plan,a.reference_summary)}
    # A series never silently changes helper implementations between epochs.
    for p in (repo/'scripts').glob('*.py'):pins[str(p.resolve())]=sha(p)
    for directory in ('cmd','internal'):
        for p in (repo/directory).rglob('*.go'):pins[str(p.resolve())]=sha(p)
    for p in (repo/'scripts').glob('*.ps1'):pins[str(p.resolve())]=sha(p)
    for p in (repo/'go.mod',a.engine/'q2ded.exe',a.engine/'baseq2/game.dll'):
        pins[str(p.resolve())]=sha(p)
    reference_plan=read(a.reference_plan);reference_summary=read(a.reference_summary)
    assert reference_summary['state']=='complete'
    worlds=lambda plan:{(t['episode']['id'],i['engine_seed']):i for t in plan['tasks'] for i in t['instances']}
    registry=repo/'scripts/scenarios/combat-training/index.json'
    pins[str(registry)]=sha(registry)
    for name in read(registry)['files']:
        p=registry.parent/name;pins[str(p)]=sha(p)
    save(a.out/'protocol.json',dict(version='combat_full_joint_fixed_series_v1',state='planned',
        source=str(source),source_seal=seal,initial_updates=start,last_update=a.last_update,
        first_offset=a.first_offset,evaluate_at=evaluations,validation_offset=curriculum['validation_seed_offset'],
        train_ids=curriculum['train_ids'],validation_ids=curriculum['validation_ids'],
        config=config,inputs=pins,promotion=False,minimum_free_gib=a.minimum_free_gib,
        scope='Fixed continuation learning curve; reused validation controls, no architecture/reward tuning or automatic promotion.'))
    save(a.out/'runner.json',dict(pid=os.getpid(),started_utc=now(),executable=sys.executable))
    def unchanged():
        for path,digest in pins.items():assert sha(path)==digest,'Pinned input changed: '+path
    def progress(stage,update,**extra):
        save(a.out/'progress.json',dict(stage=stage,update=update,utc=now(),source=str(source),**extra))
        print(stage,'update',update,flush=True)
    try:
        command(['go','build','-o',a.out/'q2episode.exe','./cmd/q2episode'],a.out,'planner-build',env)
        command(['go','build','-o',a.out/'q2ppo-data.exe','./cmd/q2ppo-data'],a.out,'exporter-build',env)
        from finalize_combat_cuda_rollout import finalize
        for update in range(start+1,a.last_update+1):
            unchanged()
            assert shutil.disk_usage('F:/').free>=a.minimum_free_gib*1024**3,'Storage reserve reached'
            folder=a.out/f'update-{update:02d}';folder.mkdir()
            offset=a.first_offset+update-start-1
            seal,cp=verify_source(source,torch);assert cp['config']==config and cp['updates_completed']==update-1
            consumed=set(cp['consumed_seeds']);old_tokens=list(cp['consumed_rollouts']);del cp
            save(folder/'source.json',dict(path=str(source),seal=seal,update=update,train_offset=offset))
            def plan(ids,split,seed_offset,target,label):
                command([a.out/'q2episode.exe','--root',repo,'--episodes',','.join(ids),
                    '--mode','learned','--model',source/'weights.json','--execution-contract','stock_weapons_fire_v1',
                    '--split',split,'--count','1','--seed-offset',seed_offset,'--checked-wall-slide',
                    '--out',target/'plan.json','--artifacts',target/'captures'],target,label,env)
            progress('planning',update)
            plan(curriculum['train_ids'],'train',offset,folder,'plan')
            p=read(folder/'plan.json');seeds={s for t in p['tasks'] for s in t['seeds']}
            assert len(seeds)==53 and not seeds&consumed,'TRAIN seeds already consumed'
            progress('collecting',update)
            command([powershell,'-NoProfile','-File',repo/'scripts/run_registered_combat_pool.ps1',
                '-Plans',folder/'plan.json','-MaxInstances','16','-Port',a.port,'-OutputRoot',folder/'pool'],folder,'collect',env)
            unchanged()
            pool=read(folder/'pool/report.json')
            assert pool['state']=='complete' and pool['source_unchanged'] and len(pool['jobs'])==53
            assert all(not j['error'] for j in pool['jobs'])
            def exports(jobs,target,model):
                def one(job):
                    index=job['task_index'];native=target/f'export-{index:02d}-native';out=target/f'export-{index:02d}-cuda'
                    command([a.out/'q2ppo-data.exe','--batch',job['root'],'--model',model,
                        '--cuda-only','--out',native],target,f'export-{index:02d}',env)
                    return job,native,out
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    exported=list(executor.map(one,sorted(jobs,key=lambda j:j['task_index'])))
                cases=[]
                for job,native,out in exported:
                    assert finalize(model,native,out)['state']=='passed'
                    cases.append(dict(seed=job['seed'],capture_root=job['root'],data=str(out)))
                return cases
            progress('cuda-replay',update)
            cases=exports(pool['jobs'],folder,source/'weights.json')
            save(folder/'train-request.json',dict(version='combat_full_joint_train_request_v1',split='train',
                plan=str(folder/'plan.json'),plan_sha256=sha(folder/'plan.json'),
                pool_report=str(folder/'pool/report.json'),pool_report_sha256=sha(folder/'pool/report.json'),cases=cases))
            progress('fitting',update)
            command([sys.executable,repo/'scripts/train_combat_full_joint_ppo_cuda.py','--model',source/'weights.json',
                '--resume',source/'checkpoint.pt','--request',folder/'train-request.json','--config',a.config,
                '--out',folder/'fit'],folder,'fit',env)
            next_source=folder/'fit';next_seal,next_cp=verify_source(next_source,torch)
            assert next_cp['updates_completed']==update and next_cp['parent_checkpoint_sha256']==seal['checkpoint_sha256']
            assert next_cp['consumed_rollouts'][:len(old_tokens)]==old_tokens
            assert len(next_cp['consumed_rollouts'])==len(old_tokens)+53 and len(set(next_cp['consumed_rollouts']))==len(old_tokens)+53
            del next_cp
            source=next_source
            if update in evaluations:
                validation=folder/'validation';validation.mkdir();progress('validation-collecting',update)
                plan(curriculum['validation_ids'],'validation',curriculum['validation_seed_offset'],validation,'plan')
                assert worlds(read(validation/'plan.json'))==worlds(reference_plan),'Frozen validation conditions changed'
                command([powershell,'-NoProfile','-File',repo/'scripts/run_registered_combat_pool.ps1',
                    '-Plans',validation/'plan.json','-MaxInstances','16','-Port',a.port,'-OutputRoot',validation/'pool'],validation,'collect',env)
                vp=read(validation/'pool/report.json')
                assert vp['state']=='complete' and vp['source_unchanged'] and len(vp['jobs'])==38
                assert all(not j['error'] for j in vp['jobs'])
                progress('validation-cuda',update)
                exports(vp['jobs'],validation,source/'weights.json')
                metrics=[]
                for job in vp['jobs']:
                    rr=read(Path(job['root'])/'report.json');assert rr['capture_complete'] and rr['provenance_valid']
                    r=rr['results'][0];assert r['capture_valid'];life=r['first_life'];damage=life['damage']
                    metrics.append(dict(map=life['map'],seed=job['seed'],goal=bool(r['goal_stop']),death=bool(r['death_stop']),
                        received_hp=damage['received_health_damage'],self_hp=damage['self_health_damage'],
                        kills=sum(x['monster_kills'] for x in damage['by_mod'])))
                save(validation/'summary.json',dict(state='complete',update=update,cases=metrics,
                    totals={key:sum(r[key] for r in metrics) for key in ('goal','death','received_hp','self_hp','kills')},
                    reference_summary=reference_summary['summary'],reference_sha256=sha(a.reference_summary),promotion=False))
            unchanged()
            save(folder/'complete.json',dict(state='complete',update=update,fit=str(source),seal=next_seal))
            progress('update-complete',update)
        save(a.out/'complete.json',dict(state='complete',last_update=a.last_update,source=str(source),source_seal=next_seal,promotion=False))
        progress('complete',a.last_update)
    except BaseException as error:
        save(a.out/'failure.json',dict(state='failed',utc=now(),error=str(error),source=str(source),
            note='Do not restart from this marker alone; inspect recorded child PIDs and phase outputs. No outcomes filtered.'))
        raise


if __name__=='__main__':main()
