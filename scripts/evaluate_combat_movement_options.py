"""Fixed four-arm registered validation: source, untrained options, fitted options, rules."""
import argparse
import concurrent.futures
import os
from pathlib import Path
import shutil
import sys
from process_combat_architecture_pool import read,save,sha
from run_combat_full_joint_series import command,now


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','initial','fitted','curriculum','engine','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seed-offset',type=int,default=401)
    p.add_argument('--port',type=int,default=55400)
    a=p.parse_args();repo=Path(__file__).resolve().parents[1];a.out=a.out.resolve()
    assert not a.out.exists() and a.out.is_relative_to(repo/'workspace/artifacts')
    assert shutil.disk_usage('F:/').free>40*1024**3
    ids=read(a.curriculum)['validation_ids'];assert len(ids)==len(set(ids))==38
    models={'source':a.source.resolve(),'initial':a.initial.resolve(),'fitted':a.fitted.resolve(),'rules':a.source.resolve()}
    from combat_movement_option_ppo import frozen_source
    assert frozen_source(read(models['initial']))==frozen_source(read(models['fitted']))==read(models['source'])
    assert read(models['initial'])['movement_options']['enabled'] and read(models['fitted'])['movement_options']['enabled']
    a.out.mkdir(parents=True)
    env=dict(os.environ,GOCACHE=str(repo/'workspace/build/gocache'),
        Q2COOPBOT_DIAGNOSTIC_ENGINE_ROOT=str(a.engine.resolve()))
    pins={str(path):sha(path) for path in set(models.values())|{a.curriculum.resolve()}}
    for folder,pattern in (('scripts','*.py'),('scripts','*.ps1'),('cmd','*.go'),('internal','*.go')):
        for path in (repo/folder).rglob(pattern):pins[str(path)]=sha(path)
    save(a.out/'protocol.json',dict(version='combat_movement_options_four_arm_validation_v1',
        models={label:str(path) for label,path in models.items()},inputs=pins,split='validation',
        seed_offset=a.seed_offset,validation_ids=ids,native_games=152,cuda_checks=114,
        max_instances=16,timescale=2,training_performed=False,promotion=False,
        scope='One frozen seed per validation descriptor. Report every outcome. No model selection or automatic promotion.'))
    save(a.out/'runner.json',dict(pid=os.getpid(),started_utc=now(),executable=sys.executable))
    def progress(stage):
        save(a.out/'progress.json',dict(stage=stage,utc=now(),runner_pid=os.getpid()))
        print(stage,flush=True)
    try:
        progress('building')
        for name in ('q2episode','q2ppo-data'):
            command(['go','build','-o',a.out/(name+'.exe'),'./cmd/'+name],a.out,name+'-build',env)
        progress('planning');plans=[];expected=None
        for label,model in models.items():
            folder=a.out/label;folder.mkdir();path=folder/'plan.json'
            command([a.out/'q2episode.exe','--root',repo,'--episodes',','.join(ids),
                '--mode','rules' if label=='rules' else 'learned',*(['--model',model] if label!='rules' else []),
                *(['--execution-contract','stock_weapons_fire_v1'] if label!='rules' else []),'--split','validation',
                '--count','1','--seed-offset',a.seed_offset,'--checked-wall-slide',
                '--out',path,'--artifacts',folder/'captures'],folder,'plan',env)
            worlds={(t['episode']['id'],i['engine_seed']):i for t in read(path)['tasks'] for i in t['instances']}
            assert len(worlds)==38
            if expected is None:expected=worlds
            else:assert worlds==expected,'Validation worlds differ across arms'
            plans.append(path)
        progress('collecting')
        # PowerShell -File cannot bind an arbitrary following argv list to a
        # string[] parameter; use a literal argument array through -Command.
        quote=lambda value:"'"+str(value).replace("'","''")+"'"
        script='& '+quote(repo/'scripts/run_registered_combat_pool.ps1')+' -Plans @('+','.join(quote(v) for v in plans)+') -MaxInstances 16 -Port '+str(a.port)+' -OutputRoot '+quote(a.out/'pool')
        command([shutil.which('pwsh.exe'),'-NoProfile','-Command',script],a.out,'collect',env)
        pool=read(a.out/'pool/report.json')
        assert pool['state']=='complete' and pool['source_unchanged'] and len(pool['jobs'])==152
        assert all(not j['error'] for j in pool['jobs'])
        progress('exporting')
        labels=list(models)
        def export(job):
            label=labels[job['plan_index']];folder=a.out/label;index=job['task_index']
            native=folder/f'export-{index:02d}-native';cuda=folder/f'export-{index:02d}-cuda'
            command([a.out/'q2ppo-data.exe','--batch',job['root'],'--model',models[label],
                '--cuda-only','--out',native],folder,f'export-{index:02d}',env)
            return dict(model=str(models[label]),model_sha256=sha(models[label]),data=str(native),
                out=str(cuda),native_report_sha256=sha(native/'report.json'))
        jobs=[j for j in pool['jobs'] if j['mode']=='learned'];assert len(jobs)==114
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:tasks=list(executor.map(export,jobs))
        save(a.out/'cuda-request.json',dict(version='combat_cuda_finalize_batch_v1',compact_finalized_streams=True,tasks=tasks))
        progress('cuda-replay')
        command([sys.executable,repo/'scripts/finalize_combat_cuda_batch.py','--request',a.out/'cuda-request.json',
            '--receipt-root',a.out/'cuda-batch'],a.out,'cuda',env)
        metrics=[]
        for job in pool['jobs']:
            cap=read(Path(job['root'])/'report.json');assert cap['capture_complete'] and cap['provenance_valid']
            result=cap['results'][0];assert result['capture_valid']
            life=result['first_life'];damage=life['damage']
            metrics.append(dict(label=labels[job['plan_index']],seed=job['seed'],map=life['map'],
                goal=bool(result['goal_stop']),death=bool(result['death_stop']),
                dealt_hp=sum(x['monster_health_damage'] for x in damage['by_mod']),received_hp=damage['received_health_damage'],
                self_hp=damage['self_health_damage'],kills=sum(x['monster_kills'] for x in damage['by_mod'])))
        totals={label:{k:sum(m[k] for m in metrics if m['label']==label)
            for k in ('goal','death','dealt_hp','received_hp','self_hp','kills')} for label in models}
        assert all(sha(Path(path))==digest for path,digest in pins.items()),'Evaluation sources changed'
        save(a.out/'summary.json',dict(state='complete',cases=metrics,totals=totals,
            native_games=152,cuda_checks=114,training_performed=False,promotion=False))
        save(a.out/'complete.json',dict(state='complete',summary_sha256=sha(a.out/'summary.json')))
        progress('complete');print(totals,flush=True)
    except BaseException as error:
        save(a.out/'failure.json',dict(state='failed',error=str(error),utc=now()));raise


if __name__=='__main__':main()
