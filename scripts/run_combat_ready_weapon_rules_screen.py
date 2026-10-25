"""Predeclared32case TRAIN screen:8worlds x2startingweapons x2controllers."""
import argparse
import concurrent.futures
import copy
import os
from pathlib import Path
import shutil
import subprocess
import sys
from evaluate_registered_combat_models import read, save, sha
from process_combat_architecture_pool import compact_closed_exports
from process_coop_learning_pool_cuda import verify_memory_export


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--registry', required=True, type=Path)
    ap.add_argument('--model', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--port', type=int, default=49100)
    a = ap.parse_args(); repo = Path(__file__).resolve().parents[1]; out = a.out.resolve()
    assert not out.exists() and shutil.disk_usage('F:/').free >= 24*1024**3
    pins = {}
    def pin(path):
        p = Path(path).resolve(); pins[str(p)] = sha(p); return p
    registry = pin(a.registry); model = pin(a.model)
    assert sha(model) == '7b47e0a5dcf8158704f2b8a9cd8023ef193c510dc54a9182761fee197568b0f3'
    assert not any(k.startswith('diagnostic_') for k in read(model))
    episodes = [read(pin(registry.parent/p)) for p in read(registry)['files']]
    selected = sorted((e for e in episodes if e['recipe']['loadout'] == 'blaster'), key=lambda e: e['id'])
    assert len(selected) == 8 and {e['map'] for e in selected} == {'base3','city1','jail1','ware2'}
    assert all(sum(e['map'] == name for e in selected) == 2 for name in ('base3','city1','jail1','ware2'))
    assert all('-site-01-' in e['id'] or '-site-02-' in e['id'] for e in selected)
    for p in (repo/'scripts').glob('*.py'): pin(p)
    out.mkdir(); os.environ['GOCACHE'] = str(repo/'workspace/build/gocache')
    def run(label, command):
        with (out/(label+'.log')).open('w', encoding='utf-8') as stream:
            result = subprocess.run(list(map(str, command)), cwd=repo, stdout=stream, stderr=subprocess.STDOUT,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        assert result.returncode == 0, str(out/(label+'.log'))
    def stage(name, **kwargs): save(out/'progress.json', dict(stage=name, **kwargs))
    try:
        planner, exporter = out/'q2episode.exe', out/'q2ppo-data.exe'
        run('build-planner', ['go','build','-o',planner,'./cmd/q2episode'])
        run('build-exporter', ['go','build','-o',exporter,'./cmd/q2ppo-data'])
        registries = {}
        for label in ('blaster','shotgun'):
            directory = out/('registry-'+label); directory.mkdir(); names=[]
            for source in selected:
                ep=copy.deepcopy(source); ep['recipe']['loadout']=label
                ep['initial_state']='TRAIN ready-weapon/rules screen; same inventory Blaster1 Shotgun1 Shells20; starts '+label
                name=ep['id']+'.json'; save(directory/name,ep); pin(directory/name); names.append(name)
            save(directory/'index.json',dict(version=1,files=names));registries[label]=pin(directory/'index.json')
        variants,plans=[],[]
        for ci,ep in enumerate(selected):
            matched=None
            for label in ('blaster','shotgun'):
                for mode in ('learned','rules'):
                    i=len(variants);plan=out/f'plan-{i:02d}.json'
                    cmd=[planner,'--registry',registries[label],'--root',repo,'--episodes',ep['id'],
                         '--split','train','--mode',mode,'--count',1,'--seed-offset',21,
                         '--checked-wall-slide','--out',plan,'--artifacts',out/f'captures-{i:02d}']
                    if mode=='learned':cmd+=['--model',model]
                    run(f'plan-{i:02d}',cmd);p=read(plan);t=p['tasks'][0];instance=t['instances'][0]
                    check=copy.deepcopy(instance);check.pop('loadout')
                    if matched is None:matched=check
                    assert check==matched and t['seeds']==[ep['splits']['train']['start']+21]
                    variants.append(dict(plan_index=i,context_index=ci,map=ep['map'],episode=ep['id'],
                        seed=t['seeds'][0],label=label,mode=mode,instance=instance,
                        model=str(model) if mode=='learned' else None))
                    plans.append(pin(plan))
        assert len(variants)==32 and len({v['seed'] for v in variants})==8
        spec=dict(version='combat_ready_weapon_rules_train_screen_v1',variants=variants,inputs=pins,
            model_sha256=sha(model),native_budget=32,cuda_budget=16,pool=8,timescale=2,independent_worlds=8,
            selection='All8Blaster episodes in pinned registry, TRAIN offset21; no future outcome selection.',
            intentional_paired_seed_reuse=True,training_performed=False,promotion=False,
            scope='TRAIN diagnosis only; different initial equipped weapons. No fitted change or held-out superiority claim.')
        save(out/'spec.json',spec)
        # Compact source snapshot excludes assets/binaries and preserves evidence
        # even if subsequent implementation changes in the shared checkout.
        snap=out/'source-snapshot'; records=[]
        files=[p for folder in ('cmd','internal') for p in (repo/folder).rglob('*.go')]
        files += [p for p in (repo/'scripts').iterdir() if p.suffix in ('.py','.ps1')]
        native=repo.parent/'yquake2'
        all_sources=[(p,Path('q2coopbot-src')/p.relative_to(repo)) for p in files]
        all_sources += [(p,Path('yquake2')/p.relative_to(native)) for p in (native/'src').rglob('*') if p.is_file() and p.suffix in ('.c','.h')]
        for source,relative in all_sources:
            target=snap/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
            assert sha(source)==sha(target)
            records.append(dict(source=str(source),copy=str(relative),sha256=sha(target)))
        save(snap/'manifest.json',dict(files=records))
        quote=lambda p:"'"+str(p).replace("'","''")+"'"
        command='& '+quote(repo/'scripts/run_registered_combat_pool.ps1')+' -Plans @('+','.join(map(quote,plans))+')'
        command+=' -MaxInstances 8 -Port '+str(a.port)+' -OutputRoot '+quote(out/'pool')
        stage('native-capture',cases=32)
        run('native-pool',[shutil.which('pwsh'),'-NoProfile','-Command',command])
        pool=read(out/'pool/report.json')
        assert pool['state']=='complete' and pool['source_unchanged'] and pool['usable_captures']==32
        assert len(pool['jobs'])==32 and all(not j['error'] for j in pool['jobs'])
        learned=[j for j in pool['jobs'] if variants[j['plan_index']]['mode']=='learned']
        def export(job):
            i=job['plan_index'];v=variants[i];capture=Path(job['root']);directory=out/f'case-{i:02d}';directory.mkdir()
            report=read(capture/'report.json');result=report['results'][0]
            assert report['capture_complete'] and report['provenance_valid'] and result['capture_valid'] and result['dispatch_valid']
            assert result['generated_fixture']==v['instance']
            run(f'export-{i:02d}',[exporter,'--batch',capture,'--model',model,'--cuda-only','--out',directory/'native'])
            verify_memory_export(Path(result['root'])/'bot.jsonl',directory/'native/replay-0');compact_closed_exports(directory/'native')
            return dict(model=str(model),model_sha256=sha(model),data=str(directory/'native'),out=str(directory/'cuda'),
                        native_report_sha256=sha(directory/'native/report.json'))
        stage('native-export')
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as workers:
            tasks=list(workers.map(export,sorted(learned,key=lambda j:j['plan_index'])))
        save(out/'cuda-request.json',dict(version='combat_cuda_finalize_batch_v1',compact_finalized_streams=True,tasks=tasks))
        stage('cuda-verification')
        run('cuda-batch',[sys.executable,repo/'scripts/finalize_combat_cuda_batch.py','--request',out/'cuda-request.json','--receipt-root',out/'cuda-batch'])
        batch=read(out/'cuda-batch/report.json');assert batch['state']=='complete' and len(batch['tasks'])==16
        assert all(sha(p)==h for p,h in pins.items())
        save(out/'complete.json',dict(state='complete',spec_sha256=sha(out/'spec.json'),pool_sha256=sha(out/'pool/report.json'),
            cuda_receipt_sha256=sha(out/'cuda-batch/report.json'),native_cases=32,cuda_cases=16,training_performed=False,promotion=False))
        stage('acceptance')
        run('acceptance',[sys.executable,repo/'scripts/audit_combat_ready_weapon_rules_screen.py','--screen',out,'--out',out/'acceptance'])
        stage('complete');print(out/'acceptance/report.json')
    except BaseException as error:
        stage('failed',error=str(error));raise


if __name__=='__main__':main()
