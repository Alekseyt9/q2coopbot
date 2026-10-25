"""Native paired zero-route migration check on three preselected TRAIN worlds."""
import argparse
import concurrent.futures
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from evaluate_registered_combat_models import read, save, sha
from process_combat_architecture_pool import compact_closed_exports
from process_coop_learning_pool_cuda import verify_memory_export
from audit_combat_contact_loss_probe import equivalent, load_steps


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selection', type=Path, required=True)
    ap.add_argument('--prepared', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--port', type=int, default=49400)
    ap.add_argument('--candidate', type=Path, help='Explicit nonzero technical candidate; identity is not expected')
    ap.add_argument('--control', type=Path, help='Explicit paired control instead of original source; requires candidate')
    ap.add_argument('--reference-source',action='store_true',help='Add the original source reference to explicit control/candidate')
    ap.add_argument('--include-rules',action='store_true',help='Add ordinary rules under identical generated conditions')
    ap.add_argument('--fresh-seed-offsets', type=int, nargs='+', help='Declared fresh TRAIN offsets for contact-loss diagnostic')
    ap.add_argument('--health-100', action='store_true', help='Explicit matched diagnostic cohort with player HP100')
    ap.add_argument('--engine-root', type=Path, help='Explicit immutable diagnostic engine.json plus q2ded.exe/baseq2/game.dll')
    a = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    out = a.out.resolve()
    assert not out.exists() and shutil.disk_usage('F:/').free >= 24*1024**3
    selected, prep = read(a.selection), read(a.prepared/'report.json')
    if a.fresh_seed_offsets:
        assert a.candidate and all(x>=0 and x!=21 for x in a.fresh_seed_offsets)
        assert len(set(a.fresh_seed_offsets))==len(a.fresh_seed_offsets)
        selected['cases']=[{**copy.deepcopy(c),'fresh_offset':offset} for c in selected['cases'] for offset in a.fresh_seed_offsets]
    count=(2+int(a.reference_source)+int(a.include_rules))*len(selected['cases'])
    cuda_count=count-int(a.include_rules)*len(selected['cases'])
    assert prep['state'] == 'passed' and not prep['training_performed']
    source, zero = Path(selected['source_model']).resolve(), Path(prep['weights']).resolve()
    assert sha(source) == selected['source_weights_sha256'] and sha(zero) == prep['weights_sha256']
    if a.candidate:
        zero = a.candidate.resolve()
        assert read(zero)['feature_version']=='combat_features_v13_memory_static_routes'
    if a.control:
        assert a.candidate
        source=a.control.resolve()
        assert read(source)['feature_version']=='combat_features_v13_memory_static_routes'
    assert not a.reference_source or a.control
    assert not a.include_rules or a.candidate
    original_source=Path(selected['source_model']).resolve()
    pins = {}
    def pin(p):
        p = Path(p).resolve(); pins[str(p)] = sha(p); return p
    engine_receipt = None
    assert not os.environ.get('Q2COOPBOT_DIAGNOSTIC_ENGINE_ROOT'), 'Unmanifested inherited engine override'
    if a.engine_root:
        engine_root = a.engine_root.resolve()
        receipt_path = pin(engine_root/'engine.json')
        engine_receipt = read(receipt_path)
        assert engine_receipt['version'] == 'combat_diagnostic_engine_v1'
        for name in ('q2ded.exe', 'baseq2/game.dll'):
            assert sha(pin(engine_root/name)) == engine_receipt['binaries'][name]
        if engine_receipt.get('source_manifest'):
            assert sha(pin(engine_receipt['source_manifest'])) == engine_receipt['source_manifest_sha256']
        os.environ['Q2COOPBOT_DIAGNOSTIC_ENGINE_ROOT'] = str(engine_root)
    for p in (source, zero, a.selection, a.prepared/'report.json'):
        pin(p)
    if a.reference_source:pin(original_source)
    for p in (repo/'scripts').glob('*.py'):
        pin(p)
    out.mkdir(parents=True)
    os.environ['GOCACHE'] = str(repo/'workspace/build/gocache')
    def run(label, command):
        with (out/(label+'.log')).open('w', encoding='utf-8') as log:
            r = subprocess.run(list(map(str, command)), cwd=repo, stdout=log, stderr=subprocess.STDOUT,
                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        assert r.returncode == 0, str(out/(label+'.log'))
    def stage(name):
        save(out/'progress.json', dict(stage=name))
    try:
        planner, exporter = out/'q2episode.exe', out/'q2ppo-data.exe'
        run('build-planner', ['go','build','-o',planner,'./cmd/q2episode'])
        run('build-exporter', ['go','build','-o',exporter,'./cmd/q2ppo-data'])
        variants, plans = [], []
        for ci, case in enumerate(selected['cases']):
            original = read(pin(case['source_plan']))
            registry = pin(original['registry_path'])
            assert sha(registry) == original['registry_sha256']
            ep = original['tasks'][0]['episode']
            offset = case.get('fresh_offset',case['seed']-ep['splits']['train']['start'])
            if a.health_100:
                ep=copy.deepcopy(ep);ep['generator']['distributions']['train']['health']=[100]
                folder=out/f'registry-{ci:02d}';folder.mkdir()
                name=ep['id']+'.json';save(folder/name,ep);pin(folder/name)
                save(folder/'index.json',dict(version=1,files=[name]));registry=pin(folder/'index.json')
            case['seed']=ep['splits']['train']['start']+offset
            models=[('source',source),('zero',zero)]
            if a.reference_source:models.append(('original',original_source))
            if a.include_rules:models.append(('rules',None))
            for label, model in models:
                i = len(variants)
                plan = out/f'plan-{i:02d}.json'
                mode='rules' if label=='rules' else 'learned'
                command=[planner,'--registry',registry,'--root',repo,'--episodes',case['episode'],
                    '--split','train','--mode',mode,'--count',1,'--seed-offset',offset,
                    '--checked-wall-slide','--out',plan,'--artifacts',out/f'captures-{i:02d}']
                if model is not None:command+=['--model',model]
                run(f'plan-{i:02d}',command)
                generated=read(plan)['tasks'][0]['instances']
                if label=='source' and (a.fresh_seed_offsets or a.health_100):case['instance']=generated[0]
                assert generated == [case['instance']]
                variants.append(dict(index=i,context_index=ci,label=label,mode=mode,model=str(model) if model else None,model_sha256=sha(model) if model else None,context=case))
                plans.append(pin(plan))
        save(out/'spec.json', dict(version='combat_memory_route_candidate_native_v1' if a.candidate else 'combat_memory_route_zero_native_v1',variants=variants,inputs=pins,
            native_budget=count,cuda_budget=cuda_count,pool=16,timescale=2,independent_worlds=len(selected['cases']),
            fixed_health_100=a.health_100,fresh_seed_offsets=a.fresh_seed_offsets,
            control_role='explicit_control' if a.control else 'original_source',control_model=str(source),
            selection_kind=selected.get('selection_kind','source_selected_development'),
            selection_description=selected.get('selection',''),
            diagnostic_engine_root=str(a.engine_root.resolve()) if a.engine_root else None,
            diagnostic_engine=engine_receipt,
            intentional_paired_seed_reuse=True,training_performed=False,promotion=False))
        # Freeze implementation evidence without game assets or binary duplication.
        snap = out/'source-snapshot'
        files = [p for folder in ('cmd','internal') for p in (repo/folder).rglob('*.go')]
        files += [p for p in (repo/'scripts').iterdir() if p.suffix in ('.py','.ps1')]
        records = []
        for p in files:
            relative = p.relative_to(repo)
            target = snap/relative; target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,target)
            assert sha(p) == sha(target)
            records.append(dict(path=str(relative),sha256=sha(target)))
        save(snap/'manifest.json',dict(files=records))
        quote = lambda p: "'"+str(p).replace("'","''")+"'"
        cmd = '& '+quote(repo/'scripts/run_registered_combat_pool.ps1')+' -Plans @('+','.join(map(quote,plans))+')'
        cmd += ' -MaxInstances 16 -Port '+str(a.port)+' -OutputRoot '+quote(out/'pool')
        stage('native-capture')
        run('native-pool',[shutil.which('pwsh'),'-NoProfile','-Command',cmd])
        pool = read(out/'pool/report.json')
        assert pool['state']=='complete' and pool['source_unchanged'] and pool['usable_captures']==count
        assert len(pool['jobs'])==count and all(not j['error'] for j in pool['jobs'])
        def export(job):
            i=job['plan_index'];v=variants[i];capture=Path(job['root']);d=out/f'case-{i:02d}';d.mkdir()
            report=read(capture/'report.json');result=report['results'][0]
            assert report['capture_complete'] and report['provenance_valid'] and result['capture_valid'] and result['dispatch_valid']
            assert result['seed_confirmed'] and result['generated_fixture']==v['context']['instance']
            assert read(capture/'manifest.json')['model_weights_sha256'].lower()==v['model_sha256']
            run(f'export-{i:02d}',[exporter,'--batch',capture,'--model',v['model'],'--cuda-only','--out',d/'native'])
            verify_memory_export(Path(result['root'])/'bot.jsonl',d/'native/replay-0')
            compact_closed_exports(d/'native')
            return dict(model=v['model'],model_sha256=v['model_sha256'],data=str(d/'native'),out=str(d/'cuda'),native_report_sha256=sha(d/'native/report.json'))
        stage('native-export')
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as workers:
            learned=[j for j in pool['jobs'] if variants[j['plan_index']]['mode']=='learned']
            tasks=list(workers.map(export,sorted(learned,key=lambda j:j['plan_index'])))
        save(out/'cuda-request.json',dict(version='combat_cuda_finalize_batch_v1',compact_finalized_streams=True,tasks=tasks))
        stage('cuda-verification')
        run('cuda-batch',[sys.executable,repo/'scripts/finalize_combat_cuda_batch.py','--request',out/'cuda-request.json','--receipt-root',out/'cuda-batch'])
        assert read(out/'cuda-batch/report.json')['state']=='complete'
        stage('identity-audit')
        assert all(sha(p)==digest for p,digest in pins.items())
        if a.candidate:
            save(out/'technical-parity.json',dict(state='passed',native_cases=count,cuda_cases=cuda_count,
                candidate_sha256=sha(zero),training_performed=False,promotion=False,
                scope='Explicit nonzero route candidate; native likelihood/context/bootstrap checked on CUDA. No identity or quality improvement claim.'))
        else:
            from audit_combat_memory_route_zero import audit
            audit(out, out/'acceptance-v2.json')
        stage('complete')
        print(out/('technical-parity.json' if a.candidate else 'acceptance-v2.json'),flush=True)
    except BaseException as error:
        save(out/'progress.json',dict(stage='failed',error=str(error)))
        raise


if __name__=='__main__':
    main()
