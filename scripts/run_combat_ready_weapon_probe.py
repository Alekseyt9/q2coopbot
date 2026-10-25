"""Fixed TRAIN-only starting-ready weapon comparison; existing native pool/CUDA exporter."""
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
    ap.add_argument('--selection', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--port', type=int, default=49000)
    a = ap.parse_args(); repo = Path(__file__).resolve().parents[1]; out = a.out.resolve()
    assert not out.exists() and shutil.disk_usage('F:/').free >= 24*1024**3
    pins = {}
    def pin(path, expected=None):
        p = Path(path).resolve(); h = sha(p); assert expected is None or h == expected, str(p)
        pins[str(p)] = h; return p
    selected = read(pin(a.selection))
    assert len(selected['cases']) == 4 and len({c['map'] for c in selected['cases']}) == 4
    for p, h in selected['inputs'].items(): pin(p, h)
    for p in (repo/'scripts').glob('*.py'): pin(p)
    model = pin(selected['source_model'], selected['source_weights_sha256'])
    assert not any(k.startswith('diagnostic_') for k in read(model))
    old_plan = read(next(Path(p) for p in selected['inputs'] if p.endswith('train-plan.json')))
    old_registry = pin(old_plan['registry_path'], old_plan['registry_sha256'])
    manifest = read(old_registry)
    episodes = [read(pin(old_registry.parent/p)) for p in manifest['files']]
    out.mkdir(parents=True); os.environ['GOCACHE'] = str(repo/'workspace/build/gocache')
    def run(label, command):
        with (out/(label+'.log')).open('w', encoding='utf-8') as stream:
            r = subprocess.run(list(map(str, command)), cwd=repo, stdout=stream, stderr=subprocess.STDOUT,
                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        assert r.returncode == 0, str(out/(label+'.log'))
    def stage(name, **kwargs): save(out/'progress.json', dict(stage=name, **kwargs))
    try:
        planner, exporter = out/'q2episode.exe', out/'q2ppo-data.exe'
        run('build-planner', ['go', 'build', '-o', planner, './cmd/q2episode'])
        run('build-exporter', ['go', 'build', '-o', exporter, './cmd/q2ppo-data'])
        registries = {}
        for label in ('blaster', 'shotgun'):
            directory = out/('registry-'+label); directory.mkdir(); names = []
            for case in selected['cases']:
                ep = copy.deepcopy(next(e for e in episodes if e['id'] == case['episode']))
                assert ep['recipe']['loadout'] == 'blaster' and ep['ppo_trainable']
                # Preserve ID/revision: generator seed is ID/revision/split/seed,
                # independent of loadout. Native geometry is verified again.
                ep['recipe']['loadout'] = label
                ep['initial_state'] = 'Ready-weapon TRAIN diagnostic; Blaster/Shotgun inventory with20shells; starts '+label
                name = ep['id']+'.json'; save(directory/name, ep); names.append(name)
            save(directory/'index.json', dict(version=1, files=names))
            registries[label] = pin(directory/'index.json')
            for name in names: pin(directory/name)
        variants, plans = [], []
        for ci, case in enumerate(selected['cases']):
            task = next(t for t in old_plan['tasks'] if case['seed'] in t['seeds'])
            expected = task['instances'][task['seeds'].index(case['seed'])]
            offset = case['seed'] - task['episode']['splits']['train']['start']
            for repeat in range(2):
                for label in ('blaster', 'shotgun'):
                    i = len(variants); plan = out/f'plan-{i:02d}.json'
                    run(f'plan-{i:02d}', [planner, '--registry', registries[label], '--root', repo,
                        '--episodes', case['episode'], '--split', 'train', '--mode', 'learned', '--model', model,
                        '--count', 1, '--seed-offset', offset, '--checked-wall-slide', '--out', plan,
                        '--artifacts', out/f'captures-{i:02d}'])
                    p = read(plan); instance = p['tasks'][0]['instances'][0]
                    check = copy.deepcopy(instance); check['loadout'] = expected['loadout']
                    assert check == expected and p['tasks'][0]['seeds'] == [case['seed']]
                    variants.append(dict(plan_index=i, context_index=ci, repeat=repeat, label=label,
                        model=str(model), model_sha256=sha(model), context=case, instance=instance))
                    plans.append(pin(plan))
        spec = dict(version='combat_starting_ready_weapon_train_probe_v1', variants=variants,
            inputs=pins, source_weights_sha256=selected['source_weights_sha256'], native_budget=16,
            cuda_verification_budget=16, pool=8, timescale=2, independent_worlds=4, cold_repeats=2,
            intentional_paired_seed_reuse=True, diagnostic_only=True, training_performed=False,
            scope='Different initial equipped weapon, not identical-state one-action counterfactual. Source policy remains free to choose weapons. No held-out evaluation or rules comparison.')
        save(out/'spec.json', spec)
        quote = lambda p: "'" + str(p).replace("'", "''") + "'"
        command = '& ' + quote(repo/'scripts/run_registered_combat_pool.ps1')
        command += ' -Plans @(' + ','.join(map(quote, plans)) + ') -MaxInstances 8 -Port ' + str(a.port)
        command += ' -OutputRoot ' + quote(out/'pool')
        stage('native-capture', cases=16)
        run('native-pool', [shutil.which('pwsh'), '-NoProfile', '-Command', command])
        pool = read(out/'pool/report.json')
        assert pool['state'] == 'complete' and pool['source_unchanged'] and pool['usable_captures'] == 16
        assert len(pool['jobs']) == 16 and all(not j['error'] for j in pool['jobs'])
        def export(job):
            i = job['plan_index']; v = variants[i]; capture = Path(job['root']); directory = out/f'case-{i:02d}'
            directory.mkdir(); report = read(capture/'report.json'); mf = read(capture/'manifest.json')
            result = report['results'][0]
            assert report['capture_complete'] and report['provenance_valid'] and result['capture_valid'] and result['dispatch_valid']
            assert result['seed_confirmed'] and result['generated_fixture'] == v['instance']
            assert mf['model_weights_sha256'].lower() == v['model_sha256'] and mf['observation_timing_contract'] == 'frozen_world_v1'
            run(f'export-{i:02d}', [exporter, '--batch', capture, '--model', model, '--cuda-only', '--out', directory/'native'])
            verify_memory_export(Path(result['root'])/'bot.jsonl', directory/'native/replay-0')
            compact_closed_exports(directory/'native')
            return dict(model=str(model), model_sha256=sha(model), data=str(directory/'native'),
                        out=str(directory/'cuda'), native_report_sha256=sha(directory/'native/report.json'))
        stage('native-export')
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as workers:
            tasks = list(workers.map(export, sorted(pool['jobs'], key=lambda j: j['plan_index'])))
        save(out/'cuda-request.json', dict(version='combat_cuda_finalize_batch_v1', compact_finalized_streams=True, tasks=tasks))
        stage('cuda-verification')
        run('cuda-batch', [sys.executable, repo/'scripts/finalize_combat_cuda_batch.py', '--request', out/'cuda-request.json', '--receipt-root', out/'cuda-batch'])
        batch = read(out/'cuda-batch/report.json'); assert batch['state'] == 'complete' and len(batch['tasks']) == 16
        for p, h in pins.items(): assert sha(p) == h, p
        save(out/'complete.json', dict(state='complete', version=spec['version'], native_cases=16, cuda_cases=16,
            spec_sha256=sha(out/'spec.json'), pool_sha256=sha(out/'pool/report.json'),
            cuda_receipt_sha256=sha(out/'cuda-batch/report.json'), diagnostic_only=True, training_performed=False,
            promotion=False, scope='Technical collection; ready-state and outcome audit separate.'))
        stage('acceptance')
        run('acceptance', [sys.executable, repo/'scripts/audit_combat_ready_weapon_probe.py', '--probe', out, '--out', out/'acceptance'])
        stage('complete'); print(out/'acceptance/report.json')
    except BaseException as error:
        stage('failed', error=str(error), exception=type(error).__name__); raise


if __name__ == '__main__': main()
