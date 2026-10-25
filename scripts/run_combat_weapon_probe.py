"""Collect the fixed24case TRAIN weapon-choice diagnostic via the existing pool."""
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
from combat_weapon_probe import VERSION, validate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selection', required=True, type=Path)
    ap.add_argument('--protocol', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--port', type=int, default=48900)
    a = ap.parse_args(); repo = Path(__file__).resolve().parents[1]; out = a.out.resolve()
    assert not out.exists() and shutil.disk_usage('F:/').free >= 32*1024**3
    pins = {}
    def pin(path, expected=None):
        p = Path(path).resolve(); h = sha(p); assert expected is None or h == expected, str(p)
        pins[str(p)] = h; return p
    selected = read(pin(a.selection)); protocol = read(pin(a.protocol))
    assert sha(a.selection) == protocol['selection_sha256'] and protocol['native_budget'] == protocol['cuda_verification_budget'] == 24
    assert protocol['cases'] == selected['cases'] and protocol['pool'] == 8 and protocol['timescale'] == 2
    for p, h in selected['inputs'].items(): pin(p, h)
    for p in (repo/'scripts').glob('*.py'): pin(p)
    model_path = pin(selected['source_model'], selected['source_weights_sha256']); original = read(model_path)
    plan_path = next(Path(p) for p in selected['inputs'] if p.endswith('train-plan.json'))
    old_plan = read(plan_path); registry = pin(old_plan['registry_path'], old_plan['registry_sha256'])
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
        models = {'source': model_path}
        for label, mode in (('identity', 'identity'), ('shotgun', 'force')):
            m = copy.deepcopy(original)
            m['diagnostic_weapon_probe'] = dict(version=VERSION, frame=98, mode=mode,
                                               expected_equipped_weapon=1, forced_weapon=2)
            validate(m); path = out/(label+'-weights.json'); save(path, m); models[label] = pin(path)
        variants, plans = [], []
        for ci, case in enumerate(selected['cases']):
            task = next(t for t in old_plan['tasks'] if case['seed'] in t['seeds'])
            expected = task['instances'][task['seeds'].index(case['seed'])]
            offset = case['seed'] - task['episode']['splits']['train']['start']
            for repeat in range(2):
                for label, model in models.items():
                    i = len(variants); plan = out/f'plan-{i:02d}.json'
                    run(f'plan-{i:02d}', [planner, '--registry', registry, '--root', repo,
                        '--episodes', case['episode'], '--split', 'train', '--mode', 'learned', '--model', model,
                        '--count', 1, '--seed-offset', offset, '--checked-wall-slide', '--out', plan,
                        '--artifacts', out/f'captures-{i:02d}'])
                    p = read(plan)
                    assert p['tasks'][0]['instances'] == [expected] and p['tasks'][0]['seeds'] == [case['seed']]
                    variants.append(dict(plan_index=i, context_index=ci, repeat=repeat, label=label,
                        model=str(model), model_sha256=sha(model), context=case, instance=expected))
                    plans.append(plan)
        assert len(variants) == 24
        spec = dict(version='combat_native_weapon_choice_probe_v1', variants=variants, protocol=protocol,
                    inputs=pins, source_weights_sha256=selected['source_weights_sha256'],
                    intentional_paired_seed_reuse=True, diagnostic_only=True, training_performed=False)
        save(out/'spec.json', spec)
        quote = lambda p: "'" + str(p).replace("'", "''") + "'"
        command = '& ' + quote(repo/'scripts/run_registered_combat_pool.ps1')
        command += ' -Plans @(' + ','.join(map(quote, plans)) + ') -MaxInstances 8 -Port ' + str(a.port)
        command += ' -OutputRoot ' + quote(out/'pool')
        stage('native-capture', cases=24)
        run('native-pool', [shutil.which('pwsh'), '-NoProfile', '-Command', command])
        pool = read(out/'pool/report.json')
        assert pool['state'] == 'complete' and pool['source_unchanged'] and pool['usable_captures'] == 24
        assert len(pool['jobs']) == 24 and all(not j['error'] for j in pool['jobs'])
        def export(job):
            i = job['plan_index']; v = variants[i]; capture = Path(job['root']); directory = out/f'case-{i:02d}'
            directory.mkdir(); report = read(capture/'report.json'); manifest = read(capture/'manifest.json')
            result = report['results'][0]
            assert report['capture_complete'] and report['provenance_valid'] and result['capture_valid'] and result['dispatch_valid']
            assert result['seed_confirmed'] and result['seed'] == v['context']['seed'] and result['generated_fixture'] == v['instance']
            assert manifest['model_weights_sha256'].lower() == v['model_sha256']
            assert manifest['observation_timing_contract'] == 'frozen_world_v1'
            run(f'export-{i:02d}', [exporter, '--batch', capture, '--model', v['model'], '--cuda-only', '--out', directory/'native'])
            verify_memory_export(Path(result['root'])/'bot.jsonl', directory/'native/replay-0')
            compact_closed_exports(directory/'native')
            return dict(model=v['model'], model_sha256=v['model_sha256'], data=str(directory/'native'),
                        out=str(directory/'cuda'), native_report_sha256=sha(directory/'native/report.json'))
        stage('native-export')
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as workers:
            tasks = list(workers.map(export, sorted(pool['jobs'], key=lambda j: j['plan_index'])))
        save(out/'cuda-request.json', dict(version='combat_cuda_finalize_batch_v1', compact_finalized_streams=True, tasks=tasks))
        stage('cuda-verification')
        run('cuda-batch', [sys.executable, repo/'scripts/finalize_combat_cuda_batch.py', '--request',
                          out/'cuda-request.json', '--receipt-root', out/'cuda-batch'])
        batch = read(out/'cuda-batch/report.json'); assert batch['state'] == 'complete' and len(batch['tasks']) == 24
        for p, h in pins.items(): assert sha(p) == h, p
        save(out/'complete.json', dict(state='complete', version=spec['version'], native_cases=24, cuda_cases=24,
            spec_sha256=sha(out/'spec.json'), pool_sha256=sha(out/'pool/report.json'),
            cuda_receipt_sha256=sha(out/'cuda-batch/report.json'), diagnostic_only=True, training_performed=False,
            promotion=False, scope='Technical collection only; causal admission and outcomes are separate.'))
        stage('pair-acceptance')
        run('pair-acceptance', [sys.executable, repo/'scripts/audit_combat_weapon_probe_v2.py', '--probe', out, '--out', out/'acceptance'])
        stage('complete'); print(out/'acceptance/report.json')
    except BaseException as error:
        stage('failed', error=str(error), exception=type(error).__name__); raise


if __name__ == '__main__':
    main()
