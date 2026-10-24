"""Uniformly reprocess all original captures after a goal-shutdown validation failure."""
import argparse
import concurrent.futures
import copy
from pathlib import Path
import shutil
import subprocess
import sys

from evaluate_registered_combat_models import read, save, sha
from run_combat_diverse_registry_ppo import process_alive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('workflow', 'protocol', 'spec', 'bridge', 'proof', 'metadata-incident', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--reuse-captures', type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    assert not args.out.exists()
    old = args.workflow / 'evaluation'
    status = read(args.workflow / 'progress.json')
    assert status['stage'] == 'failed' and not process_alive(status.get('runner_pid'))
    assert read(old / 'progress.json')['stage'] == 'failed'
    original = read(old / 'pool/report.json')
    assert original['state'] == 'failed' and original['source_unchanged']
    assert len(original['jobs']) == 32 and original['usable_captures'] == 31
    failed = [j for j in original['jobs'] if j['error']]
    assert len(failed) == 1 and failed[0]['job'] == 16
    bridge = read(args.bridge / 'report.json')
    assert sha(args.bridge / 'report.json') == read(args.bridge / 'complete.json')['report_sha256']
    proof = read(args.proof / 'report.json')
    assert sha(args.proof / 'report.json') == read(args.proof / 'complete.json')['report_sha256']
    assert proof['state'] == 'complete' and proof['original_evidence_unchanged']
    incident = read(args.metadata_incident)
    assert incident['state'] == 'documented' and len(incident['affected']) == 8
    assert all(sha(item['path']) == item['sha256'] and item['same_semantics_as_original_exported_expectation']
               for item in incident['affected'])
    run_path = args.workflow / 'run.json'
    assert sha(run_path) == bridge['original_run_sha256']
    for path, old_sha in read(run_path)['source_files'].items():
        if path in bridge['changes']:
            change = bridge['changes'][path]
            assert old_sha == change['old_sha256'] and sha(path) == change['new_sha256']
            assert sha(args.bridge / Path(path).name) == old_sha
        else:
            assert sha(path) == old_sha, path
    pins = {str(p.resolve()): sha(p) for p in (args.protocol, args.spec, run_path, old / 'pool/report.json',
        old / 'fixture-proof.json', args.bridge / 'report.json', args.proof / 'report.json', args.metadata_incident, Path(__file__))}
    spec, fixture = read(args.spec), read(old / 'fixture-proof.json')
    assert fixture['spec_sha256'] == sha(args.spec)
    args.out.mkdir(parents=True)
    evaluation = args.out / 'evaluation'
    evaluation.mkdir()
    (evaluation / 'pool').mkdir()
    for binding in fixture['plans']:
        p = Path(binding['path']); assert sha(p) == binding['sha256']
        shutil.copy2(p, evaluation / p.name)
        pins[str(p.resolve())] = binding['sha256']
    shutil.copy2(old / 'fixture-proof.json', evaluation / 'fixture-proof.json')
    shell = shutil.which('pwsh'); assert shell

    def command(label, values):
        with (args.out / (label + '.log')).open('w', encoding='utf-8') as log:
            result = subprocess.run(list(map(str, values)), cwd=repo, stdout=log, stderr=log,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        assert result.returncode == 0, str(args.out / (label + '.log'))

    def reprocess(job):
        plan = Path(fixture['plans'][job['plan_index']]['path'])
        schedule = read(plan)
        task = schedule['tasks'][job['task_index']]
        seed_index = task['seeds'].index(job['seed'])
        target = args.out / 'captures' / f"job-{job['job']:04d}"
        reused = args.reuse_captures / 'captures' / target.name if args.reuse_captures else None
        if reused and (reused / 'reprocessing.json').exists():
            assert job['mode'] == 'learned', 'Reuse admits only verified learned closures from v3'
            saved = read(reused / 'reprocessing.json')
            wrapper = args.reuse_captures / 'reprocess-wrapper-source.ps1'
            assert sha(wrapper) == saved['script_sha256']
            assert saved['finalizer_sha256'] == sha(repo / 'scripts/run_learned_combat_baseline.ps1')
            assert all(sha(p) == h for p, h in saved['inputs'].items())
            target = reused
            pins[str((reused / 'reprocessing.json').resolve())] = sha(reused / 'reprocessing.json')
            pins[str(wrapper.resolve())] = sha(wrapper)
        else:
            target.parent.mkdir(exist_ok=True)
            command(f"reprocess-{job['job']:04d}", [shell, '-NoProfile', '-File',
                repo / 'scripts/reprocess_combat_goal_capture.ps1', '-CaptureRoot', job['root'], '-Plan', plan,
                '-TaskIndex', job['task_index'], '-SeedIndex', seed_index, '-Port', job['port'], '-OutputRoot', target.resolve()])
        report, manifest, receipt = (read(target / name) for name in ('report.json', 'manifest.json', 'reprocessing.json'))
        assert report['capture_complete'] and report['provenance_valid'] and len(report['results']) == 1
        assert receipt['new_games'] is False and receipt['raw_evidence_unchanged']
        assert sha(target / 'report.json') == receipt['report_sha256']
        row = report['results'][0]
        assert row['capture_valid'] and row['dispatch_valid'] and row['seed_confirmed'] and row['frame_budget_valid']
        original_manifest = read(Path(job['root']) / 'manifest.json')
        assert manifest == original_manifest, 'Raw provenance must remain original'
        assert row['generated_fixture'] == task['instances'][seed_index]
        if job['mode'] == 'learned':
            config = read(Path(row['root']) / 'bot-config.json')
            assert sha(config['combat']['provider_file']) == row['provider_config_sha256'].lower()
        return dict(job, root=str(target.resolve()), error='', goal_boundary_reprocessed=True)

    try:
        save(args.out / 'progress.json', dict(stage='reprocessing-original-captures', cases=32, new_games=0))
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            jobs = list(executor.map(reprocess, sorted(original['jobs'], key=lambda j: j['job'])))
        derived = copy.deepcopy(original)
        derived.update(state='complete', usable_captures=32, jobs=jobs,
            goal_boundary_recovery=dict(original_pool=str((old / 'pool/report.json').resolve()),
                original_pool_sha256=sha(old / 'pool/report.json'), bridge=str(args.bridge.resolve()),
                proof=str(args.proof.resolve()), uniformly_reprocessed=32, retained_raw_battles=32, new_games=0))
        save(evaluation / 'pool/report.json', derived)
        save(evaluation / 'recovery-native-complete.json', dict(state='complete', version='combat_goal_pool_reprocessing_v1',
            pool_sha256=sha(evaluation / 'pool/report.json'), new_games=0, retained_raw_battles=32,
            uniformly_reprocessed=32, inputs=pins))
        save(evaluation / 'progress.json', dict(stage='native-recovered', training_performed=False))
        save(args.out / 'progress.json', dict(stage='cuda-verification', new_games=0))
        command('finalize', [sys.executable, repo / 'scripts/finalize_goal_recovered_registered_evaluation.py',
            '--evaluation', evaluation.resolve(), '--spec', args.spec.resolve()])
        for label, script in (('native-quality', 'audit_registered_native_action_quality.py'),
                              ('damage', 'audit_combat_damage_distribution.py')):
            command(label, [sys.executable, repo / 'scripts' / script, '--evaluation', evaluation.resolve(),
                '--out', (evaluation / label).resolve()])
        command('result', [sys.executable, repo / 'scripts/report_combat_curriculum_balance_recovered.py',
            '--workflow', args.workflow.resolve(), '--protocol', args.protocol.resolve(), '--spec', args.spec.resolve(),
            '--evaluation', evaluation.resolve(), '--out', (args.out / 'result').resolve()])
        assert all(sha(path) == digest for path, digest in pins.items())
        save(args.out / 'complete.json', dict(state='complete', version='combat_curriculum_goal_boundary_recovery_v1',
            result_seal_sha256=sha(args.out / 'result/complete.json'), inputs=pins,
            original_raw_battles_retained=32, new_games=0, training_performed=False, promotion=False))
        save(args.out / 'progress.json', dict(stage='complete', new_games=0))
    except BaseException as error:
        save(args.out / 'progress.json', dict(stage='failed', error=str(error), new_games=0))
        raise


if __name__ == '__main__':
    main()
