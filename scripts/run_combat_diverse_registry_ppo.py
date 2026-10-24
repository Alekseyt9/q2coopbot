"""Run the frozen eight-update registry curriculum; reuse only sealed steps on resume."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys

from evaluate_registered_combat_models import read, save, sha


def process_alive(pid):
    if not pid:
        return False
    # os.kill(pid, 0) is not a safe liveness probe on Windows.
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel.OpenProcess(0x100000, False, int(pid))
    if not handle:
        assert ctypes.get_last_error() != 5, 'Cannot inspect previous runner; do not restart'
        return False
    try:
        return kernel.WaitForSingleObject(handle, 0) == 258
    finally:
        kernel.CloseHandle(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'evaluation-spec', 'out'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = args.out.resolve()
    protocol_path, evaluation_path = args.protocol.resolve(), args.evaluation_spec.resolve()
    protocol, evaluation = read(protocol_path), read(evaluation_path)
    assert protocol['version'] == 'combat_diverse_registry_ppo_v1' and protocol['rounds'] == 8
    assert protocol['cases_per_update'] == 48 and len(protocol['episodes']) == 24
    assert evaluation['experiment_protocol_sha256'] == sha(protocol_path)
    os.environ['GOCACHE'] = str(repo / 'workspace/build/gocache')
    os.environ['GOTOOLCHAIN'] = 'auto'
    if args.resume:
        run = read(root / 'run.json')
        assert run['protocol_sha256'] == sha(protocol_path) and run['evaluation_spec_sha256'] == sha(evaluation_path)
        previous = read(root / 'progress.json')
        assert not process_alive(previous.get('runner_pid')) and not process_alive(previous.get('child_pid')), 'Previous job is live; inspect its original handle'
        source_files = run['source_files']
    else:
        assert not root.exists(), 'Fresh output required; --resume only reuses sealed steps'
        root.mkdir(parents=True)
        paths = set()
        for directory, suffixes in ((repo / 'scripts', ('.py', '.ps1')), (repo / 'internal', ('.go',)),
                                    (repo / 'cmd', ('.go',)), (repo.parent / 'yquake2/src', ('.c', '.h'))):
            paths.update(p.resolve() for p in directory.rglob('*') if p.is_file() and p.suffix in suffixes)
        source_files = {str(p): sha(p) for p in sorted(paths)}
        save(root / 'run.json', dict(version='combat_diverse_registry_runner_v1', protocol=str(protocol_path),
             protocol_sha256=sha(protocol_path), evaluation_spec=str(evaluation_path), evaluation_spec_sha256=sha(evaluation_path),
             source_files=source_files, training_rounds=8, native_training_cases=384, max_instances=16, timescale=2,
             scope='Resume accepts sealed completed steps only; partial outputs are preserved and require inspection. No automatic retry of battles.'))

    def unchanged():
        assert sha(protocol_path) == read(root / 'run.json')['protocol_sha256']
        assert sha(evaluation_path) == read(root / 'run.json')['evaluation_spec_sha256']
        for path, digest in source_files.items():
            assert sha(path) == digest, 'Source changed during experiment: ' + path

    def stage(name, round_number=0, child_pid=None):
        save(root / 'progress.json', dict(stage=name, round=round_number, runner_pid=os.getpid(), child_pid=child_pid))

    def run_step(label, command, output, round_number):
        unchanged()
        if (output / 'complete.json').exists():
            stage('reuse-' + label, round_number)
            return
        assert not output.exists(), 'Incomplete output preserved; inspect existing jobs before recovery: ' + str(output)
        log = root / f'round-{round_number:02d}-{label}.log'
        with log.open('w', encoding='utf-8') as stream:
            child = subprocess.Popen([sys.executable, *map(str, command)], cwd=repo, stdout=stream,
                stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            stage(label, round_number, child.pid)
            result = child.wait()
        assert result == 0, str(log)
        assert (output / 'complete.json').exists(), 'Missing completion seal: ' + str(output)
        unchanged()

    def sealed_report(directory):
        seal = read(directory / 'complete.json')
        assert sha(directory / 'report.json') == seal['report_sha256']
        return seal, read(directory / 'report.json')

    try:
        unchanged()
        history = set(evaluation['historical_plans'])
        source = Path(protocol['initializer']).resolve()
        for number in range(1, 9):
            directory = root / f'round-{number:02d}'
            directory.mkdir(exist_ok=True)
            source_seal, _ = sealed_report(source)
            assert sha(source / 'weights.json') == source_seal['weights_sha256']
            assert sha(source / 'checkpoint.pt') == source_seal['checkpoint_sha256']
            history_path = directory / 'history.json'
            expected_history = dict(historical_plans=sorted(history))
            if history_path.exists():
                assert read(history_path) == expected_history
            else:
                save(history_path, expected_history)
            corpus = directory / 'corpus'
            run_step('collect', ['scripts/collect_combat_registry_curriculum.py', '--source', source, '--registry', protocol['registry'],
                '--episodes', ','.join(protocol['episodes']), '--train-count', '2', '--history-spec', history_path,
                '--train-offset', str(protocol['initial_train_offset'] + 2 * (number - 1)), '--port', '48000',
                '--allow-groups', '--ppo-ready', '--train-only', '--out', corpus], corpus, number)
            closed = read(corpus / 'complete.json')
            assert closed['native_cases'] == closed['cuda_cases'] == 48 and sha(corpus / 'protocol.json') == closed['protocol_sha256']
            collection = read(corpus / 'protocol.json')
            assert Path(collection['source']).resolve() == source and collection['train_only']
            assignment = collection['assignments'][0]
            assert len(collection['assignments']) == 1 and assignment['split'] == 'train' and len(set(assignment['seeds'])) == 48
            plan = read(assignment['plan'])
            assert sha(assignment['plan']) == assignment['plan_sha256'] and plan['model_sha256'] == source_seal['weights_sha256']
            assert len(plan['tasks']) == 24 and all(len(t['seeds']) == 2 for t in plan['tasks'])
            used = {seed for name in history for task in read(name).get('tasks', []) for seed in task.get('seeds', [])}
            assert not set(assignment['seeds']).intersection(used), 'Historical TRAIN fixture reused'
            freshness = dict(state='passed', seed_count=48, offset=assignment['offset'], history_sha256=sha(history_path),
                             corpus_protocol_sha256=sha(corpus / 'protocol.json'))
            save(directory / 'freshness.json', freshness)
            history.add(assignment['plan'])
            native, advantage, update = (directory / name for name in ('native-credit', 'advantage-credit', 'update'))
            run_step('native-credit', ['scripts/audit_combat_native_credit.py', '--corpus', corpus, '--out', native], native, number)
            sealed_report(native)
            run_step('advantage-credit', ['scripts/audit_combat_advantage_credit.py', '--corpus', corpus,
                     '--native-audit', native, '--out', advantage], advantage, number)
            sealed_report(advantage)
            run_step('training', ['scripts/train_combat_diverse_registry_ppo_cuda.py', '--source', source, '--corpus', corpus,
                '--credit-audit', native, '--advantage-audit', advantage, '--protocol', protocol_path, '--out', update,
                '--arm', 'G', '--round', str(number), '--boundary-acceptance',
                'workspace/artifacts/combat-handoff-live-acceptance-sealed-v1-20261023'], update, number)
            seal, report = sealed_report(update)
            assert report['device'] == 'cuda' and report['version'] == protocol['version'] and report['round'] == number
            assert report['behavior_sha256'] == source_seal['weights_sha256'] and report['experiment_protocol_sha256'] == sha(protocol_path)
            assert report['checkpoint_optimizer_roundtrip_exact'] and not report['quality_verified']
            assert sha(update / 'weights.json') == seal['weights_sha256'] and sha(update / 'checkpoint.pt') == seal['checkpoint_sha256']
            source = update.resolve()
        assert Path(next(v['update'] for v in evaluation['variants'] if v['label'] == 'trained')).resolve() == source
        target = root / 'evaluation'
        run_step('evaluation', ['scripts/evaluate_registered_combat_models.py', '--spec', evaluation_path, '--out', target], target, 8)
        closed = read(target / 'complete.json')
        assert closed['native_cases'] == 72 and closed['cuda_cases'] == 48
        save(root / 'complete.json', dict(state='complete', version='combat_diverse_registry_runner_v1',
             completed_training_rounds=8, native_training_cases=384, cuda_training_cases=384,
             native_evaluation_cases=72, cuda_evaluation_cases=48, final_weights_sha256=sha(source / 'weights.json'),
             protocol_sha256=sha(protocol_path), evaluation_seal_sha256=sha(target / 'complete.json'), promotion=False))
        stage('complete', 8)
    except BaseException as error:
        save(root / 'progress.json', dict(stage='failed', runner_pid=os.getpid(), error=str(error)))
        raise


if __name__ == '__main__':
    main()
