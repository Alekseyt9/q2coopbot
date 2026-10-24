"""Fixed one-update curriculum ablation from identical decision-scope state."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from evaluate_registered_combat_models import read, save, sha
from run_combat_diverse_registry_ppo import process_alive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'spec', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = args.out.resolve()
    study, spec = read(args.protocol), read(args.spec)
    assert study['version'] == 'combat_curriculum_balance_v1'
    assert study['native_training_cases'] == 48 and study['cases_per_update'] == 24
    assert spec['experiment_protocol_sha256'] == sha(args.protocol)
    assert sha(study['prerequisite_report']) == study['prerequisite_report_sha256']
    assert all(sha(arm['contract']) == arm['contract_sha256'] for arm in study['arms'])
    os.environ['GOCACHE'] = str(repo / 'workspace/build/gocache')
    os.environ['GOTOOLCHAIN'] = 'auto'
    configs = {str(p.resolve()): sha(p) for p in (args.protocol, args.spec, Path(study['old_contract']),
        *(Path(arm['contract']) for arm in study['arms']))}
    if args.resume:
        run, status = read(root / 'run.json'), read(root / 'progress.json')
        assert not process_alive(status.get('runner_pid')) and not process_alive(status.get('child_pid'))
        assert run['configs'] == configs
    else:
        assert not root.exists()
        root.mkdir(parents=True)
        paths = set()
        for folder, suffixes in ((repo / 'scripts', ('.py', '.ps1')), (repo / 'internal', ('.go',)),
                                 (repo / 'cmd', ('.go',)), (repo.parent / 'yquake2/src', ('.c', '.h'))):
            paths.update(p.resolve() for p in folder.rglob('*') if p.is_file() and p.suffix in suffixes)
        run = dict(configs=configs, source_files={str(p): sha(p) for p in sorted(paths)},
            native_training_cases=48, native_evaluation_cases=32, max_instances=16, timescale=2)
        save(root / 'run.json', run)

    def stage(label, child_pid=None):
        save(root / 'progress.json', dict(stage=label, runner_pid=os.getpid(), child_pid=child_pid))

    def unchanged():
        assert all(sha(p) == h for p, h in configs.items())
        assert all(sha(p) == h for p, h in run['source_files'].items())

    def closed(directory):
        receipt = read(directory / 'complete.json')
        assert receipt['state'] == 'complete'
        if 'report_sha256' in receipt:
            assert sha(directory / 'report.json') == receipt['report_sha256']
        return receipt

    def step(label, command, target):
        unchanged()
        if (target / 'complete.json').exists():
            closed(target); stage('reuse-' + label); return
        assert not target.exists(), 'Partial output preserved: ' + str(target)
        with (root / (label + '.log')).open('w', encoding='utf-8') as log:
            child = subprocess.Popen([sys.executable, *map(str, command)], cwd=repo, stdout=log, stderr=log,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            stage(label, child.pid)
            code = child.wait()
        assert code == 0, str(root / (label + '.log'))
        closed(target); unchanged()

    try:
        source = Path(study['source'])
        seal = closed(source)
        assert seal['weights_sha256'] == study['source_weights_sha256']
        assert sha(source / 'weights.json') == seal['weights_sha256']
        assert sha(source / 'checkpoint.pt') == seal['checkpoint_sha256']
        history = set(study['historical_plans'])
        ledger = []
        for arm in study['arms']:
            label = arm['label']; directory = root / label
            directory.mkdir(exist_ok=True)
            contract = read(arm['contract'])
            assert len(contract['episodes']) * arm['train_count'] == 24
            initializer = directory / 'initializer'
            step(label + '-fork', ['scripts/fork_combat_balanced_curriculum_cuda.py', '--source', source,
                '--old-protocol', study['old_contract'], '--protocol', arm['contract'], '--out', initializer,
                '--expected-weights-sha256', seal['weights_sha256'], '--expected-scope-updates', '4'], initializer)
            assert closed(initializer)['weights_sha256'] == seal['weights_sha256']
            history_path = directory / 'history.json'
            expected = dict(historical_plans=sorted(history))
            if history_path.exists():
                assert read(history_path) == expected
            else:
                save(history_path, expected)
            corpus = directory / 'corpus'
            step(label + '-collect', ['scripts/collect_combat_registry_curriculum.py', '--source', initializer,
                '--registry', study['registry'], '--episodes', ','.join(contract['episodes']),
                '--train-count', arm['train_count'], '--train-offset', study['train_offset'],
                '--history-spec', history_path, '--port', '48000', '--allow-groups', '--ppo-ready', '--train-only',
                '--out', corpus], corpus)
            corpus_seal = closed(corpus)
            assert corpus_seal['native_cases'] == corpus_seal['cuda_cases'] == 24
            collection = read(corpus / 'protocol.json')
            assert sha(corpus / 'protocol.json') == corpus_seal['protocol_sha256']
            assignment = collection['assignments'][0]
            assert assignment['split'] == 'train' and len(set(assignment['seeds'])) == 24
            assert sha(assignment['plan']) == assignment['plan_sha256']
            assert read(assignment['plan'])['model_sha256'] == seal['weights_sha256']
            history.add(str(Path(assignment['plan']).resolve()))
            native, advantage, update = (directory / name for name in ('native-credit', 'advantage-credit', 'update'))
            step(label + '-native', ['scripts/audit_combat_blocked_ground_native_credit.py', '--corpus', corpus,
                '--out', native], native)
            step(label + '-advantage', ['scripts/audit_combat_advantage_credit.py', '--corpus', corpus,
                '--native-audit', native, '--out', advantage], advantage)
            step(label + '-train', ['scripts/train_combat_decision_scope_cuda.py', '--source', initializer,
                '--corpus', corpus, '--native-credit', native, '--advantage-credit', advantage,
                '--protocol', arm['contract'], '--out', update], update)
            report = read(update / 'report.json')
            assert report['device'] == 'cuda' and report['scope_updates'] == 5
            assert report['behavior_sha256'] == seal['weights_sha256']
            assert report['checkpoint_roundtrip_exact'] and report['disposable_resume_adam_step_exact']
            assert report['frozen_coordinates_exact']
            ledger.append(dict(label=label, rows=report['rows'], actor_steps=report['actor_steps'],
                critic_steps=report['critic_steps'], weights_sha256=closed(update)['weights_sha256']))
        # Verify equal update budgets and inherited consumed prefix on CUDA.
        from train_combat_bc import torch, training_devices
        training_devices()
        parent = torch.load(source / 'checkpoint.pt', map_location='cuda', weights_only=True)
        for arm in study['arms']:
            update = root / arm['label'] / 'update'
            assert sha(update / 'checkpoint.pt') == closed(update)['checkpoint_sha256']
            cp = torch.load(update / 'checkpoint.pt', map_location='cuda', weights_only=True)
            assert cp['updates_completed'] == parent['updates_completed'] + 1
            assert cp['consumed_rollouts'][:-1] == parent['consumed_rollouts']
        save(root / 'matched-update-counts.json', dict(state='matched', parent_updates=parent['updates_completed'],
            new_updates_per_arm=1, cases_per_arm=24, ledger=ledger))
        evaluation = root / 'evaluation'
        step('evaluation', ['scripts/evaluate_registered_combat_models.py', '--spec', args.spec, '--out', evaluation], evaluation)
        assert closed(evaluation)['native_cases'] == 32 and closed(evaluation)['cuda_cases'] == 24
        for label, script in (('native-quality', 'audit_registered_native_action_quality.py'),
                              ('damage', 'audit_combat_damage_distribution.py')):
            step(label, ['scripts/' + script, '--evaluation', evaluation, '--out', root / label], root / label)
        step('result', ['scripts/report_combat_curriculum_balance.py', '--workflow', root, '--protocol', args.protocol,
                       '--spec', args.spec, '--out', root / 'result'], root / 'result')
        save(root / 'complete.json', dict(state='complete', version=study['version'], native_training_cases=48,
            cuda_training_cases=48, native_evaluation_cases=32, cuda_evaluation_cases=24,
            protocol_sha256=sha(args.protocol), result_seal_sha256=sha(root / 'result/complete.json'), promotion=False))
        stage('complete')
    except BaseException as error:
        save(root / 'progress.json', dict(stage='failed', runner_pid=os.getpid(), error=str(error)))
        raise


if __name__ == '__main__':
    main()
