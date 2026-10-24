"""Run a frozen two-branch, two-update objective experiment; one pool at a time."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from evaluate_registered_combat_models import read, save, sha
from run_combat_diverse_registry_ppo import process_alive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'preflight-spec', 'evaluation-spec', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = args.out.resolve()
    protocol = read(args.protocol)
    assert protocol['version'] == 'combat_blocked_ground_ppo_v1' and protocol['rounds'] == 2
    assert protocol['cases_per_update'] == 16 and len(protocol['episodes']) == 8
    os.environ['GOCACHE'] = str(repo / 'workspace/build/gocache')
    os.environ['GOTOOLCHAIN'] = 'auto'
    configs = [args.protocol.resolve(), args.preflight_spec.resolve(), args.evaluation_spec.resolve()]
    if args.resume:
        run = read(root / 'run.json')
        state = read(root / 'progress.json')
        assert not process_alive(state.get('runner_pid')) and not process_alive(state.get('child_pid'))
        assert run['configs'] == {str(path): sha(path) for path in configs}
    else:
        assert not root.exists()
        root.mkdir(parents=True)
        paths = set(configs)
        for directory, suffixes in ((repo / 'scripts', ('.py', '.ps1')), (repo / 'internal', ('.go',)),
                                    (repo / 'cmd', ('.go',)), (repo.parent / 'yquake2/src', ('.c', '.h'))):
            paths.update(path.resolve() for path in directory.rglob('*') if path.is_file() and path.suffix in suffixes)
        run = dict(configs={str(path): sha(path) for path in configs},
                   source_files={str(path): sha(path) for path in sorted(paths)},
                   native_training_cases=64, native_preflight_cases=4, native_evaluation_cases=32,
                   max_instances=16, timescale=2,
                   resume='Only sealed steps; partial outputs preserved, never automatically replayed')
        save(root / 'run.json', run)

    def unchanged():
        assert all(sha(path) == digest for path, digest in run['source_files'].items())

    def stage(name, branch=None, number=None, child_pid=None):
        save(root / 'progress.json', dict(stage=name, branch=branch, round=number,
             runner_pid=os.getpid(), child_pid=child_pid))

    def sealed(directory):
        closed = read(directory / 'complete.json')
        assert closed['state'] == 'complete'
        if 'report_sha256' in closed:
            assert sha(directory / 'report.json') == closed['report_sha256']
        return closed

    def step(label, command, output, branch=None, number=None):
        unchanged()
        if (output / 'complete.json').exists():
            sealed(output)
            stage('reuse-' + label, branch, number)
            return
        assert not output.exists(), 'Partial output preserved; inspect before recovery: ' + str(output)
        with (root / (label + '.log')).open('w', encoding='utf-8') as log:
            child = subprocess.Popen([sys.executable, *map(str, command)], cwd=repo, stdout=log, stderr=log,
                                     creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            stage(label, branch, number, child.pid)
            result = child.wait()
        assert result == 0, str(root / (label + '.log'))
        sealed(output)
        unchanged()

    try:
        preflight = root / 'preflight'
        step('preflight', ['scripts/evaluate_registered_combat_models.py', '--spec', args.preflight_spec, '--out', preflight], preflight)
        zero_proof = root / 'zero-proof'
        step('zero-proof', ['scripts/audit_combat_blocked_ground_zero.py', '--evaluation', preflight, '--out', zero_proof], zero_proof)
        assert read(zero_proof / 'report.json')['observations_actions_samples_next_states_exact']
        history = set(protocol['historical_plans'])
        sources = {label: Path(binding['initializer']) for label, binding in protocol['branches'].items()}
        for number in (1, 2):
            paired_reference = None
            for label in ('zero', 'cost'):
                directory = root / label / f'round-{number:02d}'
                directory.mkdir(parents=True, exist_ok=True)
                history_path = directory / 'history.json'
                expected = dict(historical_plans=sorted(history))
                if history_path.exists():
                    assert read(history_path) == expected
                else:
                    save(history_path, expected)
                source = sources[label]
                source_seal = sealed(source)
                for name, key in (('weights.json', 'weights_sha256'), ('checkpoint.pt', 'checkpoint_sha256')):
                    assert sha(source / name) == source_seal[key]
                corpus = directory / 'corpus'
                command = ['scripts/collect_combat_blocked_ground_curriculum.py', '--source', source,
                    '--registry', protocol['branches'][label]['registry'], '--episodes', ','.join(protocol['episodes']),
                    '--train-count', '2', '--history-spec', history_path, '--train-offset', str(protocol['train_offset'] + 2*(number-1)),
                    '--port', '48000', '--allow-groups', '--ppo-ready', '--train-only', '--out', corpus]
                if paired_reference:
                    command += ['--paired-reference', paired_reference]
                prefix = f'{label}-{number:02d}'
                step(prefix + '-collect', command, corpus, label, number)
                closed = sealed(corpus)
                assert closed['native_cases'] == closed['cuda_cases'] == 16
                collection = read(corpus / 'protocol.json')
                assert sha(corpus / 'protocol.json') == closed['protocol_sha256']
                assignment = collection['assignments'][0]
                plan = Path(assignment['plan'])
                assert sha(plan) == assignment['plan_sha256']
                assert read(plan)['model_sha256'] == source_seal['weights_sha256']
                history.add(str(plan.resolve()))
                if label == 'zero':
                    paired_reference = plan.resolve()
                else:
                    assert read(corpus / 'paired-train-proof.json')['state'] == 'matched'
                native, advantage, update = (directory / name for name in ('native-credit', 'advantage-credit', 'update'))
                step(prefix + '-native', ['scripts/audit_combat_blocked_ground_native_credit.py', '--corpus', corpus, '--out', native], native, label, number)
                step(prefix + '-advantage', ['scripts/audit_combat_advantage_credit.py', '--corpus', corpus, '--native-audit', native, '--out', advantage], advantage, label, number)
                step(prefix + '-training', ['scripts/train_combat_blocked_ground_ppo_cuda.py', '--source', source, '--corpus', corpus,
                     '--credit-audit', native, '--advantage-audit', advantage, '--protocol', args.protocol, '--out', update,
                     '--arm', 'G', '--branch', label, '--round', str(number), '--boundary-acceptance', protocol['boundary_acceptance']], update, label, number)
                report = read(update / 'report.json')
                assert report['branch'] == label and report['round'] == number and report['device'] == 'cuda'
                assert report['behavior_sha256'] == source_seal['weights_sha256']
                assert report['checkpoint_optimizer_roundtrip_exact'] and report['rng_preserved']
                sources[label] = update.resolve()
        evaluation = root / 'evaluation'
        step('evaluation', ['scripts/evaluate_registered_combat_models.py', '--spec', args.evaluation_spec, '--out', evaluation], evaluation)
        closed = sealed(evaluation)
        assert closed['native_cases'] == 32 and closed['cuda_cases'] == 24
        save(root / 'complete.json', dict(state='complete', native_training_cases=64, cuda_training_cases=64,
             native_preflight_cases=4, cuda_preflight_cases=4, native_evaluation_cases=32, cuda_evaluation_cases=24,
             final_weights={label: sha(source / 'weights.json') for label, source in sources.items()},
             protocol_sha256=sha(args.protocol), evaluation_seal_sha256=sha(evaluation / 'complete.json'), promotion=False))
        stage('complete')
    except BaseException as error:
        save(root / 'progress.json', dict(stage='failed', runner_pid=os.getpid(), error=str(error)))
        raise


if __name__ == '__main__':
    main()
