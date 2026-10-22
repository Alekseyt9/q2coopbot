"""Four fixed fresh PPO rounds, CUDA only, evaluate endpoints without selection.

Resume only reuses sealed phases. Incomplete phases are preserved and rejected.
No intermediate quality selection, no automatic checkpoint promotion.
"""
import argparse
import pathlib
import subprocess
import sys

from evaluate_registered_combat_models import read, save, sha
from run_registered_combat_ppo_pilot import lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=pathlib.Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    path = args.protocol.resolve(); root = path.parent; p = read(path)
    repo = pathlib.Path(__file__).resolve().parents[1]
    assert p['version'] == 'combat_ppo_learning_curve_v1'
    assert set(p['sources']) == {'full'} and p['fixed_endpoints'] == [0, 2, 4]
    assert p['rounds'] == 4 and p['cases_per_round_per_variant'] == 32
    assert p['new_training_cases'] == 128 and len(p['episodes']) == 8
    assert p['cuda_only'] and p['max_instances'] == 16 and p['timescale'] == 2
    assert not p['promotion'] and not p['intermediate_quality_selection']
    import torch
    assert torch.cuda.is_available()
    with lock(root / 'execution.lock'):
        assert args.resume or not (root / 'progress.json').exists()
        pins = {str(path): sha(path), **p['inputs']}
        for item in read(root / 'execution-source.json')['files']:
            pins[item['path']] = item['sha256']
        def verify():
            for name, digest in pins.items():
                assert sha(pathlib.Path(name)) == digest, 'Pinned input changed: ' + name
        def run(label, script, arguments, output):
            verify(); save(root / 'progress.json', dict(stage=label, completed=completed))
            if not output.exists():
                with (root / (label + '.log')).open('w', encoding='utf-8') as log:
                    result = subprocess.run([sys.executable, str(repo / 'scripts' / script),
                        *map(str, arguments)], cwd=repo, stdout=log, stderr=subprocess.STDOUT,
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                assert result.returncode == 0, 'Inspect ' + str(root / (label + '.log'))
            else:
                assert args.resume, 'Unexpected existing phase'
            seal = output / 'complete.json' if output.is_dir() else output
            value = read(seal)
            assert value.get('state', 'complete') == 'complete'
            if 'weights_sha256' in value:
                for file, key in [('weights.json', 'weights_sha256'), ('checkpoint.pt', 'checkpoint_sha256'), ('report.json', 'report_sha256')]:
                    assert sha(output / file) == value[key]
            verify(); return value
        completed = []; sources = {k: pathlib.Path(v) for k, v in p['sources'].items()}; seen = set()
        try:
            verify()
            for number in range(1, 5):
                for label, scope in [('full', 'all')]:
                    prefix = f'round-{number}-{label}'; source = sources[label]
                    corpus = root / (prefix + '-corpus')
                    run(prefix + '-collect', 'collect_combat_aim_feedback_corpus.py', [
                        '--source', source, '--registry', p['registry'], '--episodes', ','.join(p['episodes']),
                        '--train-offset', p['offsets'][number-1][label], '--port', p['ports'][number-1][label],
                        '--allow-groups', '--ppo-ready', '--train-only',
                        '--minimum-free-gib', p['minimum_free_gib'], '--out', corpus], corpus)
                    protocol = read(corpus / 'protocol.json')
                    assert pathlib.Path(protocol['source']) == source and protocol['train_only']
                    assert len(protocol['assignments']) == 1 and protocol['assignments'][0]['split'] == 'train'
                    seeds = set(protocol['assignments'][0]['seeds'])
                    assert len(seeds) == 32 and not seen.intersection(seeds)
                    seen.update(seeds)
                    retention = root / (prefix + '-retention.json')
                    run(prefix + '-retention', 'prepare_combat_sequence_retention.py', [
                        '--pool', corpus / 'pool', '--plan-index', '0',
                        '--processing', corpus / 'processing-train', '--out', retention], retention)
                    update = root / (prefix + '-update'); boundary = root / (prefix + '-initial.pt')
                    run(prefix + '-fit', 'ppo_recurrent.py', [
                        '--model', source / 'weights.json', '--resume', source / 'checkpoint.pt',
                        '--data', corpus / 'processing-train/merged', '--config', corpus / 'training-config.json',
                        '--anchor-model', p['anchor'], '--retention-bank', p['bank'],
                        '--retention-weight', '0', '--bank-weight', '0',
                        '--sequence-retention-spec', retention, '--sequence-retention-weight', '1', '--fork-sequence-retention',
                        '--actor-head-scope', scope, '--weapon-state-lr', '.0003', '--motor-lr', '.0003',
                        '--initial-boundary', boundary, '--out', update], update)
                    audit = root / (prefix + '-audit.json')
                    run(prefix + '-audit', 'audit_combat_motor_feedback_cuda.py', [
                        '--source', source, '--update', update, '--boundary', boundary,
                        '--corpus', corpus, '--scope', scope, '--out', audit], audit)
                    sources[label] = update
                    completed.append(dict(round=number, label=label, source=str(source), update=str(update),
                        cases=32, seeds=sorted(seeds), corpus_complete_sha256=sha(corpus / 'complete.json'),
                        update_complete_sha256=sha(update / 'complete.json'), audit_sha256=sha(audit)))
            assert len(seen) == 128
            save(root / 'training-complete.json', dict(state='complete', completed=completed,
                new_native_cases=128, device='cuda', promotion=False))
            run('evaluation', 'evaluate_registered_combat_models.py', [
                '--spec', root / 'evaluation-spec.json', '--out', root / 'evaluation'], root / 'evaluation')
            save(root / 'complete.json', dict(state='complete', protocol_sha256=sha(path),
                training_complete_sha256=sha(root / 'training-complete.json'),
                evaluation_complete_sha256=sha(root / 'evaluation/complete.json'), promotion=False))
            save(root / 'progress.json', dict(stage='complete', completed=completed))
        except BaseException as error:
            save(root / 'progress.json', dict(stage='failed', completed=completed, error=str(error)))
            raise


if __name__ == '__main__':
    main()
