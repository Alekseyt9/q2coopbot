"""Matched own-policy CUDA continuation for legacy versus checked wall execution."""
import argparse
import os
import pathlib
import shutil
import subprocess
import sys

from process_combat_architecture_pool import read, save, sha


def main():
    ap = argparse.ArgumentParser()
    for name in ('source', 'registry', 'config', 'anchor', 'out'):
        ap.add_argument('--' + name, type=pathlib.Path, required=True)
    ap.add_argument('--seed-offset', type=int, required=True)
    ap.add_argument('--count', type=int, default=2)
    ap.add_argument('--port', type=int, default=41800)
    ap.add_argument('--prepare-only', action='store_true')
    args = ap.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[1]
    source, registry, config, anchor, out = [p.resolve() for p in
        (args.source, args.registry, args.config, args.anchor, args.out)]
    assert 1 <= args.count <= 8 and not out.exists()
    assert shutil.disk_usage('F:/').free >= 24 * 1024**3
    assert not read(source/'weights.json')['deterministic']
    seal = read(source/'complete.json')
    assert read(source/'report.json')['device'] == 'cuda'
    for name, key in [('weights.json', 'weights_sha256'), ('checkpoint.pt', 'checkpoint_sha256'),
                      ('report.json', 'report_sha256')]:
        assert sha(source/name) == seal[key]
    assert anchor.name == 'anchor.json' and (anchor.parent/'bank.json').is_file()
    out.mkdir(parents=True)
    os.environ['GOCACHE'] = str(repo/'workspace/build/gocache')
    os.environ['GOTOOLCHAIN'] = 'auto'

    def run(label, command):
        with (out/(label+'.log')).open('w', encoding='utf-8') as log:
            result = subprocess.run(list(map(str, command)), cwd=repo, stdout=log,
                stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        assert result.returncode == 0, str(out/(label+'.log'))

    def python(label, script, *options):
        run(label, [sys.executable, repo/'scripts'/script, *options])

    pins = {p: sha(p) for p in (source/'weights.json', source/'checkpoint.pt',
        source/'complete.json', source/'report.json', registry, config, anchor, anchor.parent/'bank.json')}
    try:
        run('build-planner', ['go', 'build', '-o', out/'q2episode.exe', './cmd/q2episode'])
        for name, p in [('config.json', config), ('anchor.json', anchor),
                        ('bank.json', anchor.parent/'bank.json')]:
            shutil.copyfile(p, out/name)
        registry_value = read(registry)
        families = [pathlib.Path(p).stem for p in registry_value['files']]
        assert len(families) == len(set(families)) == 28
        history = []
        used = set()
        base = repo/'workspace/artifacts'
        paths = set(base.glob('*plan*.json')) | set(base.glob('*/*plan*.json'))
        paths |= set(base.glob('*/round-*/*plan*.json')) | set(base.glob('*/preflight/*plan*.json'))
        for p in sorted(paths):
            value = read(p)
            if not isinstance(value, dict) or not isinstance(value.get('tasks'), list):
                continue
            history.append(dict(path=str(p), sha256=sha(p)))
            for task in value['tasks']:
                used.update(task.get('seeds', []))
        labels = ['legacy', 'checked']
        plans = []
        for label in labels:
            p = out/(label+'-plan.json')
            command = [out/'q2episode.exe', '--registry', registry, '--root', repo,
                '--episodes', ','.join(families), '--split', 'train', '--mode', 'learned',
                '--model', source/'weights.json', '--count', args.count,
                '--seed-offset', args.seed_offset, '--out', p, '--artifacts', out/('captures-'+label)]
            if label == 'checked':
                command.append('--checked-wall-slide')
            run('plan-'+label, command)
            plans.append(p)
        a, b = map(read, plans)
        assert not a.get('checked_wall_slide', False) and b['checked_wall_slide']
        assert a['model_sha256'] == b['model_sha256'] == sha(source/'weights.json')
        assert len(a['tasks']) == len(b['tasks']) == len(families)
        new_seeds = set()
        for x, y in zip(a['tasks'], b['tasks']):
            for field in ('episode', 'instances', 'seeds', 'split', 'reward_sha256'):
                assert x[field] == y[field], field
            assert x['split'] == 'train' and len(x['seeds']) == args.count
            assert x['reward_sha256'] == read(config)['objective_reward_sha256']
            assert not used.intersection(x['seeds']) and not new_seeds.intersection(x['seeds'])
            new_seeds.update(x['seeds'])
        expected = len(new_seeds) * 2
        save(out/'protocol.json', dict(state='prepared', source=str(source),
            inputs=[dict(path=str(p), sha256=d) for p, d in pins.items()],
            historical_plans=history, plans=[dict(path=str(p), sha256=sha(p)) for p in plans],
            native_cases=expected, cases_per_arm=len(new_seeds), seed_offset=args.seed_offset,
            device='cuda', slots=16, timescale=2, training_performed=False, promotion=False,
            budget='Same initial checkpoint, scenes, seed assignments and PPO config/optimizer steps; actual transitions can differ.',
            script_sha256=sha(__file__)))
        if args.prepare_only:
            save(out/'progress.json', dict(stage='prepared', training_performed=False))
            return
        save(out/'progress.json', dict(stage='native-own-policy', native_cases=expected))
        literal = lambda p: "'" + str(p).replace("'", "''") + "'"
        command = '& ' + literal(repo/'scripts/run_registered_combat_pool.ps1')
        command += ' -Plans @(' + ','.join(map(literal, plans)) + ') -MaxInstances 16'
        command += ' -Port ' + str(args.port) + ' -OutputRoot ' + literal(out/'pool')
        run('capture', [shutil.which('pwsh'), '-NoProfile', '-Command', command])
        pool = read(out/'pool/report.json')
        assert pool['state'] == 'complete' and pool['source_unchanged']
        assert pool['usable_captures'] == len(pool['jobs']) == expected
        assert all(not j['error'] for j in pool['jobs'])
        for job in pool['jobs']:
            assert job['plan_index'] in (0, 1) and job['mode'] == 'learned'
            manifest = read(pathlib.Path(job['root'])/'manifest.json')
            assert manifest['checked_wall_slide'] == (job['plan_index'] == 1)
            assert manifest['model_weights_sha256'].lower() == sha(source/'weights.json')
        updates = []
        for index, label in enumerate(labels):
            processing = out/('processing-'+label)
            save(out/'progress.json', dict(stage='cuda-processing', label=label, updates=updates))
            python('processing-'+label, 'process_registered_combat_pool_cuda.py',
                '--pool', out/'pool', '--plan-index', index, '--out', processing)
            meta = read(processing/'report.json')
            assert meta['device'] == 'cuda' and len(meta['cases']) == len(new_seeds)
            retention = out/('retention-'+label+'.json')
            python('retention-'+label, 'prepare_combat_sequence_retention.py', '--pool', out/'pool',
                '--plan-index', index, '--processing', processing, '--out', retention)
            update = out/('update-'+label)
            save(out/'progress.json', dict(stage='gpu-training', label=label, updates=updates))
            python('training-'+label, 'ppo_recurrent.py', '--model', source/'weights.json',
                '--data', processing/'merged', '--config', out/'config.json',
                '--resume', source/'checkpoint.pt', '--anchor-model', out/'anchor.json',
                '--retention-bank', out/'bank.json', '--retention-weight', 0, '--bank-weight', 0,
                '--sequence-retention-spec', retention, '--sequence-retention-weight', 1,
                '--fork-sequence-retention', '--actor-head-scope', 'all', '--weapon-state-lr', .0003,
                '--out', update)
            python('checkpoint-audit-'+label, 'audit_combat_full_checkpoint_cuda.py',
                '--before', source/'checkpoint.pt', '--after', update/'checkpoint.pt',
                '--report', update/'report.json', '--spec', retention,
                '--out', out/('checkpoint-audit-'+label+'.json'))
            python('input-use-'+label, 'audit_combat_input_use_cuda.py',
                '--behavior', source/'weights.json', '--model', update/'weights.json',
                '--data', processing/'merged', '--out', out/('input-use-'+label+'.json'))
            closed = read(update/'complete.json')
            for name, key in [('weights.json', 'weights_sha256'), ('checkpoint.pt', 'checkpoint_sha256'),
                              ('report.json', 'report_sha256')]:
                assert sha(update/name) == closed[key]
            report = read(update/'report.json')
            assert report['device'] == 'cuda' and report['actor_steps'] == 10
            assert report['actor_optimizer_group_lrs'] == [.00003, .0003]
            assert read(update/'weights.json')['weapon_exploration'] == read(source/'weights.json')['weapon_exploration']
            updates.append(dict(label=label, update=str(update), rows=meta['rows'], **closed))
        for p, d in pins.items():
            assert sha(p) == d, str(p)
        save(out/'complete.json', dict(state='complete', device='cuda', native_cases=expected,
            cuda_cases=expected, updates=updates, protocol_sha256=sha(out/'protocol.json'),
            pool_sha256=sha(out/'pool/report.json'), training_performed=True, promotion=False,
            scope='One equal-config own-policy update per execution arm; no quality or coop acceptance.'))
        save(out/'progress.json', dict(stage='complete', updates=updates))
    except BaseException as error:
        save(out/'progress.json', dict(stage='failed', error=str(error)))
        raise


if __name__ == '__main__':
    main()
