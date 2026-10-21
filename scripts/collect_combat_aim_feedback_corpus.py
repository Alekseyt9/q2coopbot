"""Fresh own-policy corpus; optional checkpoint-aligned PPO training objective."""
import argparse
import os
import pathlib
import shutil
import subprocess
import sys

from evaluate_registered_combat_models import read, save, sha


def main():
    ap = argparse.ArgumentParser()
    for name in ('source', 'registry', 'out'):
        ap.add_argument('--' + name, type=pathlib.Path, required=True)
    ap.add_argument('--episodes', required=True)
    ap.add_argument('--train-offset', type=int, default=600)
    ap.add_argument('--development-offset', type=int, default=64)
    ap.add_argument('--port', type=int, default=42400)
    ap.add_argument('--allow-groups', action='store_true')
    ap.add_argument('--ppo-ready', action='store_true', help='Require reward SHA matching source checkpoint; allow all registered loadouts')
    ap.add_argument('--train-only', action='store_true', help='Collect training data only; evaluate quality with a separate held-out run')
    args = ap.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[1]
    source, registry, out = [p.resolve() for p in (args.source, args.registry, args.out)]
    assert not out.exists() and shutil.disk_usage('F:/').free >= 32 * 1024**3
    seal = read(source / 'complete.json')
    for name, key in [('weights.json', 'weights_sha256'), ('checkpoint.pt', 'checkpoint_sha256'),
                      ('report.json', 'report_sha256')]:
        assert sha(source / name) == seal[key]
    assert read(source / 'report.json')['device'] == 'cuda'
    assert not read(source / 'weights.json')['deterministic']
    names = args.episodes.split(',')
    assert len(names) == len(set(names)) and names
    descriptors = {read(registry.parent / f)['id']: read(registry.parent / f)
                   for f in read(registry)['files']}
    for name in names:
        e = descriptors[name]
        assert e['status'] == 'runnable' and e['ppo_trainable']
        assert args.ppo_ready or e['recipe']['loadout'] == 'machinegun'
        assert args.allow_groups or not e['recipe'].get('mixed'), 'Group collection requires --allow-groups'
        assert e['timescale'] == 2
    objective = None
    if args.ppo_ready:
        import torch
        assert torch.cuda.is_available(), 'CUDA required for checkpoint metadata; no CPU fallback'
        checkpoint = torch.load(source/'checkpoint.pt', map_location='cuda', weights_only=True)
        assert checkpoint['weights_sha256'] == seal['weights_sha256']
        config = checkpoint['config']
        expected = config.get('objective_reward_sha256'); assert expected
        assert all(sha(repo/descriptors[n]['recipe']['reward_config']) == expected for n in names), 'Registry reward differs from checkpoint objective'
        objective = dict(config=config, reward_sha256=expected, checkpoint_sha256=seal['checkpoint_sha256'])
        del checkpoint
    out.mkdir(parents=True)
    os.environ['GOCACHE'] = str(repo / 'workspace/build/gocache')
    os.environ['GOTOOLCHAIN'] = 'auto'

    def run(label, command):
        with (out / (label + '.log')).open('w', encoding='utf-8') as log:
            r = subprocess.run(list(map(str, command)), cwd=repo, stdout=log,
                stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        assert r.returncode == 0, str(out / (label + '.log'))

    pins = {p: sha(p) for p in (registry, source / 'weights.json', source / 'complete.json')}
    if objective:
        config_path = out/'training-config.json'; save(config_path, objective.pop('config'))
        objective.update(config=str(config_path), config_sha256=sha(config_path))
        pins[config_path] = sha(config_path)
        pins[source/'checkpoint.pt'] = seal['checkpoint_sha256']
        for name in names:
            pins[(repo/descriptors[name]['recipe']['reward_config']).resolve()] = objective['reward_sha256']
    historical = set()
    history = []
    base = repo / 'workspace/artifacts'
    paths = set(base.glob('*plan*.json')) | set(base.glob('*/*plan*.json'))
    paths |= set(base.glob('*/round-*/*plan*.json')) | set(base.glob('*/preflight/*plan*.json'))
    for path in sorted(paths):
        value = read(path)
        if not isinstance(value, dict) or not isinstance(value.get('tasks'), list):
            continue
        history.append(dict(path=str(path), sha256=sha(path)))
        for task in value['tasks']:
            historical.update(task.get('seeds', []))
    try:
        save(out / 'progress.json', dict(stage='preparing', training_performed=False))
        planner = out / 'q2episode.exe'
        run('build-planner', ['go', 'build', '-o', planner, './cmd/q2episode'])
        plans, chosen = [], []
        assignments = [('train', 'train', 4, args.train_offset)]
        if not args.train_only:
            assignments.append(('development', 'validation', 2, args.development_offset))
        for label, split, count, offset in assignments:
            # Search unused registered seed ranges before freezing a plan.
            while True:
                assert all(offset + count <= descriptors[n]['splits'][split]['count'] for n in names)
                candidates = {descriptors[n]['splits'][split]['start'] + offset + i
                              for n in names for i in range(count)}
                if not candidates & historical:
                    break
                offset += count
            plan = out / (label + '-plan.json')
            run('plan-' + label, [planner, '--registry', registry, '--root', repo,
                '--episodes', args.episodes, '--split', split, '--mode', 'learned',
                '--model', source / 'weights.json', '--count', count, '--seed-offset', offset,
                '--checked-wall-slide', '--out', plan, '--artifacts', out / ('captures-' + label)])
            frozen = read(plan)
            seeds = [s for t in frozen['tasks'] for s in t['seeds']]
            assert len(seeds) == len(set(seeds)) == len(names) * count
            assert not historical.intersection(seeds)
            assert not frozen.get('diagnostic_nominal_aim')
            assert all(t['split'] == split and t['modes'] == ['learned'] for t in frozen['tasks'])
            historical.update(seeds)
            plans.append(plan)
            chosen.append(dict(label=label, split=split, offset=offset, count=count, seeds=seeds,
                               plan=str(plan), plan_sha256=sha(plan)))
        save(out / 'protocol.json', dict(version='combat_aim_feedback_corpus_v1', state='prepared',
            source=str(source), inputs=[dict(path=str(p), sha256=h) for p, h in pins.items()],
            assignments=chosen, history=history, max_instances=16, timescale=2,
            diagnostic_intervention=False, allow_groups=args.allow_groups, train_only=args.train_only, ppo_training_objective=objective, training_performed=False,
            scope='Registered own-policy trajectories. PPO-ready requires checkpoint-aligned reward. Native poses are offline labels only; no validation data enters fitting. Development is not final quality evidence.'))
        shell = shutil.which('pwsh'); assert shell
        literal = lambda p: "'" + str(p).replace("'", "''") + "'"
        command = '& ' + literal(repo / 'scripts/run_registered_combat_pool.ps1')
        command += ' -Plans @(' + ','.join(map(literal, plans)) + ') -MaxInstances 16 -Port '
        command += str(args.port) + ' -OutputRoot ' + literal(out / 'pool')
        save(out / 'progress.json', dict(stage='native-collection', cases=sum(len(c['seeds']) for c in chosen)))
        run('native-pool', [shell, '-NoProfile', '-Command', command])
        pool = read(out / 'pool/report.json')
        assert pool['state'] == 'complete' and pool['source_unchanged']
        assert pool['usable_captures'] == len(pool['jobs']) == sum(len(c['seeds']) for c in chosen)
        processed = []
        for index, assignment in enumerate(chosen):
            label = assignment['label']
            save(out / 'progress.json', dict(stage='cuda-replay', label=label, completed=processed))
            target = out / ('processing-' + label)
            run('processing-' + label, [sys.executable, repo / 'scripts/process_registered_combat_pool_cuda.py',
                '--pool', out / 'pool', '--plan-index', index, '--out', target])
            report = read(target / 'report.json')
            assert report['state'] == 'complete' and report['device'] == 'cuda'
            assert not report['training_performed'] and len(report['cases']) == len(assignment['seeds'])
            processed.append(dict(label=label, cases=len(report['cases']), rows=report['rows'],
                                  report=str(target / 'report.json'), sha256=sha(target / 'report.json')))
        for path, digest in pins.items():
            assert sha(path) == digest
        save(out / 'complete.json', dict(state='complete', version='combat_aim_feedback_corpus_v1',
            protocol_sha256=sha(out / 'protocol.json'), pool_sha256=sha(out / 'pool/report.json'),
            processed=processed, native_cases=len(pool['jobs']), cuda_cases=sum(p['cases'] for p in processed),
            training_performed=False, promotion=False, script_sha256=sha(__file__)))
        save(out / 'progress.json', dict(stage='complete', processed=processed))
        print(out / 'complete.json')
    except BaseException as error:
        save(out / 'progress.json', dict(stage='failed', error=str(error)))
        raise


if __name__ == '__main__':
    main()
