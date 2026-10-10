"""Prepare model-independent registered train conditions from sealed diagnostics."""
import argparse
import pathlib
import re
import subprocess

from process_combat_architecture_pool import read, save, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=pathlib.Path, required=True)
    parser.add_argument('--out', type=pathlib.Path, required=True)
    parser.add_argument('--planner', type=pathlib.Path, required=True)
    parser.add_argument('--seed-offset', type=int, default=176)
    args = parser.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[1]
    reference, out, planner = args.reference.resolve(), args.out.resolve(), args.planner.resolve()
    assert not out.exists() and args.seed_offset >= 176
    protocol = read(reference / 'protocol.json')
    behavior = read(reference / 'outcome-behavior.json')
    assert behavior['state'] == 'complete' and behavior['protocol_sha256'] == sha(reference / 'protocol.json')
    for name, digest in behavior['source_sha256'].items():
        assert sha(reference / name) == digest
    template = read(protocol['evaluations'][0]['plan'])
    registry = pathlib.Path(template['registry_path'])
    assert sha(registry) == template['registry_sha256']
    families = [t['episode']['id'] for t in template['tasks']]
    assert len(families) == 20
    groups = ('m1-stochastic-a', 'parent3-stochastic-a')
    sites = {}
    for name in groups:
        entry = next(e for e in protocol['evaluations'] if e['model'] + '-' + e['label'] == name)
        assert [t['episode']['id'] for t in read(entry['plan'])['tasks']] == families
        for row in behavior['groups'][name]['episodes']:
            match = re.fullmatch(r'case-(\d+)-learned', row['case'])
            assert match, ('Unexpected case identity', row['case'])
            index = int(match[1])
            family = families[index]
            if not family.startswith('campaign-'):
                continue
            site = family.rsplit('-', 1)[0]
            sites.setdefault(site, []).append(row)
    ranking = []
    for site, rows in sites.items():
        deaths = [r for r in rows if r['outcome'] == 'death']
        requested = sum(r['counts'].get('provider_requested_movement', 0) for r in deaths)
        stopped = sum(r['counts'].get('provider_movement_stopped', 0) for r in deaths)
        cancellation = stopped / requested if requested else 0
        death_fraction = len(deaths) / len(rows)
        ranking.append(dict(site=site, episodes=len(rows), deaths=len(deaths),
            death_fraction=death_fraction, death_movement_cancellation=cancellation,
            score=death_fraction + cancellation))
    # Keep both maps and both weapons; selection uses development only.
    selected = []
    for map_name in ('base1', 'base2'):
        candidates = sorted((r for r in ranking if f'-{map_name}-' in r['site']),
                            key=lambda r: (-r['score'], r['site']))
        assert len(candidates) == 4
        selected.extend(r['site'] for r in candidates[:2])
    focused = [f for f in families if f.rsplit('-', 1)[0] in selected]
    core = [f for f in families if not f.startswith('campaign-')]
    assert len(focused) == 8 and len(core) == 4
    out.mkdir()
    plans = []
    for name, episodes, count in [('wall-sites', focused, 8), ('mixed-retention', core, 4)]:
        path = out / (name + '-plan.json')
        command = [str(planner), '--registry', str(registry), '--episodes', ','.join(episodes),
            '--split', 'train', '--mode', 'rules', '--count', str(count),
            '--seed-offset', str(args.seed_offset), '--root', str(repo), '--out', str(path),
            '--artifacts', str(out / (name + '-not-dispatched'))]
        with (out / (name + '-compile.log')).open('w', encoding='utf-8') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
            subprocess.run([str(planner), '--verify-plan', str(path)], stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=60)
        plan = read(path)
        assert all(t['split'] == 'train' and t['modes'] == ['rules'] for t in plan['tasks'])
        for task in plan['tasks']:
            assert len(task['seeds']) == count
            assert not set(task['seeds']) & {seed for old in template['tasks'] for seed in old['seeds']}
        plans.append(dict(name=name, path=str(path), sha256=sha(path),
            families=episodes, count=count, conditions=sum(len(t['instances']) for t in plan['tasks'])))
    assert sum(p['conditions'] for p in plans) == 80
    save(out / 'preparation.json', dict(state='static_preflight_complete_native_pending',
        reference_sha256=sha(reference / 'protocol.json'), behavior_sha256=sha(reference / 'outcome-behavior.json'),
        planner_sha256=sha(planner), registry=str(registry), registry_sha256=sha(registry),
        selection_variants=groups, ranking=sorted(ranking, key=lambda r: (-r['score'], r['site'])),
        selected_sites=selected, plans=plans, conditions_per_model=80,
        training_seed_offset=args.seed_offset, training_device='cuda', evaluation_selection_pending=True,
        scope='Draft training mix only:64 registered wall-site conditions plus16 mixed retention. '
        'No new geometry, native dispatch, model selection, training or validation quality. '
        'Development-selected sites are not held-out. Rules mode is static planner preflight only; '
        'recompile learned plans after fresh evaluation selects parent. Current registry/test untouched.'))
    print('Prepared80 model-independent train conditions, native and CUDA training pending', flush=True)


if __name__ == '__main__':
    main()
