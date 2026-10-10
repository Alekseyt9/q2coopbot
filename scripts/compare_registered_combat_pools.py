"""Compare closed paired registry captures using native first-life outcomes."""
import argparse
import collections
import pathlib

from process_combat_architecture_pool import read, save, sha


def load(root, plan_index=None):
    report = read(root/'report.json')
    assert report['state'] == 'complete' and report['source_unchanged']
    assert report['usable_captures'] == len(report['jobs'])
    cases = {}
    if plan_index is not None:
        assert 0 <= plan_index < len(report['plans'])
    for job in report['jobs']:
        if plan_index is not None and job['plan_index'] != plan_index:
            continue
        assert not job['error'] and job['mode'] == 'learned'
        path = pathlib.Path(job['root'])
        capture, manifest = read(path/'report.json'), read(path/'manifest.json')
        assert capture['capture_complete'] and capture['provenance_valid']
        assert len(capture['results']) == 1
        row = capture['results'][0]
        assert row['capture_valid'] and row['dispatch_valid'] and row['seed_confirmed']
        fixture = row['generated_fixture']
        assert fixture['split'] in ('validation', 'test', 'confirmation')
        assert fixture['engine_seed'] == row['seed'] == job['seed']
        binding = report['plans'][job['plan_index']]
        assert sha(binding['path']) == binding['sha256']
        plan = read(binding['path'])
        assert manifest['model_weights_sha256'].lower() == plan['model_sha256'] == sha(plan['model_path'])
        key = fixture['episode_id'], row['seed']
        assert key not in cases
        damage = row['first_life']['damage']
        cases[key] = dict(fixture=fixture, client_sha256=manifest['client_sha256'].lower(),
                          native_source_fingerprint=manifest['native_source_fingerprint'].lower(),
                          exporter_sha256=manifest['exporter_sha256'].lower(),
                          reward_sha256=manifest['reward_config_sha256'].lower(),
                          capture_sha256=sha(path/'report.json'),
                          kills=sum(mod['monster_kills'] for mod in damage['by_mod']),
                          dealt=sum(mod['monster_health_damage'] for mod in damage['by_mod']),
                          received=damage['received_health_damage'],
                          death=bool(row['death_stop'] or row['first_life']['minimum_health'] <= 0),
                          goal=bool(row['goal_stop']), frames=row['actual_game_frames'])
    return cases


def main():
    parser = argparse.ArgumentParser()
    for name in ('before', 'after', 'out'):
        parser.add_argument('--'+name, type=pathlib.Path, required=True)
    parser.add_argument('--before-plan-index', type=int)
    parser.add_argument('--after-plan-index', type=int)
    args = parser.parse_args()
    assert not args.out.exists(), 'Fresh comparison output required'
    before, after = load(args.before.resolve(), args.before_plan_index), load(args.after.resolve(), args.after_plan_index)
    assert before.keys() == after.keys() and before
    totals = {label: collections.Counter() for label in ('before', 'after')}
    maps, pairs = {}, []
    for key in sorted(before):
        left, right = before[key], after[key]
        assert left['fixture'] == right['fixture'], 'Map/seed/geometry/loadout/reset differs'
        for field in ('client_sha256', 'exporter_sha256', 'reward_sha256', 'native_source_fingerprint'):
            assert left[field] == right[field], 'Frozen environment differs: '+field
        name = left['fixture']['map']
        maps.setdefault(name, {label: collections.Counter() for label in totals})
        metrics = {}
        for label, case in (('before', left), ('after', right)):
            metrics[label] = {k: case[k] for k in ('kills', 'dealt', 'received', 'death', 'goal', 'frames')}
            for target in (totals[label], maps[name][label]):
                target['battles'] += 1
                for k, v in metrics[label].items():
                    target[k] += v
        pairs.append(dict(episode=key[0], seed=key[1], map=name, metrics=metrics,
                          before_capture_sha256=left['capture_sha256'], after_capture_sha256=right['capture_sha256']))
    result = dict(version='registered_combat_paired_quality_v1', state='complete',
                  before_pool_sha256=sha(args.before/'report.json'), after_pool_sha256=sha(args.after/'report.json'),
                  before_plan_index=args.before_plan_index, after_plan_index=args.after_plan_index,
                  totals=totals, maps=maps, pairs=pairs, promotion='Not assessed',
                  scope='Frozen client/exporter/reward and native source fingerprint, identical fixture and seeds. Native first-life outcomes; command counts are not accuracy. Small held-out site comparison, not general superiority.')
    save(args.out, result)
    print(dict(totals=totals, maps=maps), flush=True)


if __name__ == '__main__':
    main()
