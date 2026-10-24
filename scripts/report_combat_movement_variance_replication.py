"""Combine the fixed two-seed movement-variance comparison without mixing discovery data."""
import argparse
import collections
from pathlib import Path

import report_combat_movement_variance_confirmation as summary
from evaluate_registered_combat_models import read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluations', nargs=2, type=Path, required=True)
    parser.add_argument('--quality-audits', nargs=2, type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), 'Fresh output required'
    pins, stages, combined, seen = {}, [], collections.defaultdict(list), set()
    bindings = None

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        assert expected is None or digest == expected, str(path)
        assert str(path) not in pins or pins[str(path)] == digest
        pins[str(path)] = digest
        return path

    pin(__file__)
    pin(summary.__file__)
    environment = collections.defaultdict(set)
    for root, quality in zip(args.evaluations, args.quality_audits):
        root, quality = root.resolve(), quality.resolve()
        closed_path = pin(root / 'complete.json')
        closed = read(closed_path)
        assert closed['state'] == 'complete' and closed['native_cases'] == 72 and closed['cuda_cases'] == 48
        assert not closed['training_performed'] and not closed['promotion']
        fixture = read(pin(root / 'fixture-proof.json'))
        assert fixture['spec_sha256'] == closed['spec_sha256']
        current = dict(models=fixture['models'], registry_sha256=fixture['registry_sha256'])
        assert bindings is None or current == bindings, 'Policies or registry changed between stages'
        bindings = current
        pool = read(pin(root / 'pool/report.json', closed['pool_sha256']))
        assert pool['state'] == 'complete' and pool['source_unchanged'] and pool['usable_captures'] == 72
        for job in pool['jobs']:
            manifest = read(pin(Path(job['root']) / 'manifest.json'))
            for key in ('client_sha256', 'native_source_fingerprint', 'exporter_sha256', 'reward_config_sha256',
                        'timescale', 'skill', 'release_game_frame', 'fixed_world_hold', 'monster_no_infighting',
                        'standard_monster_spawn_height', 'post_frame_rng_reset', 'synchronous', 'stop_on_goal'):
                environment[key].add(manifest[key])
        seal = read(pin(quality / 'complete.json'))
        assert seal['state'] == 'complete'
        audit = read(pin(quality / 'report.json', seal['report_sha256']))
        assert audit['inputs'][str(closed_path)] == sha(closed_path)
        assert set(audit['variants']) == {'source', 'noise-only'}
        assert all(v['counts']['cases'] == 24 for v in audit['variants'].values())
        stage = dict(evaluation=str(root), seed_offset=fixture['seed_offset'], actions=audit['variants'], comparisons={})
        for comparison in closed['comparisons']:
            name = comparison['name']
            report = read(pin(root / ('quality-' + name + '.json'), comparison['sha256']))
            pairs = report['pairs']
            assert len(pairs) == 24
            ids = {(name, p['episode'], p['seed']) for p in pairs}
            assert len(ids) == 24 and not seen.intersection(ids), 'Repeated cases across stages'
            seen.update(ids)
            combined[name].extend(pairs)
            stage['comparisons'][name] = summary.paired_summary(pairs)
        stages.append(stage)
    assert all(len(values) == 1 for values in environment.values()), 'Native environment changed across stages'
    results = {}
    for name, pairs in combined.items():
        assert len(pairs) == 48
        by_episode = collections.Counter(p['episode'] for p in pairs)
        assert len(by_episode) == 24 and set(by_episode.values()) == {2}
        groups = collections.defaultdict(list)
        for pair in pairs:
            episode = pair['episode']
            groups['map:' + pair['map']].append(pair)
            groups['site:' + ('01' if '-site-01-' in episode else '02')].append(pair)
            groups['loadout:' + ('recovery' if episode.endswith('-recovery') else 'ordinary')].append(pair)
            groups['enemies:' + ('3' if '-gunner-' in episode else '2')].append(pair)
        assert all(len(items) == (12 if key.startswith('map:') else 24) for key, items in groups.items())
        results[name] = dict(paired=summary.paired_summary(pairs),
                             strata={key: summary.paired_summary(items) for key, items in sorted(groups.items())})
    assert set(results) == {'noise-source', 'noise-rules'}
    for path, digest in pins.items():
        assert sha(path) == digest, path
    args.out.mkdir(parents=True)
    save(args.out / 'report.json', dict(
        state='complete', version='combat_movement_variance_two_seed_summary_v1', stages=stages,
        combined=results, bindings=bindings, inputs=pins, training_performed=False, promotion=False,
        scope='Two fresh seeds for each of 24 existing recipes, with fixed policies and native environment. '
              'Discovery excluded; strata and sign tests descriptive. No campaign/coop or general superiority claim.'))
    save(args.out / 'complete.json', dict(state='complete', report_sha256=sha(args.out / 'report.json')))
    for label, result in results.items():
        print(label, result['paired'])


if __name__ == '__main__':
    main()
