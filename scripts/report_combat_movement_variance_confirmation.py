"""Summarize sealed movement-variance comparisons, including paired outcomes and strata."""
import argparse
import collections
import math
from pathlib import Path

from evaluate_registered_combat_models import read, save, sha


def paired_summary(pairs):
    result = collections.Counter(battles=len(pairs))
    for pair in pairs:
        before, after = (pair['metrics'][key] for key in ('before', 'after'))
        result['new_goals'] += bool(after['goal'] and not before['goal'])
        result['lost_goals'] += bool(before['goal'] and not after['goal'])
        for side, metrics in (('before', before), ('after', after)):
            for key in ('goal', 'death', 'kills', 'received', 'dealt'):
                result[side + '_' + key] += metrics[key]
    n = result['new_goals'] + result['lost_goals']
    smaller = min(result['new_goals'], result['lost_goals'])
    # Descriptive exact paired sign test; no claim of a predeclared statistical gate.
    p = min(1., 2 * sum(math.comb(n, k) for k in range(smaller + 1)) / 2**n) if n else 1.
    return dict(result, goal_discordant_pairs=n, descriptive_goal_sign_test_two_sided_p=p)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    assert not args.out.exists(), 'Fresh output required'
    root = args.evaluation.resolve()
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        assert expected is None or digest == expected, str(path)
        pins[str(path)] = digest
        return path

    pin(__file__)
    closed = read(pin(root / 'complete.json'))
    assert closed['state'] == 'complete' and closed['native_cases'] == 72 and closed['cuda_cases'] == 48
    assert not closed['training_performed'] and not closed['promotion']
    fixture = read(pin(root / 'fixture-proof.json'))
    assert fixture['spec_sha256'] == closed['spec_sha256']
    pool = read(pin(root / 'pool/report.json', closed['pool_sha256']))
    assert pool['state'] == 'complete' and pool['source_unchanged'] and pool['usable_captures'] == 72
    ownership = {}
    for label in ('source', 'noise-only'):
        audit = read(pin(root / ('ownership-' + label + '.json')))
        assert audit['state'] == 'complete' and audit['pool_sha256'] == closed['pool_sha256']
        assert len(audit['cases']) == 24
        ownership[label] = dict(
            totals=audit['totals'],
            goal_cases_with_rules_attack_commands=sum(bool(c['goal'] and c['rules_attack_events']) for c in audit['cases']),
            scope=audit['scope'])
    comparisons = {}
    for comparison in closed['comparisons']:
        report = read(pin(root / ('quality-' + comparison['name'] + '.json'), comparison['sha256']))
        pairs = report['pairs']
        assert len(pairs) == 24 and len({(p['episode'], p['seed']) for p in pairs}) == 24
        groups = collections.defaultdict(list)
        for pair in pairs:
            episode = pair['episode']
            groups['map:' + pair['map']].append(pair)
            groups['site:' + ('01' if '-site-01-' in episode else '02')].append(pair)
            groups['loadout:' + ('recovery' if episode.endswith('-recovery') else 'ordinary')].append(pair)
            groups['enemies:' + ('3' if '-gunner-' in episode else '2')].append(pair)
        assert all(len(items) == (6 if key.startswith('map:') else 12) for key, items in groups.items())
        comparisons[comparison['name']] = dict(
            before=comparison['before'], after=comparison['after'], totals=report['totals'],
            paired=paired_summary(pairs), strata={key: paired_summary(items) for key, items in sorted(groups.items())})
    for path, digest in pins.items():
        assert sha(path) == digest, path
    args.out.mkdir(parents=True)
    save(args.out / 'report.json', dict(
        state='complete', version='combat_movement_variance_confirmation_summary_v1',
        comparisons=comparisons, ownership=ownership, inputs=pins, training_performed=False, promotion=False,
        scope='24 paired fresh validation conditions. Strata and sign test are descriptive, not new selection gates. '
              'Existing geometries, one seed per recipe. Not campaign/coop acceptance or general superiority.'))
    save(args.out / 'complete.json', dict(state='complete', report_sha256=sha(args.out / 'report.json')))
    for label, result in comparisons.items():
        print(label, result['paired'])


if __name__ == '__main__':
    main()
