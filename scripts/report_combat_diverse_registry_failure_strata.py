"""Summarize sealed native action diagnostics by fixed evaluation conditions."""
import argparse
import collections
import json
from pathlib import Path

from evaluate_registered_combat_models import read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--quality', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    pins = {}

    def pin(path):
        path = Path(path).resolve()
        pins[str(path)] = sha(path)
        return path

    def closed(directory):
        receipt = read(pin(directory / 'complete.json'))
        path = pin(directory / 'report.json')
        assert sha(path) == receipt['report_sha256']
        return read(path)

    pin(__file__)
    result, quality = closed(args.result), closed(args.quality)
    # Tie both inputs to the same sealed evaluation, rather than joining by seed alone.
    evaluation_inputs = {p: h for p, h in result['inputs'].items()
                         if p.endswith('evaluation/complete.json') or p.endswith('evaluation\\complete.json')}
    assert len(evaluation_inputs) == 1
    assert all(quality['inputs'].get(p) == h for p, h in evaluation_inputs.items())
    index = {}
    for case in result['damage_distribution']:
        key = case['label'], case['map'], case['seed']
        assert key not in index
        index[key] = case
    assert len(index) == len(quality['cases']) == 48
    groups = collections.defaultdict(collections.Counter)
    seen = set()
    for case in quality['cases']:
        key = case['label'], case['map'], case['seed']
        assert key not in seen
        seen.add(key)
        damage = index[key]
        episode = damage['episode']
        conditions = ['all', 'recovery' if episode.endswith('-recovery') else 'ordinary',
                      'map:' + case['map'], 'site:01' if '-site-01-' in episode else 'site:02',
                      'enemies:3' if '-gunner-' in episode else 'enemies:2']
        for condition in conditions:
            target = groups[case['label'], condition]
            target.update(case['counts'])
            target.update(damage['counts'])
            # provider_frames occurs in both auditors, so count it only once.
            target['provider_frames'] -= damage['counts']['provider_frames']
    assert seen == set(index)
    rows = []
    for (label, condition), counts in sorted(groups.items()):
        def ratio(numerator, denominator):
            return counts[numerator] / counts[denominator] if counts[denominator] else None
        rows.append(dict(label=label, condition=condition, counts=dict(counts),
            standing_fraction=ratio('standing_grounded_visible_frames', 'grounded_visible_frames'),
            large_yaw_fraction=ratio('large_yaw_frames', 'provider_frames'),
            attack_without_target_fraction=ratio('attack_without_target_frames', 'provider_frames'),
            movement_reduced_fraction=ratio('request_reduced_near_zero_frames', 'large_requested_movement_frames'),
            mg_live_damage_fraction=ratio('mg_live_monster_damage', 'mg_shots'),
            previous_visible_switch_fraction=ratio('switched_while_previous_target_still_visible',
                                                   'previous_target_still_visible_frames')))
    for row in rows:
        if row['condition'] == 'all':
            assert row['counts']['cases'] == 24
            assert row['counts']['provider_frames'] == result['actions'][row['label']]['counts']['provider_frames']
            assert row['counts']['alive_goals'] == result['totals'][row['label']]['goal']
    assert all(sha(path) == digest for path, digest in pins.items())
    args.out.mkdir(parents=True)
    save(args.out / 'report.json', dict(state='complete', rows=rows, inputs=pins,
        neural_execution=False, new_games=False,
        scope='Descriptive closed-loop rates over different trajectories. No causal attribution, model selection or promotion.'))
    save(args.out / 'complete.json', dict(state='complete', report_sha256=sha(args.out / 'report.json')))
    print(json.dumps([r for r in rows if r['condition'] in ('all', 'ordinary', 'recovery')]))


if __name__ == '__main__':
    main()
