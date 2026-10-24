"""Join sealed confirmation outcomes and native actions without running games or models."""
import argparse
import collections
from pathlib import Path

from evaluate_registered_combat_models import read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('result', 'evaluation', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        assert expected is None or digest == expected, str(path)
        pins[str(path)] = digest
        return path

    def sealed(root):
        receipt = read(pin(root / 'complete.json'))
        assert receipt['state'] == 'complete'
        return read(pin(root / 'report.json', receipt['report_sha256']))

    pin(__file__)
    result = sealed(args.result)
    assert result['version'] == 'combat_trio_independent_confirmation_v1'
    evaluation_seal = pin(args.evaluation / 'complete.json')
    assert result['inputs'][str(evaluation_seal)] == sha(evaluation_seal)
    quality = sealed(args.evaluation / 'native-quality')
    damage = sealed(args.evaluation / 'damage')
    for audit in (quality, damage):
        assert audit['inputs'][str(evaluation_seal)] == sha(evaluation_seal)
    qindex = {(c['label'], c['map'], c['seed']): c for c in quality['cases']}
    dindex = {(c['label'], c['map'], c['seed']): c for c in damage['cases']}
    assert len(qindex) == len(dindex) == 72 and qindex.keys() == dindex.keys()
    outcomes = {}
    comparisons = {}
    for opponent in ('parent', 'full', 'rules'):
        path = pin(args.evaluation / f'quality-decisions-{opponent}.json')
        assert result['inputs'][str(path)] == sha(path)
        data = read(path)
        assert len(data['pairs']) == 24
        discordant = []
        for case in data['pairs']:
            key = case['map'], case['seed']
            before, after = case['metrics']['before'], case['metrics']['after']
            for label, metrics in ((opponent, before), ('decisions', after)):
                fullkey = (label, *key)
                assert fullkey not in outcomes or outcomes[fullkey] == metrics
                outcomes[fullkey] = metrics
            if before['goal'] != after['goal']:
                discordant.append(dict(episode=case['episode'], map=key[0], seed=key[1],
                    change='gained' if after['goal'] else 'lost', before=before, after=after))
        comparisons[opponent] = discordant
    groups = collections.defaultdict(collections.Counter)
    cases = []
    for key, q in sorted(qindex.items()):
        d = dindex[key]
        counts = collections.Counter(q['counts'])
        assert counts['provider_frames'] == d['counts']['provider_frames']
        counts.update({k: v for k, v in d['counts'].items() if k != 'provider_frames'})
        metrics = outcomes[key]
        assert counts['alive_goals'] == int(metrics['goal'])
        assert counts['deaths'] == int(metrics['death'])
        episode = d['episode']
        strata = ['all', 'enemies:3' if '-gunner-' in episode else 'enemies:2',
                  'recovery' if episode.endswith('-recovery') else 'ordinary', 'map:' + key[1],
                  'won' if metrics['goal'] else 'not_won']
        for condition in strata:
            groups[key[0], condition].update(counts)
        cases.append(dict(label=key[0], map=key[1], seed=key[2], episode=episode,
            outcome=metrics, counts=dict(counts), native_damage_by_enemy=d['native_damage_by_enemy']))

    def rates(counts):
        def fraction(n, denominator):
            return counts.get(n, 0) / counts[denominator] if counts.get(denominator) else None
        return dict(standing=fraction('standing_grounded_visible_frames', 'grounded_visible_frames'),
            large_yaw=fraction('large_yaw_frames', 'provider_frames'),
            attack_without_target=fraction('attack_without_target_frames', 'provider_frames'),
            reduced_movement=fraction('request_reduced_near_zero_frames', 'large_requested_movement_frames'),
            mg_live_damage=fraction('mg_live_monster_damage', 'mg_shots'),
            previous_visible_switch=fraction('switched_while_previous_target_still_visible', 'previous_target_still_visible_frames'))

    rows = [dict(label=label, condition=condition, counts=dict(counts), rates=rates(counts))
            for (label, condition), counts in sorted(groups.items())]
    for row in rows:
        if row['condition'] == 'all':
            assert row['counts']['cases'] == 24
            assert row['counts']['alive_goals'] == result['totals'][row['label']]['goal']
    assert all(sha(path) == digest for path, digest in pins.items())
    args.out.mkdir(parents=True)
    save(args.out / 'report.json', dict(state='complete', version='combat_trio_confirmation_diagnostics_v1',
        rows=rows, cases=cases, discordant=comparisons, inputs=pins,
        neural_execution=False, new_games=False, training_performed=False, promotion=False,
        scope='Post-hoc description of the closed confirmation. Rates have different trajectories and durations; '
              'they do not prove causality. Last-hit health is not final health. No evaluation trajectories enter fitting.'))
    lines = ['# Confirmation diagnostics', '',
        '| Policy | Condition | Won | Deaths | Standing % | Large yaw % | MG damage % | Switch % |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    def pct(value):
        return '-' if value is None else f'{100 * value:.2f}'
    for row in rows:
        if row['condition'] not in ('all', 'enemies:2', 'enemies:3', 'ordinary', 'recovery'):
            continue
        c, r = row['counts'], row['rates']
        lines.append(f"| {row['label']} | {row['condition']} | {c['alive_goals']}/{c['cases']} | {c['deaths']} | "
                     f"{pct(r['standing'])} | {pct(r['large_yaw'])} | {pct(r['mg_live_damage'])} | {pct(r['previous_visible_switch'])} |")
    for label, discordant in comparisons.items():
        lines.extend(['', f'## Decisions versus {label}', '', '| Change | Episode | Seed | Goals before/after | Deaths before/after |',
                      '|---|---|---:|---|---|'])
        for c in discordant:
            b, a = c['before'], c['after']
            lines.append(f"| {c['change']} | {c['episode']} | {c['seed']} | {int(b['goal'])}/{int(a['goal'])} | {int(b['death'])}/{int(a['death'])} |")
    lines.extend(['', 'Descriptive only: unequal closed-loop states; no causal or general-superiority claim.'])
    (args.out / 'results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    save(args.out / 'complete.json', dict(state='complete', report_sha256=sha(args.out / 'report.json'),
        markdown_sha256=sha(args.out / 'results.md')))
    print(args.out / 'results.md')


if __name__ == '__main__':
    main()
