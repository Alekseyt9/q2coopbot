"""Close fresh confirmation of the promising control from a failed body screen."""
import argparse
from collections import defaultdict
from pathlib import Path

from evaluate_registered_combat_models import read, save, sha
from report_combat_movement_variance_confirmation import paired_summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('evaluation', 'protocol', 'spec', 'native-quality', 'out'):
        ap.add_argument('--' + name, type=Path, required=True)
    a = ap.parse_args()
    assert not a.out.exists()
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        assert expected is None or digest == expected, str(path)
        pins[str(path)] = digest
        return path

    protocol = read(pin(a.protocol))
    assert protocol['version'] == 'combat_body_control_confirmation_v1'
    prerequisite = read(pin(protocol['prerequisite_report'], protocol['prerequisite_report_sha256']))
    assert not prerequisite['gate_passed'], 'Do not rewrite failed body hypothesis'
    closed = read(pin(a.evaluation / 'complete.json'))
    assert closed['state'] == 'complete' and closed['native_cases'] == 72 and closed['cuda_cases'] == 48
    assert not closed['training_performed'] and not closed['promotion']
    fixture = read(pin(a.evaluation / 'fixture-proof.json'))
    spec = read(pin(a.spec, closed['spec_sha256']))
    assert fixture['spec_sha256'] == closed['spec_sha256']
    assert spec['experiment_protocol_sha256'] == sha(a.protocol)
    assert spec['expansion_gate'] == protocol['gate'] and spec['split'] == 'confirmation'
    models = {m['label']: m['model_sha256'] for m in fixture['models']}
    for label, endpoint in protocol['endpoints'].items():
        seal = read(pin(Path(endpoint['path']) / 'complete.json', endpoint['seal_sha256']))
        pin(Path(endpoint['path']) / 'weights.json', endpoint['weights_sha256'])
        assert models[label] == seal['weights_sha256'] == endpoint['weights_sha256']
    assert set(models) == {'initial', 'candidate'}
    totals, comparisons = {}, {}
    for item in closed['comparisons']:
        data = read(pin(a.evaluation / ('quality-' + item['name'] + '.json'), item['sha256']))
        assert item['after'] == 'candidate' and len(data['pairs']) == 24
        assert len({(p['episode'], p['seed']) for p in data['pairs']}) == 24
        totals[item['before']] = data['totals']['before']
        if 'candidate' in totals:
            assert totals['candidate'] == data['totals']['after']
        totals['candidate'] = data['totals']['after']
        groups = defaultdict(list)
        for pair in data['pairs']:
            episode = pair['episode']
            groups['map:' + pair['map']].append(pair)
            groups['site:01' if '-site-01-' in episode else 'site:02'].append(pair)
            groups['recovery' if episode.endswith('-recovery') else 'ordinary'].append(pair)
            groups['trio' if '-gunner-' in episode else 'duo'].append(pair)
        assert all(len(rows) == (6 if name.startswith('map:') else 12) for name, rows in groups.items())
        comparisons[item['name']] = dict(paired=paired_summary(data['pairs']),
            strata={name: paired_summary(rows) for name, rows in sorted(groups.items())})
    assert set(totals) == {'initial', 'candidate', 'rules'}
    candidate, gate = totals['candidate'], protocol['gate']
    checks = {f'goals_{label}_plus_{gate["additional_goals_over_" + label]}':
        candidate['goal'] >= totals[label]['goal'] + gate['additional_goals_over_' + label]
        for label in ('initial', 'rules')}
    for label in gate['deaths_no_more_than']:
        checks['deaths_no_more_than_' + label] = candidate['death'] <= totals[label]['death']
    for label in gate['received_compare_to']:
        checks['received_within_limit_' + label] = (gate['received_denominator'] * candidate['received']
            <= gate['received_numerator'] * totals[label]['received'])
    for label in ('initial', 'candidate'):
        ownership = read(pin(a.evaluation / ('ownership-' + label + '.json')))
        assert ownership['state'] == 'complete' and ownership['pool_sha256'] == closed['pool_sha256']
        assert len(ownership['cases']) == 24
        checks['model_owns_visible_combat_' + label] = all(ownership['totals'][key] == 0 for key in
            ('visible_enemy_rules_frames', 'visible_enemy_rules_attack_frames',
             'visible_rules_missing_applied_frames', 'unknown_observation_rows')) and not any(
                 case['goal'] and case['rules_attack_events'] for case in ownership['cases'])
    qseal = read(pin(a.native_quality / 'complete.json'))
    quality = read(pin(a.native_quality / 'report.json', qseal['report_sha256']))
    assert all(sha(path) == digest for path, digest in pins.items())
    a.out.mkdir(parents=True)
    save(a.out / 'report.json', dict(state='complete', version=protocol['version'],
        gate_passed=all(checks.values()), checks=checks, totals=totals, comparisons=comparisons,
        actions=quality['variants'], inputs=pins, training_performed=False, promotion=False,
        selection=protocol['selection'], scope=protocol['scope']))
    lines = ['# Independent control confirmation', '',
        '| Variant | Goals / 24 | Deaths | Kills | Received HP |', '|---|---:|---:|---:|---:|']
    for label in ('initial', 'candidate', 'rules'):
        t = totals[label]
        lines.append(f"| {label} | {t['goal']} | {t['death']} | {t['kills']} | {t['received']} |")
    lines.extend(['', f'Gate: {all(checks.values())}. Promotion: false.'])
    (a.out / 'results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    save(a.out / 'complete.json', dict(state='complete', report_sha256=sha(a.out / 'report.json'),
        markdown_sha256=sha(a.out / 'results.md')))
    print(dict(gate_passed=all(checks.values()), totals=totals, checks=checks))


if __name__ == '__main__':
    main()
