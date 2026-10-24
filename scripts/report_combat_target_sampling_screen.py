"""Apply the frozen gate to the target-temperature screen and retain action diagnostics."""
import argparse
from pathlib import Path

from evaluate_registered_combat_models import read, save, sha
from report_combat_movement_variance_confirmation import paired_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('evaluation', 'spec', 'quality', 'damage', 'out'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    assert not args.out.exists()
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        assert expected is None or digest == expected, str(path)
        pins[str(path)] = digest
        return path

    pin(__file__)
    root = args.evaluation.resolve()
    closed_path = pin(root / 'complete.json')
    closed = read(closed_path)
    assert closed['state'] == 'complete' and closed['native_cases'] == 24 and closed['cuda_cases'] == 16
    spec = read(pin(args.spec, closed['spec_sha256']))
    gate = spec['expansion_gate']
    assert gate == dict(additional_goals_over_source=2, goals_at_least_rules=True, deaths_no_more_than_source=True,
                        received_numerator=11, received_denominator=10, automatic_promotion=False)
    pool = read(pin(root / 'pool/report.json', closed['pool_sha256']))
    assert pool['state'] == 'complete' and pool['source_unchanged'] and pool['usable_captures'] == 24
    source = next(v for v in spec['variants'] if v['label'] == 'source')
    calibrated = next(v for v in spec['variants'] if v['label'] == 'calibrated')
    original = read(pin(Path(source['update']) / 'weights.json'))
    candidate = read(pin(Path(calibrated['policy_artifact']) / 'weights.json'))
    header = candidate.pop('target_sampling')
    assert header == dict(version='combat_target_temperature_v1', temperature=.5) and candidate == original
    audits = []
    for directory in (args.quality, args.damage):
        receipt = read(pin(directory / 'complete.json'))
        data = read(pin(directory / 'report.json', receipt['report_sha256']))
        assert data['inputs'][str(closed_path)] == sha(closed_path)
        audits.append(data)
    assert set(audits[0]['variants']) == {'source', 'calibrated'} and len(audits[1]['cases']) == 16
    totals, paired = {}, {}
    for comparison in closed['comparisons']:
        data = read(pin(root / ('quality-' + comparison['name'] + '.json'), comparison['sha256']))
        assert comparison['after'] == 'calibrated' and len(data['pairs']) == 8
        assert 'calibrated' not in totals or totals['calibrated'] == data['totals']['after']
        totals['calibrated'] = data['totals']['after']
        totals[comparison['before']] = data['totals']['before']
        paired[comparison['name']] = paired_summary(data['pairs'])
    assert set(totals) == {'source', 'calibrated', 'rules'}
    ownership = read(pin(root / 'ownership-calibrated.json'))
    assert ownership['state'] == 'complete' and ownership['pool_sha256'] == closed['pool_sha256']
    assert len(ownership['cases']) == 8
    after, before, rules = (totals[k] for k in ('calibrated', 'source', 'rules'))
    assert all(type(t[k]) is int for t in totals.values() for k in ('goal', 'death', 'received'))
    checks = dict(goals_at_least_source_plus_two=after['goal'] >= before['goal'] + 2,
                  goals_at_least_rules=after['goal'] >= rules['goal'], deaths_no_more_than_source=after['death'] <= before['death'],
                  received_within_110_percent_source=10 * after['received'] <= 11 * before['received'],
                  observed_combat_owned_by_model=all(ownership['totals'][k] == 0 for k in
                    ('visible_enemy_rules_frames', 'visible_enemy_rules_attack_frames', 'visible_rules_missing_applied_frames', 'unknown_observation_rows'))
                    and not any(c['goal'] and c['rules_attack_events'] for c in ownership['cases']))
    for path, digest in pins.items():
        assert sha(path) == digest, path
    args.out.mkdir(parents=True)
    save(args.out / 'report.json', dict(state='complete', version='combat_target_sampling_screen_decision_v1',
         gate_passed=all(checks.values()), checks=checks, totals=totals, paired=paired, actions=audits[0]['variants'],
         damage_distribution=audits[1]['cases'], inputs=pins, neural_execution=False, training_performed=False,
         promotion=False, scope='Fixed eight-condition development screen. A pass permits a separately frozen '
         '24-condition comparison, not production promotion or general superiority. Switching and native HP are diagnostics.'))
    save(args.out / 'complete.json', dict(state='complete', report_sha256=sha(args.out / 'report.json')))
    print(dict(gate_passed=all(checks.values()), checks=checks, totals=totals))


if __name__ == '__main__':
    main()
