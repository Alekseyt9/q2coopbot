"""Apply the frozen development-screen gate to final second-update combat results."""
import argparse
from pathlib import Path

from evaluate_registered_combat_models import read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('evaluation', 'quality-audit', 'spec', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), 'Fresh output required'
    root, quality = args.evaluation.resolve(), args.quality_audit.resolve()
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        assert expected is None or digest == expected, str(path)
        assert str(path) not in pins or pins[str(path)] == digest
        pins[str(path)] = digest
        return path

    pin(__file__)
    closed_path = pin(root / 'complete.json')
    closed = read(closed_path)
    assert closed['state'] == 'complete' and closed['native_cases'] == 32 and closed['cuda_cases'] == 24
    assert not closed['training_performed'] and not closed['promotion']
    spec = read(pin(args.spec, closed['spec_sha256']))
    experiment = read(pin(spec['experiment_protocol'], spec['experiment_protocol_sha256']))
    assert experiment['rounds'] == 2 and experiment['evaluation']['cases_per_variant'] == 8
    assert experiment['evaluation']['expansion_gate'] == (
        'trained goals>=noise-only+2 and >=rules; deaths<=noise-only; received<=1.1*noise-only. '
        'This is a development screen, not a production promotion.')
    fixture = read(pin(root / 'fixture-proof.json'))
    assert fixture['spec_sha256'] == closed['spec_sha256']
    pool = read(pin(root / 'pool/report.json', closed['pool_sha256']))
    assert pool['state'] == 'complete' and pool['source_unchanged'] and pool['usable_captures'] == 32
    receipt = read(pin(quality / 'complete.json'))
    audit = read(pin(quality / 'report.json', receipt['report_sha256']))
    assert audit['inputs'][str(closed_path)] == sha(closed_path)
    assert set(audit['variants']) == {'source', 'noise-only', 'trained'}
    trained_folder = Path(next(v['update'] for v in spec['variants'] if v['label'] == 'trained'))
    trained_seal = read(pin(trained_folder / 'complete.json'))
    trained_report = read(pin(trained_folder / 'report.json', trained_seal['report_sha256']))
    assert trained_report['device'] == 'cuda' and trained_report['round'] == 2
    assert trained_report['experiment_protocol_sha256'] == spec['experiment_protocol_sha256']
    assert trained_report['training_performed'] and not trained_report['quality_verified']
    assert sha(pin(trained_folder / 'weights.json', trained_seal['weights_sha256'])) == trained_report['weights_sha256']
    assert next(m['model_sha256'] for m in fixture['models'] if m['label'] == 'trained') == trained_seal['weights_sha256']
    totals, comparisons = {}, {}
    for comparison in closed['comparisons']:
        assert comparison['after'] == 'trained'
        data = read(pin(root / ('quality-' + comparison['name'] + '.json'), comparison['sha256']))
        assert len(data['pairs']) == 8
        assert 'trained' not in totals or totals['trained'] == data['totals']['after']
        totals['trained'] = data['totals']['after']
        totals[comparison['before']] = data['totals']['before']
        comparisons[comparison['name']] = dict(maps=data['maps'], pairs=data['pairs'])
    assert set(totals) == {'source', 'noise-only', 'trained', 'rules'}
    assert all(t['battles'] == 8 and all(type(t[k]) is int for k in ('goal', 'death', 'received')) for t in totals.values())
    ownership = {}
    for label in ('source', 'noise-only', 'trained'):
        data = read(pin(root / ('ownership-' + label + '.json')))
        assert data['state'] == 'complete' and data['pool_sha256'] == closed['pool_sha256']
        assert len(data['cases']) == 8
        ownership[label] = dict(totals=data['totals'],
                               goal_cases_with_rules_attack_commands=sum(bool(c['goal'] and c['rules_attack_events']) for c in data['cases']))
    trained, noise, rules = (totals[k] for k in ('trained', 'noise-only', 'rules'))
    counts = ownership['trained']['totals']
    checks = dict(
        trained_goals_at_least_noise_plus_two=trained['goal'] >= noise['goal'] + 2,
        trained_goals_at_least_rules=trained['goal'] >= rules['goal'],
        trained_deaths_no_more_than_noise=trained['death'] <= noise['death'],
        trained_received_no_more_than_110_percent_noise=10 * trained['received'] <= 11 * noise['received'],
        observed_combat_owned_by_model=all(counts[k] == 0 for k in ('visible_enemy_rules_frames',
            'visible_enemy_rules_attack_frames', 'visible_rules_missing_applied_frames', 'unknown_observation_rows'))
            and ownership['trained']['goal_cases_with_rules_attack_commands'] == 0)
    for path, digest in pins.items():
        assert sha(path) == digest, path
    args.out.mkdir(parents=True)
    report = dict(state='complete', version='combat_movement_variance_ppo_screen_decision_v1',
                  gate_passed=all(checks.values()), checks=checks, totals=totals, actions=audit['variants'],
                  ownership=ownership, comparisons=comparisons, inputs=pins, neural_execution=False,
                  new_games=False, training_performed=False, promotion=False,
                  scope='Frozen eight-condition development gate for final update only. Passing permits a fresh '
                        '24-condition comparison; it does not prove superiority or campaign/coop acceptance. '
                        'Shot/movement rates describe unequal closed-loop states. Received-damage gate uses exact integer arithmetic.')
    save(args.out / 'report.json', report)
    save(args.out / 'complete.json', dict(state='complete', report_sha256=sha(args.out / 'report.json')))
    print(dict(gate_passed=report['gate_passed'], checks=checks, totals=totals))


if __name__ == '__main__':
    main()
