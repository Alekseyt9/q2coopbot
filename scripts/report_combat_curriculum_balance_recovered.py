"""Close the preregistered balanced-versus-trio curriculum screen."""
import argparse
from pathlib import Path

from evaluate_registered_combat_models import read, save, sha
from report_combat_movement_variance_confirmation import paired_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('workflow', 'protocol', 'spec', 'evaluation', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve(); digest = sha(path)
        assert expected is None or digest == expected, str(path)
        pins[str(path)] = digest
        return path

    def sealed(root):
        receipt = read(pin(root / 'complete.json'))
        assert receipt['state'] == 'complete'
        return receipt, read(pin(root / 'report.json', receipt['report_sha256']))

    pin(__file__)
    study, spec = read(pin(args.protocol)), read(pin(args.spec))
    assert study['version'] == 'combat_curriculum_balance_v1'
    assert spec['experiment_protocol_sha256'] == sha(args.protocol)
    gate = dict(additional_goals_over_trio=2, goals_at_least_source=True, goals_at_least_rules=True,
        duo_goals_at_least_source_and_trio=True, trio_goals_at_least_source=True,
        deaths_no_more_than=['trio', 'rules'], received_compare_to=['trio', 'rules'],
        received_numerator=11, received_denominator=10, automatic_promotion=False)
    assert spec['expansion_gate'] == gate
    evaluation = args.evaluation
    recovery = read(pin(evaluation / 'recovery-native-complete.json'))
    assert recovery['version'] == 'combat_goal_pool_reprocessing_v1' and recovery['new_games'] == 0
    closed = read(pin(evaluation / 'complete.json'))
    assert closed['state'] == 'complete' and closed['native_cases'] == 32 and closed['cuda_cases'] == 24
    assert closed['spec_sha256'] == sha(args.spec)
    counts = read(pin(args.workflow / 'matched-update-counts.json'))
    assert counts['state'] == 'matched' and counts['new_updates_per_arm'] == 1 and counts['cases_per_arm'] == 24
    ledger = []
    training_seeds = set()
    for arm in study['arms']:
        root = args.workflow / arm['label']
        seal, report = sealed(root / 'update')
        pin(root / 'update/weights.json', seal['weights_sha256'])
        pin(root / 'update/checkpoint.pt', seal['checkpoint_sha256'])
        assert report['device'] == 'cuda' and report['behavior_sha256'] == study['source_weights_sha256']
        assert report['scope_updates'] == 5 and report['frozen_coordinates_exact']
        assert report['checkpoint_roundtrip_exact'] and report['disposable_resume_adam_step_exact']
        corpus = read(pin(root / 'corpus/complete.json'))
        assert corpus['native_cases'] == corpus['cuda_cases'] == 24
        collection = read(pin(root / 'corpus/protocol.json', corpus['protocol_sha256']))
        assert collection['train_only'] and len(collection['assignments']) == 1
        assignment = collection['assignments'][0]
        assert assignment['split'] == 'train' and len(set(assignment['seeds'])) == 24
        assert not training_seeds.intersection(assignment['seeds'])
        training_seeds.update(assignment['seeds'])
        plan = read(pin(assignment['plan'], assignment['plan_sha256']))
        assert plan['model_sha256'] == study['source_weights_sha256']
        contract = read(pin(arm['contract']))
        assert {task['episode']['id'] for task in plan['tasks']} == set(contract['episodes'])
        ledger.append(dict(label=arm['label'], rows=report['rows'], actor_steps=report['actor_steps'],
            critic_steps=report['critic_steps'], weights_sha256=seal['weights_sha256']))
    totals, comparisons, strata = {}, {}, {}
    for item in closed['comparisons']:
        data = read(pin(evaluation / ('quality-' + item['name'] + '.json'), item['sha256']))
        assert item['after'] == 'balanced' and len(data['pairs']) == 8
        assert not training_seeds.intersection(c['seed'] for c in data['pairs'])
        before = item['before']
        totals[before], totals['balanced'] = data['totals']['before'], data['totals']['after']
        comparisons[before] = paired_summary(data['pairs'])
        strata[before] = {}
        for n in (2, 3):
            cases = [c for c in data['pairs'] if (3 if '-gunner-' in c['episode'] else 2) == n]
            assert len(cases) == 4
            strata[before][str(n)] = paired_summary(cases)
    assert set(totals) == {'balanced', 'trio', 'source', 'rules'}
    balanced = totals['balanced']
    ownership = read(pin(evaluation / 'ownership-balanced.json'))
    assert ownership['pool_sha256'] == closed['pool_sha256'] and len(ownership['cases']) == 8
    checks = dict(goals_trio_plus_two=balanced['goal'] >= totals['trio']['goal'] + 2,
        goals_at_least_source=balanced['goal'] >= totals['source']['goal'],
        goals_at_least_rules=balanced['goal'] >= totals['rules']['goal'],
        observed_combat_owned_by_model=all(ownership['totals'][key] == 0 for key in
            ('visible_enemy_rules_frames', 'visible_enemy_rules_attack_frames', 'visible_rules_missing_applied_frames', 'unknown_observation_rows'))
            and not any(c['goal'] and c['rules_attack_events'] for c in ownership['cases']),
        duo_goals_at_least_source_and_trio=all(strata[label]['2']['after_goal'] >= strata[label]['2']['before_goal'] for label in ('source', 'trio')),
        trio_goals_at_least_source=strata['source']['3']['after_goal'] >= strata['source']['3']['before_goal'])
    for label in ('trio', 'rules'):
        checks['deaths_no_more_than_' + label] = balanced['death'] <= totals[label]['death']
        checks['received_within_110_percent_' + label] = 10 * balanced['received'] <= 11 * totals[label]['received']
    _, quality = sealed(evaluation / 'native-quality')
    _, damage = sealed(evaluation / 'damage')
    for audit in (quality, damage):
        assert audit['inputs'][str((evaluation / 'complete.json').resolve())] == sha(evaluation / 'complete.json')
    assert all(sha(path) == digest for path, digest in pins.items())
    args.out.mkdir(parents=True)
    save(args.out / 'report.json', dict(state='complete', version='combat_curriculum_balance_result_recovered_v1',
        totals=totals, comparisons=comparisons, strata=strata, checks=checks, gate_passed=all(checks.values()),
        training=ledger, actions=quality['variants'], inputs=pins, promotion=False, neural_execution=False,
        scope='One own-policy update per arm from identical state,24 games each. TRAIN seeds independent across arms; '
              'held-out evaluation fixtures paired. Fixed endpoints only. No automatic promotion or general-superiority claim.'))
    lines = ['# Curriculum balance screen', '', '| Policy | Goals / 8 | Deaths | Received HP |', '|---|---:|---:|---:|']
    for label in ('source', 'trio', 'balanced', 'rules'):
        t = totals[label]
        lines.append(f"| {label} | {t['goal']} | {t['death']} | {t['received']} |")
    lines.extend(['', 'Gate: ' + str(all(checks.values())) + '. Promotion: false.'])
    (args.out / 'results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    save(args.out / 'complete.json', dict(state='complete', report_sha256=sha(args.out / 'report.json'),
        markdown_sha256=sha(args.out / 'results.md')))
    print(dict(gate_passed=all(checks.values()), totals=totals))


if __name__ == '__main__':
    main()
