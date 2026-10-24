"""Close the preregistered zero/cost PPO screen using task and native action outcomes."""
import argparse
import collections
from pathlib import Path

from evaluate_registered_combat_models import read, save, sha
import report_combat_movement_variance_confirmation as paired_module
from report_combat_movement_variance_confirmation import paired_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('workflow', 'spec', 'quality', 'damage', 'out'):
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

    def sealed(directory):
        receipt = read(pin(directory / 'complete.json'))
        return receipt, read(pin(directory / 'report.json', receipt['report_sha256']))

    pin(__file__)
    pin(paired_module.__file__)
    workflow = read(pin(args.workflow / 'complete.json'))
    assert workflow['native_training_cases'] == workflow['cuda_training_cases'] == 64
    spec = read(pin(args.spec))
    protocol = read(pin(spec['experiment_protocol'], spec['experiment_protocol_sha256']))
    assert workflow['protocol_sha256'] == spec['experiment_protocol_sha256']
    assert protocol['version'] == 'combat_blocked_ground_ppo_v1' and protocol['rounds'] == 2
    assert protocol['cases_per_update'] == 16 and len(protocol['episodes']) == 8
    expected_gate = dict(additional_goals_over_zero=2, goals_at_least_parent=True, goals_at_least_rules=True,
        deaths_no_more_than=['zero', 'rules'], received_compare_to=['zero', 'rules'], received_numerator=11,
        received_denominator=10, blocked_request_fraction_less_than_zero=True, standing_fraction_max_increase=.05,
        automatic_promotion=False)
    assert spec['expansion_gate'] == expected_gate
    rounds, all_seeds, paired_seeds, rollouts = [], {}, {}, set()
    for label in ('zero', 'cost'):
        all_seeds[label] = set()
        previous = protocol['initializer_weights_sha256']
        binding = protocol['branches'][label]
        for number in (1, 2):
            root = args.workflow / label / f'round-{number:02d}'
            seal, update = sealed(root / 'update')
            assert update['version'] == protocol['version'] and update['branch'] == label and update['round'] == number
            assert update['device'] == 'cuda' and update['behavior_sha256'] == previous
            assert update['experiment_protocol_sha256'] == sha(spec['experiment_protocol'])
            assert update['checkpoint_optimizer_roundtrip_exact'] and update['rng_preserved']
            assert not update['quality_verified'] and update['verified_native_cases'] == 16
            pin(root / 'update/weights.json', seal['weights_sha256'])
            pin(root / 'update/checkpoint.pt', seal['checkpoint_sha256'])
            closed = read(pin(root / 'corpus/complete.json'))
            assert closed['native_cases'] == closed['cuda_cases'] == 16
            collection = read(pin(root / 'corpus/protocol.json', closed['protocol_sha256']))
            assert collection['train_only'] and len(collection['assignments']) == 1
            assignment = collection['assignments'][0]
            assert assignment['split'] == 'train' and len(set(assignment['seeds'])) == 16
            assert not all_seeds[label].intersection(assignment['seeds'])
            all_seeds[label].update(assignment['seeds'])
            paired_seeds[label, number] = set(assignment['seeds'])
            plan = read(pin(assignment['plan'], assignment['plan_sha256']))
            assert plan['model_sha256'] == previous
            assert {task['episode']['id'] for task in plan['tasks']} == set(protocol['episodes'])
            assert all(task['split'] == 'train' and len(task['seeds']) == 2 for task in plan['tasks'])
            assert collection['ppo_training_objective']['reward_sha256'] == binding['reward_sha256']
            meta = read(pin(root / 'corpus/processing-train/merged/report.json'))
            assert meta['model_sha256'] == previous and meta['rollout_sha256'] == update['rollout_sha256']
            assert meta['rollout_sha256'] not in rollouts
            rollouts.add(meta['rollout_sha256'])
            _, credit = sealed(root / 'native-credit')
            _, advantage = sealed(root / 'advantage-credit')
            assert credit['totals']['rows'] == advantage['rows'] == update['rows']
            assert not credit['unassigned_native_effects']
            if label == 'cost':
                matched = read(pin(root / 'corpus/paired-train-proof.json'))
                assert matched['state'] == 'matched' and matched['own_policy'] and not matched['reused_trajectories']
            rounds.append(dict(branch=label, round=number, rows=update['rows'], actor_steps=update['base_steps'],
                critic_steps=update['critic_steps'], full_kl=update['full_approx_kl'],
                blocked_ground_requests=credit['totals']['blocked_ground_requests'],
                accepted_step_lr_scale_sum=sum(.5**trial['retry'] for trial in update['trials'] if trial['accepted']),
                weights_sha256=seal['weights_sha256']))
            previous = seal['weights_sha256']
        assert previous == workflow['final_weights'][label]
    assert len(all_seeds['zero']) == len(all_seeds['cost']) == 32
    assert all(paired_seeds['zero', number] == paired_seeds['cost', number] for number in (1, 2))
    evaluation = args.workflow / 'evaluation'
    evaluation_receipt = pin(evaluation / 'complete.json', workflow['evaluation_seal_sha256'])
    closed = read(evaluation_receipt)
    assert closed['spec_sha256'] == sha(args.spec) and closed['native_cases'] == 32 and closed['cuda_cases'] == 24
    fixture = read(pin(evaluation / 'fixture-proof.json'))
    for label in ('zero', 'cost'):
        assert next(model['model_sha256'] for model in fixture['models'] if model['label'] == label) == workflow['final_weights'][label]
    _, quality = sealed(args.quality)
    _, damage = sealed(args.damage)
    for audit in (quality, damage):
        assert audit['inputs'][str(evaluation_receipt)] == sha(evaluation_receipt)
    assert set(quality['variants']) == {'parent', 'zero', 'cost'} and len(damage['cases']) == 24
    totals, comparisons = {}, {}
    for item in closed['comparisons']:
        data = read(pin(evaluation / ('quality-' + item['name'] + '.json'), item['sha256']))
        assert item['after'] == 'cost' and len(data['pairs']) == 8
        assert 'cost' not in totals or totals['cost'] == data['totals']['after']
        totals['cost'], totals[item['before']] = data['totals']['after'], data['totals']['before']
        groups = collections.defaultdict(list)
        for case in data['pairs']:
            assert case['seed'] not in all_seeds['zero'] | all_seeds['cost']
            groups['map:' + case['map']].append(case)
            groups['recovery' if case['episode'].endswith('-recovery') else 'ordinary'].append(case)
            groups['enemies:3' if '-gunner-' in case['episode'] else 'enemies:2'].append(case)
        comparisons[item['name']] = dict(paired=paired_summary(data['pairs']),
            strata={name: paired_summary(cases) for name, cases in groups.items()})
    assert set(totals) == {'parent', 'zero', 'cost', 'rules'}
    cost, zero = totals['cost'], totals['zero']
    actions = quality['variants']
    c, z = actions['cost']['counts'], actions['zero']['counts']
    assert all(row['battles'] == 8 for row in totals.values())
    assert c['large_requested_movement_frames'] > 0 and z['large_requested_movement_frames'] > 0
    assert c['grounded_visible_frames'] > 0 and z['grounded_visible_frames'] > 0
    ownership = read(pin(evaluation / 'ownership-cost.json'))
    assert ownership['pool_sha256'] == closed['pool_sha256'] and len(ownership['cases']) == 8
    checks = dict(goals_over_zero_by_two=cost['goal'] >= zero['goal'] + 2,
        goals_at_least_parent=cost['goal'] >= totals['parent']['goal'],
        goals_at_least_rules=cost['goal'] >= totals['rules']['goal'],
        reduced_request_fraction_lower=c['request_reduced_near_zero_frames'] * z['large_requested_movement_frames']
            < z['request_reduced_near_zero_frames'] * c['large_requested_movement_frames'],
        standing_increase_within_five_points=20 * (c['standing_grounded_visible_frames'] * z['grounded_visible_frames']
            - z['standing_grounded_visible_frames'] * c['grounded_visible_frames'])
            <= c['grounded_visible_frames'] * z['grounded_visible_frames'],
        observed_combat_owned_by_model=all(ownership['totals'][key] == 0 for key in
          ('visible_enemy_rules_frames', 'visible_enemy_rules_attack_frames', 'visible_rules_missing_applied_frames', 'unknown_observation_rows'))
          and not any(case['goal'] and case['rules_attack_events'] for case in ownership['cases']))
    for label in ('zero', 'rules'):
        checks['deaths_no_more_than_' + label] = cost['death'] <= totals[label]['death']
        checks['received_within_110_percent_' + label] = 10 * cost['received'] <= 11 * totals[label]['received']
    assert all(sha(path) == digest for path, digest in pins.items())
    args.out.mkdir(parents=True)
    save(args.out / 'report.json', dict(state='complete', version='combat_blocked_ground_ppo_result_v1', gate_passed=all(checks.values()), checks=checks,
        totals=totals, rounds=rounds, comparisons=comparisons, actions=actions, inputs=pins,
        neural_execution=False, new_games=False, training_performed=False, promotion=False,
        scope='Fixed final second updates, eight validation recipes. Small screen; action rates descriptive over different trajectories. '
              'No general-superiority, campaign, human-coop or automatic-promotion claim.'))
    lines = ['# Blocked ground request PPO screen', '', '| Variant | Goals / 8 | Deaths | Kills | Received HP |',
             '|---|---:|---:|---:|---:|']
    for label in ('parent', 'zero', 'cost', 'rules'):
        t = totals[label]
        lines.append(f"| {label} | {t['goal']} | {t['death']} | {t['kills']} | {t['received']} |")
    lines.extend(['', 'Expansion gate: ' + str(all(checks.values())) + '. Promotion: false.'])
    (args.out / 'results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    save(args.out / 'complete.json', dict(state='complete', report_sha256=sha(args.out / 'report.json'),
                                         markdown_sha256=sha(args.out / 'results.md')))
    print(dict(gate_passed=all(checks.values()), totals=totals, checks=checks))


if __name__ == '__main__':
    main()
