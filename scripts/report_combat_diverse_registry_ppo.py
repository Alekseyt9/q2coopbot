"""Seal the eight-update training ledger and its fixed final 24-condition comparison."""
import argparse
import collections
from pathlib import Path

from evaluate_registered_combat_models import read, save, sha
import report_combat_movement_variance_confirmation as paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('workflow', 'spec', 'quality', 'damage', 'out'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    assert not args.out.exists()
    root, pins = args.workflow.resolve(), {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        assert expected is None or digest == expected, str(path)
        assert str(path) not in pins or pins[str(path)] == digest
        pins[str(path)] = digest
        return path

    def sealed(directory):
        receipt = read(pin(directory / 'complete.json'))
        return receipt, read(pin(directory / 'report.json', receipt['report_sha256']))

    pin(__file__)
    pin(paired.__file__)
    workflow = read(pin(root / 'complete.json'))
    assert workflow['state'] == 'complete' and workflow['completed_training_rounds'] == 8
    assert workflow['native_training_cases'] == workflow['cuda_training_cases'] == 384
    spec = read(pin(args.spec))
    protocol = read(pin(spec['experiment_protocol'], spec['experiment_protocol_sha256']))
    assert protocol['version'] == 'combat_diverse_registry_ppo_v1' and protocol['rounds'] == 8
    assert spec['expansion_gate'] == protocol['evaluation']['expansion_gate']
    gate = spec['expansion_gate']
    assert gate == dict(additional_goals_over_source=3, goals_at_least_rules=True, deaths_no_more_than_source=True,
                        deaths_no_more_than_rules=True, received_numerator=11, received_denominator=10,
                        received_compare_to=['source', 'rules'])
    rounds, training_seeds, rollouts = [], set(), set()
    previous = protocol['initializer_weights_sha256']
    for number in range(1, 9):
        directory = root / f'round-{number:02d}'
        seal, report = sealed(directory / 'update')
        assert report['version'] == protocol['version'] and report['round'] == number and report['device'] == 'cuda'
        assert report['behavior_sha256'] == previous and report['experiment_protocol_sha256'] == spec['experiment_protocol_sha256']
        assert report['training_performed'] and report['checkpoint_optimizer_roundtrip_exact'] and report['rng_preserved']
        assert not report['quality_verified']
        pin(directory / 'update/weights.json', seal['weights_sha256'])
        pin(directory / 'update/checkpoint.pt', seal['checkpoint_sha256'])
        corpus = directory / 'corpus'
        closed = read(pin(corpus / 'complete.json'))
        assert closed['native_cases'] == closed['cuda_cases'] == 48 and not closed['training_performed']
        collection = read(pin(corpus / 'protocol.json', closed['protocol_sha256']))
        assert collection['train_only'] and len(collection['assignments']) == 1
        assignment = collection['assignments'][0]
        assert assignment['split'] == 'train' and len(set(assignment['seeds'])) == 48
        assert not training_seeds.intersection(assignment['seeds'])
        training_seeds.update(assignment['seeds'])
        plan = read(pin(assignment['plan'], assignment['plan_sha256']))
        assert plan['model_sha256'] == previous and {t['episode']['id'] for t in plan['tasks']} == set(protocol['episodes'])
        assert all(t['split'] == 'train' and len(t['seeds']) == 2 and t['modes'] == ['learned'] for t in plan['tasks'])
        meta = read(pin(corpus / 'processing-train/merged/report.json'))
        assert meta['model_sha256'] == previous and meta['rollout_sha256'] == report['rollout_sha256']
        assert meta['rollout_sha256'] not in rollouts
        rollouts.add(meta['rollout_sha256'])
        freshness = read(pin(directory / 'freshness.json'))
        assert freshness['state'] == 'passed' and freshness['seed_count'] == 48
        native_seal, native = sealed(directory / 'native-credit')
        _, advantage = sealed(directory / 'advantage-credit')
        assert not native['unassigned_native_effects'] and native['totals']['rows'] == advantage['rows'] == report['rows']
        rounds.append(dict(round=number, rows=report['rows'], geometry_eligible_rows=report['eligible_rows'],
            actor_steps=report['base_steps'], head_steps=report['head_steps'], critic_steps=report['critic_steps'],
            accepted_step_lr_scale_sum=sum(.5**t['retry'] for t in report['trials'] if t['accepted']),
            accepted_backtracked_steps=sum(t['accepted'] and t['retry']>0 for t in report['trials']),
            full_kl=report['full_approx_kl'], geometry_kl=report['category_exact_kl_active'],
            value_train_mse_before=report['value_mse_before'], value_train_mse_after=report['value_mse_after'],
            weights_sha256=seal['weights_sha256']))
        previous = seal['weights_sha256']
    assert len(training_seeds) == 384 and previous == workflow['final_weights_sha256']
    evaluation = root / 'evaluation'
    closed_path = pin(evaluation / 'complete.json', workflow['evaluation_seal_sha256'])
    closed = read(closed_path)
    assert closed['spec_sha256'] == sha(args.spec) and closed['native_cases'] == 72 and closed['cuda_cases'] == 48
    fixture = read(pin(evaluation / 'fixture-proof.json'))
    assert next(m['model_sha256'] for m in fixture['models'] if m['label'] == 'trained') == previous
    for name in ('source', 'trained', 'rules'):
        plan = read(pin(evaluation / (name + '-plan.json')))
        assert not training_seeds.intersection(s for t in plan['tasks'] for s in t['seeds'])
    audits = []
    for directory in (args.quality, args.damage):
        _, report = sealed(directory)
        assert report['inputs'][str(closed_path)] == sha(closed_path)
        audits.append(report)
    assert set(audits[0]['variants']) == {'source', 'trained'} and len(audits[1]['cases']) == 48
    totals, comparisons = {}, {}
    for item in closed['comparisons']:
        assert item['after'] == 'trained'
        data = read(pin(evaluation / ('quality-' + item['name'] + '.json'), item['sha256']))
        pairs = data['pairs']
        assert len(pairs) == 24
        assert 'trained' not in totals or totals['trained'] == data['totals']['after']
        totals['trained'] = data['totals']['after']
        totals[item['before']] = data['totals']['before']
        groups = collections.defaultdict(list)
        for case in pairs:
            episode = case['episode']
            groups['map:' + case['map']].append(case)
            groups['site:' + ('01' if '-site-01-' in episode else '02')].append(case)
            groups['loadout:' + ('recovery' if episode.endswith('-recovery') else 'ordinary')].append(case)
            groups['enemies:' + ('3' if '-gunner-' in episode else '2')].append(case)
        assert all(len(cases) == (6 if name.startswith('map:') else 12) for name, cases in groups.items())
        comparisons[item['name']] = dict(paired=paired.paired_summary(pairs),
            strata={name: paired.paired_summary(cases) for name, cases in sorted(groups.items())})
    assert set(totals) == {'source', 'trained', 'rules'}
    ownership = read(pin(evaluation / 'ownership-trained.json'))
    assert ownership['pool_sha256'] == closed['pool_sha256'] and len(ownership['cases']) == 24
    after, before, rules = (totals[k] for k in ('trained', 'source', 'rules'))
    assert all(type(t[k]) is int for t in totals.values() for k in ('goal', 'death', 'received'))
    checks = dict(goals_at_least_source_plus_three=after['goal'] >= before['goal'] + 3,
        goals_at_least_rules=after['goal'] >= rules['goal'], deaths_no_more_than_source=after['death'] <= before['death'],
        deaths_no_more_than_rules=after['death'] <= rules['death'],
        received_within_110_percent_source=10 * after['received'] <= 11 * before['received'],
        received_within_110_percent_rules=10 * after['received'] <= 11 * rules['received'],
        observed_combat_owned_by_model=all(ownership['totals'][k] == 0 for k in
          ('visible_enemy_rules_frames', 'visible_enemy_rules_attack_frames', 'visible_rules_missing_applied_frames', 'unknown_observation_rows'))
          and not any(c['goal'] and c['rules_attack_events'] for c in ownership['cases']))
    for path, digest in pins.items():
        assert sha(path) == digest, path
    args.out.mkdir(parents=True)
    save(args.out / 'report.json', dict(state='complete', version='combat_diverse_registry_ppo_result_v1',
        gate_passed=all(checks.values()), checks=checks, rounds=rounds, training_cases=384,
        training_rows=sum(r['rows'] for r in rounds), totals=totals, comparisons=comparisons,
        actions=audits[0]['variants'], damage_distribution=audits[1]['cases'], inputs=pins,
        neural_execution=False, new_games=False, training_performed=False, promotion=False,
        scope='Only final eighth update evaluated. Existing4maps/2sites/three-monster-class recipes, fresh validation seeds. '
              'Strata/sign tests descriptive; training MSE is not battle quality. No campaign/human-coop or general-superiority claim.'))
    save(args.out / 'complete.json', dict(state='complete', report_sha256=sha(args.out / 'report.json')))
    print(dict(gate_passed=all(checks.values()), checks=checks, totals=totals))


if __name__ == '__main__':
    main()
