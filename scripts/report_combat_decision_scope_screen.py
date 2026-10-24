"""Seal fixed first-endpoint decision-scope quality and fresh continuation evidence."""
import argparse
import collections
from pathlib import Path
import time

from evaluate_registered_combat_models import read, save, sha
import report_combat_movement_variance_confirmation as paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('workflow','spec','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--wait',action='store_true')
    args = parser.parse_args()
    assert not args.out.exists()
    if args.wait:
        deadline = time.monotonic()+7200
        while not (args.workflow/'complete.json').exists():
            progress = args.workflow/'progress.json'
            if progress.exists():
                assert read(progress)['stage']!='failed','Original workflow failed; no restart'
            assert time.monotonic()<deadline,'Observation expired; no automatic restart'
            time.sleep(5)
    pins = {}

    def pin(path,expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        assert expected is None or digest==expected,str(path)
        pins[str(path)] = digest
        return path

    def sealed(directory,expected=None):
        receipt = read(pin(directory/'complete.json',expected))
        return receipt,read(pin(directory/'report.json',receipt['report_sha256']))

    pin(__file__)
    pin(paired.__file__)
    workflow = read(pin(args.workflow/'complete.json'))
    assert workflow['native_training_cases']==workflow['cuda_training_cases']==16
    assert workflow['native_evaluation_cases']==32 and workflow['cuda_evaluation_cases']==24
    spec = read(pin(args.spec))
    protocol = read(pin(spec['experiment_protocol'],spec['experiment_protocol_sha256']))
    assert workflow['protocol_sha256']==spec['experiment_protocol_sha256']
    assert protocol['version']=='combat_decision_scope_ppo_v1'
    gate = dict(additional_goals_over_full=2,goals_at_least_parent=True,goals_at_least_rules=True,
        deaths_no_more_than=['full','rules'],received_compare_to=['full','rules'],received_numerator=11,received_denominator=10,
        automatic_promotion=False)
    assert spec['expansion_gate']==gate
    first_seal,first = sealed(args.workflow/'update',workflow['first_update_seal_sha256'])
    last_seal,last = sealed(args.workflow/'continuation/update',workflow['continuation_seal_sha256'])
    assert first['version']==last['version']==protocol['version']
    assert first['scope_updates']==1 and last['scope_updates']==2
    assert first['behavior_sha256']==protocol['initial_source_weights_sha256']
    assert last['behavior_sha256']==first_seal['weights_sha256']
    assert workflow['quality_weights_sha256']==first_seal['weights_sha256']
    for directory,seal,report in ((args.workflow/'update',first_seal,first),(args.workflow/'continuation/update',last_seal,last)):
        assert report['device']=='cuda' and report['experiment_protocol_sha256']==sha(spec['experiment_protocol'])
        assert all(report[name] for name in ('source_checkpoint_exact','frozen_coordinates_exact','frozen_legacy_optimizer_exact',
                    'checkpoint_roundtrip_exact','disposable_resume_adam_step_exact','rng_preserved'))
        pin(directory/'weights.json',seal['weights_sha256'])
        pin(directory/'checkpoint.pt',seal['checkpoint_sha256'])
    assert first['matched_control_critic_exact']
    full_seal,full_report = sealed(Path(protocol['control_update']))
    assert full_report['behavior_sha256']==first['behavior_sha256']
    assert full_report['rollout_sha256']==first['rollout_sha256']==protocol['initial_rollout_sha256']
    training_seeds = set()
    for corpus in (Path(protocol['initial_corpus']),args.workflow/'continuation/corpus'):
        closed = read(pin(corpus/'complete.json'))
        config = read(pin(corpus/'protocol.json',closed['protocol_sha256']))
        assert config['train_only'] and closed['native_cases']==closed['cuda_cases']==16
        assignment = config['assignments'][0]
        assert assignment['split']=='train' and len(set(assignment['seeds']))==16
        assert not training_seeds.intersection(assignment['seeds'])
        training_seeds.update(assignment['seeds'])
        plan = read(pin(assignment['plan'],assignment['plan_sha256']))
        assert all(task['split']=='train' for task in plan['tasks'])
    assert len(training_seeds)==32
    evaluation = args.workflow/'evaluation'
    evaluation_path = pin(evaluation/'complete.json',workflow['evaluation_seal_sha256'])
    closed = read(evaluation_path)
    assert closed['spec_sha256']==sha(args.spec) and closed['native_cases']==32 and closed['cuda_cases']==24
    fixture = read(pin(evaluation/'fixture-proof.json'))
    models = {model['label']:model for model in fixture['models']}
    assert set(models)=={'parent','full','decisions'}
    assert models['decisions']['model_sha256']==first_seal['weights_sha256']
    assert models['parent']['model_sha256']==protocol['initial_source_weights_sha256']
    assert models['full']['model_sha256']==full_seal['weights_sha256']
    _,quality = sealed(args.workflow/'native-quality')
    _,damage = sealed(args.workflow/'damage')
    for audit in (quality,damage):
        assert audit['inputs'][str(evaluation_path)]==sha(evaluation_path)
    assert len(damage['cases'])==24
    totals, comparisons = {},{}
    for item in closed['comparisons']:
        data = read(pin(evaluation/('quality-'+item['name']+'.json'),item['sha256']))
        assert item['after']=='decisions' and len(data['pairs'])==8
        assert 'decisions' not in totals or totals['decisions']==data['totals']['after']
        totals['decisions'],totals[item['before']] = data['totals']['after'],data['totals']['before']
        groups = collections.defaultdict(list)
        for case in data['pairs']:
            assert case['seed'] not in training_seeds
            groups['map:'+case['map']].append(case)
            groups['recovery' if case['episode'].endswith('-recovery') else 'ordinary'].append(case)
            groups['enemies:3' if '-gunner-' in case['episode'] else 'enemies:2'].append(case)
        comparisons[item['name']] = dict(paired=paired.paired_summary(data['pairs']),
            strata={name:paired.paired_summary(cases) for name,cases in groups.items()})
    assert set(totals)=={'parent','full','decisions','rules'}
    decision = totals['decisions']
    ownership = read(pin(evaluation/'ownership-decisions.json'))
    assert ownership['pool_sha256']==closed['pool_sha256'] and len(ownership['cases'])==8
    checks = dict(goals_full_plus_two=decision['goal']>=totals['full']['goal']+2,
        goals_at_least_parent=decision['goal']>=totals['parent']['goal'],
        goals_at_least_rules=decision['goal']>=totals['rules']['goal'],
        observed_combat_owned_by_model=all(ownership['totals'][key]==0 for key in
            ('visible_enemy_rules_frames','visible_enemy_rules_attack_frames','visible_rules_missing_applied_frames','unknown_observation_rows'))
            and not any(case['goal'] and case['rules_attack_events'] for case in ownership['cases']))
    for label in ('full','rules'):
        checks['deaths_no_more_than_'+label] = decision['death']<=totals[label]['death']
        checks['received_within_110_percent_'+label] = 10*decision['received']<=11*totals[label]['received']
    assert all(sha(path)==digest for path,digest in pins.items())
    args.out.mkdir(parents=True)
    save(args.out/'report.json',dict(state='complete',version='combat_decision_scope_screen_result_v1',
        gate_passed=all(checks.values()),checks=checks,totals=totals,comparisons=comparisons,actions=quality['variants'],
        first_update={name:first[name] for name in ('actor_steps','critic_steps','movement_conditional_kl','aim_conditional_kl','weights_sha256')},
        continuation={name:last[name] for name in ('actor_steps','critic_steps','scope_updates','behavior_sha256','weights_sha256')},
        inputs=pins,neural_execution=False,new_games=False,training_performed=False,promotion=False,
        scope='First checkpoint only, fresh8recipe screen. Second checkpoint verifies fresh on-policy continuation and is not evaluated. '
              'Small sample and descriptive motion rates; no general-superiority/campaign/coop claim.'))
    lines=['# Decision-scope comparison','','| Variant | Goals / 8 | Deaths | Kills | Received HP |','|---|---:|---:|---:|---:|']
    for label in ('parent','full','decisions','rules'):
        t=totals[label]
        lines.append(f"| {label} | {t['goal']} | {t['death']} | {t['kills']} | {t['received']} |")
    lines.extend(['','Expansion gate: '+str(all(checks.values()))+'. Promotion: false.'])
    (args.out/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(args.out/'complete.json',dict(state='complete',report_sha256=sha(args.out/'report.json'),markdown_sha256=sha(args.out/'results.md')))
    print(dict(gate_passed=all(checks.values()),totals=totals,checks=checks))


if __name__=='__main__':
    main()
