"""Close the fixed matched four-update trio curriculum comparison."""
import argparse
from pathlib import Path
import time
import collections

from evaluate_registered_combat_models import read,save,sha
import report_combat_movement_variance_confirmation as paired


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('workflow','spec','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--wait',action='store_true')
    args=parser.parse_args();assert not args.out.exists()
    loaded_source_sha256=sha(__file__)
    if args.wait:
        deadline=time.monotonic()+7200
        while not (args.workflow/'complete.json').exists():
            if (args.workflow/'progress.json').exists():assert read(args.workflow/'progress.json')['stage']!='failed','Upstream failed; no restart'
            assert time.monotonic()<deadline,'Observation timeout; no restart'
            time.sleep(5)
    pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();digest=sha(path)
        assert expected is None or digest==expected,str(path)
        pins[str(path)]=digest;return path
    def sealed(directory):
        receipt=read(pin(directory/'complete.json'))
        return receipt,read(pin(directory/'report.json',receipt['report_sha256']))
    pin(__file__,loaded_source_sha256);pin(paired.__file__)
    workflow=read(pin(args.workflow/'complete.json'));spec=read(pin(args.spec))
    study=read(pin(spec['experiment_protocol'],spec['experiment_protocol_sha256']))
    assert workflow['protocol_sha256']==spec['experiment_protocol_sha256']
    assert workflow['native_training_cases']==workflow['cuda_training_cases']==80
    assert workflow['matched_updates_from_parent']==4
    counts=read(pin(args.workflow/'matched-update-counts.json'))
    assert counts['state']=='matched' and all(c['new_updates']==c['new_consumed_rollouts']==4 for c in counts['counts'].values())
    expected_gate=dict(additional_goals_over_full=2,goals_at_least_parent=True,goals_at_least_rules=True,
        deaths_no_more_than=['full','rules'],received_compare_to=['full','rules'],received_numerator=11,received_denominator=10,automatic_promotion=False)
    assert spec['expansion_gate']==expected_gate
    full_protocol=read(pin(study['full_contract'],study['full_contract_sha256']))
    decision_protocol=read(pin(study['decision_contract'],study['decision_contract_sha256']))
    migration_seal,migration=sealed(Path(study['decision_source']))
    assert migration['weights_bytes_exact'] and migration['all_neural_optimizer_rng_history_exact'] and not migration['training_performed']
    previous={'full':full_protocol['initial_full_weights_sha256'],'decisions':migration_seal['weights_sha256']}
    phases=[('full-round-01','full',1),('full-round-02','full',2),('decision-round-03','decisions',3),('full-round-03','full',3),('decision-round-04','decisions',4)]
    ledger=[];seeds={};all_seeds=set();rollouts=set()
    for label,kind,number in phases:
        root=args.workflow/label
        seal,update=sealed(root/'update')
        assert update['device']=='cuda' and update['behavior_sha256']==previous[kind]
        assert update['experiment_protocol_sha256']==sha(study['full_contract'] if kind=='full' else study['decision_contract'])
        if kind=='full':
            assert update['version']==full_protocol['version'] and update['round']==number
            assert update['checkpoint_optimizer_roundtrip_exact']
        else:
            assert update['version']==decision_protocol['version'] and update['scope_updates']==number
            assert update['checkpoint_roundtrip_exact'] and update['disposable_resume_adam_step_exact'] and update['frozen_coordinates_exact']
        pin(root/'update/weights.json',seal['weights_sha256']);pin(root/'update/checkpoint.pt',seal['checkpoint_sha256'])
        closed=read(pin(root/'corpus/complete.json'));assert closed['native_cases']==closed['cuda_cases']==16
        collection=read(pin(root/'corpus/protocol.json',closed['protocol_sha256']))
        assert collection['train_only'] and len(collection['assignments'])==1
        assignment=collection['assignments'][0];assert assignment['split']=='train' and len(set(assignment['seeds']))==16
        seeds[label]=set(assignment['seeds']);all_seeds.update(assignment['seeds'])
        plan=read(pin(assignment['plan'],assignment['plan_sha256']))
        assert plan['model_sha256']==previous[kind]
        expected=study['alignment_episodes'] if label=='full-round-01' else study['episodes']
        assert {task['episode']['id'] for task in plan['tasks']}==set(expected)
        if label!='full-round-01':assert all(len(task['episode']['monsters'])==3 for task in plan['tasks'])
        if kind=='decisions' or label=='full-round-01':assert read(pin(root/'corpus/paired-train-proof.json'))['state']=='matched'
        meta=read(pin(root/'corpus/processing-train/merged/report.json'))
        assert meta['rollout_sha256']==update['rollout_sha256'] and meta['rollout_sha256'] not in rollouts
        rollouts.add(meta['rollout_sha256'])
        _,native=sealed(root/'native-credit');_,advantage=sealed(root/'advantage-credit')
        assert native['totals']['rows']==advantage['rows']==update['rows'] and not native['unassigned_native_effects']
        ledger.append(dict(phase=label,rows=update['rows'],actor_steps=update.get('actor_steps',update.get('base_steps')),
            critic_steps=update['critic_steps'],weights_sha256=seal['weights_sha256'],
            accepted_lr_scale_sum=sum(.5**trial['retry'] for trial in update['trials'] if trial['accepted'])))
        previous[kind]=seal['weights_sha256']
    assert previous==workflow['final_weights']
    assert seeds['full-round-02']==seeds['decision-round-03'] and seeds['full-round-03']==seeds['decision-round-04']
    assert not seeds['full-round-01'].intersection(seeds['full-round-02']|seeds['full-round-03'])
    assert not seeds['full-round-02'].intersection(seeds['full-round-03'])
    evaluation=args.workflow/'evaluation'
    evaluation_path=pin(evaluation/'complete.json',workflow['evaluation_seal_sha256']);closed=read(evaluation_path)
    assert closed['native_cases']==32 and closed['cuda_cases']==24 and closed['spec_sha256']==sha(args.spec)
    fixture=read(pin(evaluation/'fixture-proof.json'))
    for label in ('full','decisions'):assert next(model['model_sha256'] for model in fixture['models'] if model['label']==label)==previous[label]
    _,quality=sealed(args.workflow/'native-quality');_,damage=sealed(args.workflow/'damage')
    assert len(damage['cases'])==24
    for audit in (quality,damage):assert audit['inputs'][str(evaluation_path)]==sha(evaluation_path)
    totals={};comparisons={}
    for item in closed['comparisons']:
        data=read(pin(evaluation/('quality-'+item['name']+'.json'),item['sha256']))
        assert item['after']=='decisions' and len(data['pairs'])==8
        assert 'decisions' not in totals or totals['decisions']==data['totals']['after']
        totals['decisions'],totals[item['before']]=data['totals']['after'],data['totals']['before']
        groups=collections.defaultdict(list)
        for case in data['pairs']:
            assert case['seed'] not in all_seeds
            groups['map:'+case['map']].append(case)
            groups['recovery' if case['episode'].endswith('-recovery') else 'ordinary'].append(case)
            groups['enemies:3' if '-gunner-' in case['episode'] else 'enemies:2'].append(case)
        comparisons[item['name']]=dict(paired=paired.paired_summary(data['pairs']),strata={key:paired.paired_summary(values) for key,values in groups.items()})
    assert set(totals)=={'parent','full','decisions','rules'}
    decision=totals['decisions'];ownership=read(pin(evaluation/'ownership-decisions.json'))
    assert ownership['pool_sha256']==closed['pool_sha256'] and len(ownership['cases'])==8
    checks=dict(goals_full_plus_two=decision['goal']>=totals['full']['goal']+2,goals_at_least_parent=decision['goal']>=totals['parent']['goal'],
        goals_at_least_rules=decision['goal']>=totals['rules']['goal'],observed_combat_owned_by_model=all(ownership['totals'][key]==0 for key in
          ('visible_enemy_rules_frames','visible_enemy_rules_attack_frames','visible_rules_missing_applied_frames','unknown_observation_rows'))
          and not any(case['goal'] and case['rules_attack_events'] for case in ownership['cases']))
    for label in ('full','rules'):
        checks['deaths_no_more_than_'+label]=decision['death']<=totals[label]['death']
        checks['received_within_110_percent_'+label]=10*decision['received']<=11*totals[label]['received']
    assert all(sha(path)==digest for path,digest in pins.items())
    args.out.mkdir(parents=True)
    save(args.out/'report.json',dict(state='complete',version='combat_trio_matched_screen_result_v1',gate_passed=all(checks.values()),
        checks=checks,totals=totals,comparisons=comparisons,actions=quality['variants'],ledger=ledger,matched_update_counts=counts,inputs=pins,
        neural_execution=False,training_performed=False,new_games=False,promotion=False,
        scope='Fixed matched four-update endpoints, fresh8case screen. Trio curriculum modifies training; evaluation remains balanced. '
              'Small sample; action metrics descriptive. No campaign/coop or general-superiority claim.'))
    lines=['# Matched trio curriculum comparison','','| Variant | Goals / 8 | Deaths | Kills | Received HP |','|---|---:|---:|---:|---:|']
    for label in ('parent','full','decisions','rules'):
        t=totals[label];lines.append(f"| {label} | {t['goal']} | {t['death']} | {t['kills']} | {t['received']} |")
    lines.extend(['','Expansion gate: '+str(all(checks.values()))+'. Promotion: false.'])
    (args.out/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(args.out/'complete.json',dict(state='complete',report_sha256=sha(args.out/'report.json'),markdown_sha256=sha(args.out/'results.md')))
    print(dict(gate_passed=all(checks.values()),totals=totals,checks=checks))


if __name__=='__main__':main()
