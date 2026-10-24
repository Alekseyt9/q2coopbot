"""Close the fixed final-endpoint body-input comparison without checkpoint selection."""
import argparse
from pathlib import Path

from evaluate_registered_combat_models import read,save,sha
from combat_execution_contract import LEGACY,require_same
from combat_body_training_scope import SCOPE
from report_combat_movement_variance_confirmation import paired_summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('workflow','protocol','spec','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();digest=pins.get(str(path)) or sha(path)
        assert expected is None or digest==expected,str(path)
        pins[str(path)]=digest;return path
    def sealed(path):
        seal=read(pin(path/'complete.json'));assert seal['state']=='complete'
        return seal,read(pin(path/'report.json',seal['report_sha256']))
    pin(__file__);protocol=read(pin(a.protocol));spec=read(pin(a.spec))
    assert protocol['version']==SCOPE and protocol['rounds']==4 and protocol['cases_per_update']==24
    assert spec['expansion_gate']==protocol['expansion_gate']
    arms=[('ablated',LEGACY),('observed',LEGACY)];training_seeds=set();training=[];endpoint_shas={}
    for arm,contract in arms:
        initial,init=sealed(a.workflow/'initial'/arm)
        assert init['device']=='cuda' and init['protocol_sha256']==sha(a.protocol)
        assert init['source_weights_sha256']==protocol['initial_sources'][arm]['weights_sha256']
        assert all(init[k] for k in ('source_training_tensors_exact','source_optimizers_exact','source_rng_exact','checkpoint_roundtrip_exact'))
        require_same(contract,init['combat_execution_contract'])
        assert init['feature_version']==protocol['features_by_arm'][arm] and init['shared_input_bias']
        assert init['disposable_resume_adam_step_exact'] and init['disposable_gradients']['shared_bias']>0
        endpoint_shas[arm+'-initial']=initial['weights_sha256'];previous=initial['weights_sha256']
        for number in range(1,5):
            root=a.workflow/f'round-{number:02d}'/arm
            seal,update=sealed(root/'update')
            assert update['version']==SCOPE and update['scope_updates']==number and update['device']=='cuda'
            assert update['experiment_protocol_sha256']==sha(a.protocol) and update['behavior_sha256']==previous
            assert all(update[k] for k in ('source_checkpoint_exact','frozen_original_actor_parameters_exact','original_coordinate_rows_exact','frozen_legacy_optimizer_exact',
                'checkpoint_roundtrip_exact','disposable_resume_adam_step_exact','disposable_critic_resume_adam_step_exact','rng_preserved'))
            require_same(contract,update['combat_execution_contract'])
            assert update['feature_version']==protocol['features_by_arm'][arm] and update['shared_input_bias']
            if arm=='ablated':assert not update['body_columns_changed'] and update['body_known_rows']==0
            corpus=read(pin(root/'corpus/complete.json'));assert corpus['native_cases']==corpus['cuda_cases']==24
            collection=read(pin(root/'corpus/protocol.json',corpus['protocol_sha256']))
            assignment=collection['assignments'][0]
            assert collection['train_only'] and assignment['split']=='train' and len(set(assignment['seeds']))==24
            require_same(contract,collection['combat_execution_contract'])
            assert not training_seeds.intersection(assignment['seeds'])
            training_seeds.update(assignment['seeds'])
            training.append(dict(arm=arm,round=number,rows=update['rows'],actor_steps=update['actor_steps'],critic_steps=update['critic_steps']))
            previous=seal['weights_sha256']
        endpoint_shas[arm+'-trained']=previous
    assert len(training_seeds)==192
    evaluation=a.workflow/'evaluation';closed=read(pin(evaluation/'complete.json'))
    assert closed['state']=='complete' and closed['native_cases']==40 and closed['cuda_cases']==32 and closed['spec_sha256']==sha(a.spec)
    fixture=read(pin(evaluation/'fixture-proof.json'));assert fixture['spec_sha256']==sha(a.spec)
    assert {m['label']:m['model_sha256'] for m in fixture['models']}==endpoint_shas
    totals,comparisons={},{}
    for item in closed['comparisons']:
        data=read(pin(evaluation/('quality-'+item['name']+'.json'),item['sha256']))
        assert item['after']=='observed-trained' and len(data['pairs'])==8
        assert not training_seeds.intersection(p['seed'] for p in data['pairs'])
        totals[item['before']],totals['observed-trained']=data['totals']['before'],data['totals']['after']
        comparisons[item['name']]=paired_summary(data['pairs'])
    assert set(totals)=={'ablated-initial','ablated-trained','observed-initial','observed-trained','rules'}
    ownership=read(pin(evaluation/'ownership-observed-trained.json'))
    assert ownership['pool_sha256']==closed['pool_sha256'] and len(ownership['cases'])==8
    gate=protocol['expansion_gate'];candidate=totals['observed-trained']
    checks=dict(goals_ablated_trained_plus_two=candidate['goal']>=totals['ablated-trained']['goal']+gate['additional_goals_over_ablated_trained'],
        goals_observed_initial_plus_one=candidate['goal']>=totals['observed-initial']['goal']+gate['additional_goals_over_observed_initial'],
        goals_at_least_ablated_initial=candidate['goal']>=totals['ablated-initial']['goal'],
        goals_at_least_rules=candidate['goal']>=totals['rules']['goal'],
        observed_combat_owned_by_model=all(ownership['totals'][k]==0 for k in
            ('visible_enemy_rules_frames','visible_enemy_rules_attack_frames','visible_rules_missing_applied_frames','unknown_observation_rows')))
    for label in ('ablated-trained','rules'):
        checks['deaths_no_more_than_'+label]=candidate['death']<=totals[label]['death']
        checks['received_within_110_percent_'+label]=10*candidate['received']<=11*totals[label]['received']
    qseal=read(pin(a.workflow/'native-quality/complete.json'))
    quality=read(pin(a.workflow/'native-quality/report.json',qseal['report_sha256']))
    assert all(sha(p)==h for p,h in pins.items())
    a.out.mkdir(parents=True)
    save(a.out/'report.json',dict(state='complete',version='combat_body_input_screen_result_v2',
        gate_passed=all(checks.values()),checks=checks,totals=totals,comparisons=comparisons,training=training,
        training_cases=192,training_rows=sum(t['rows'] for t in training),actions=quality['variants'],inputs=pins,
        neural_execution=False,new_games=False,promotion=False,
        scope='Four predeclared own-policy updates per arm. Final endpoints only, same CUDA initialization/Adam/RNG and learner scope. '
              'Seven Adam groups including shared input bias in both arms; body information is the difference. Eight fresh paired validation recipes, no checkpoint selection or held-out fitting. No general-superiority/campaign/coop claim.'))
    lines=['# Body input training comparison','','| Variant | Goals / 8 | Deaths | Kills | Received HP |','|---|---:|---:|---:|---:|']
    for label in ('ablated-initial','ablated-trained','observed-initial','observed-trained','rules'):
        t=totals[label];lines.append(f"| {label} | {t['goal']} | {t['death']} | {t['kills']} | {t['received']} |")
    lines+=['','Expansion gate: '+str(all(checks.values()))+'. Promotion: false.']
    (a.out/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json'),markdown_sha256=sha(a.out/'results.md')))
    print(dict(gate_passed=all(checks.values()),totals=totals))


if __name__=='__main__':main()
