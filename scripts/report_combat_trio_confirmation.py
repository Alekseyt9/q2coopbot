"""Close independent confirmation of unchanged matched endpoints after the small gate."""
import argparse
import collections
from pathlib import Path
import subprocess
import sys
import time

from evaluate_registered_combat_models import read,save,sha
import report_combat_movement_variance_confirmation as paired


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('evaluation','spec','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--wait',action='store_true')
    args=parser.parse_args();assert not args.out.exists()
    source_sha=sha(__file__)
    if args.wait:
        deadline=time.monotonic()+7200
        while not (args.evaluation/'complete.json').exists():
            status=args.evaluation/'progress.json'
            if status.exists():assert read(status)['stage']!='failed','Confirmation failed; inspect original, no restart'
            assert time.monotonic()<deadline,'Observation timeout; no restart'
            time.sleep(5)
    assert sha(__file__)==source_sha
    pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();digest=sha(path)
        assert expected is None or digest==expected,str(path)
        pins[str(path)]=digest;return path
    def sealed(directory):
        receipt=read(pin(directory/'complete.json'))
        return receipt,read(pin(directory/'report.json',receipt['report_sha256']))
    pin(__file__,source_sha);pin(paired.__file__)
    spec=read(pin(args.spec));assert spec['split']=='confirmation' and spec['count']==1
    prerequisite=read(pin(spec['prerequisite_report'],spec['prerequisite_report_sha256']))
    assert prerequisite['gate_passed'] and prerequisite['version']=='combat_trio_matched_screen_result_v1'
    gate=dict(additional_goals_over_full=3,additional_goals_over_rules=3,goals_at_least_parent=True,
        deaths_no_more_than=['full','rules'],received_compare_to=['full','rules'],received_numerator=11,received_denominator=10,automatic_promotion=False)
    assert spec['expansion_gate']==gate
    evaluation_path=pin(args.evaluation/'complete.json');closed=read(evaluation_path)
    assert closed['native_cases']==96 and closed['cuda_cases']==72 and closed['spec_sha256']==sha(args.spec)
    fixture=read(pin(args.evaluation/'fixture-proof.json'))
    assert fixture['split']=='confirmation' and fixture['cases_per_variant']==24
    previous_fixtures=[path for path in prerequisite['inputs'] if path.endswith(('evaluation/fixture-proof.json','evaluation\\fixture-proof.json'))]
    assert len(previous_fixtures)==1
    previous=read(pin(previous_fixtures[0],prerequisite['inputs'][previous_fixtures[0]]))
    weights={model['label']:model['model_sha256'] for model in fixture['models']}
    assert weights=={model['label']:model['model_sha256'] for model in previous['models']},'Endpoint weights changed'
    for label,script in (('native-quality','audit_registered_native_action_quality.py'),('damage','audit_combat_damage_distribution.py')):
        target=args.evaluation/label
        if not (target/'complete.json').exists():
            assert not target.exists(),'Partial audit preserved'
            with (args.evaluation/(label+'.log')).open('w',encoding='utf-8') as log:
                subprocess.run([sys.executable,'scripts/'+script,'--evaluation',str(args.evaluation),'--out',str(target)],stdout=log,stderr=log,check=True)
    _,quality=sealed(args.evaluation/'native-quality');_,damage=sealed(args.evaluation/'damage')
    assert len(damage['cases'])==72
    for audit in (quality,damage):assert audit['inputs'][str(evaluation_path)]==sha(evaluation_path)
    totals={};comparisons={}
    for item in closed['comparisons']:
        data=read(pin(args.evaluation/('quality-'+item['name']+'.json'),item['sha256']))
        assert item['after']=='decisions' and len(data['pairs'])==24
        assert 'decisions' not in totals or totals['decisions']==data['totals']['after']
        totals['decisions'],totals[item['before']]=data['totals']['after'],data['totals']['before']
        groups=collections.defaultdict(list)
        for case in data['pairs']:
            episode=case['episode'];groups['map:'+case['map']].append(case)
            groups['recovery' if episode.endswith('-recovery') else 'ordinary'].append(case)
            groups['enemies:3' if '-gunner-' in episode else 'enemies:2'].append(case)
            groups['site:01' if '-site-01-' in episode else 'site:02'].append(case)
        assert all(len(values)==(6 if name.startswith('map:') else 12) for name,values in groups.items())
        comparisons[item['name']]=dict(paired=paired.paired_summary(data['pairs']),strata={name:paired.paired_summary(values) for name,values in groups.items()})
    assert set(totals)=={'parent','full','decisions','rules'}
    decision=totals['decisions'];ownership=read(pin(args.evaluation/'ownership-decisions.json'))
    assert ownership['pool_sha256']==closed['pool_sha256'] and len(ownership['cases'])==24
    checks=dict(goals_full_plus_three=decision['goal']>=totals['full']['goal']+3,
        goals_rules_plus_three=decision['goal']>=totals['rules']['goal']+3,goals_at_least_parent=decision['goal']>=totals['parent']['goal'],
        observed_combat_owned_by_model=all(ownership['totals'][key]==0 for key in
            ('visible_enemy_rules_frames','visible_enemy_rules_attack_frames','visible_rules_missing_applied_frames','unknown_observation_rows'))
            and not any(case['goal'] and case['rules_attack_events'] for case in ownership['cases']))
    for label in ('full','rules'):
        checks['deaths_no_more_than_'+label]=decision['death']<=totals[label]['death']
        checks['received_within_110_percent_'+label]=10*decision['received']<=11*totals[label]['received']
    assert all(sha(path)==digest for path,digest in pins.items())
    args.out.mkdir(parents=True)
    save(args.out/'report.json',dict(state='complete',version='combat_trio_independent_confirmation_v1',gate_passed=all(checks.values()),
        checks=checks,totals=totals,comparisons=comparisons,actions=quality['variants'],weights_sha256=weights,inputs=pins,
        neural_execution=False,training_performed=False,new_games=False,promotion=False,
        scope='Independent confirmation split on24existing recipes/4maps/2sites with same frozen weights. '
              'No refit or endpoint selection after small-screen outcomes; no whole-campaign/human-coop or general-superiority claim.'))
    lines=['# Independent trio-policy confirmation','','| Variant | Goals / 24 | Deaths | Kills | Received HP |','|---|---:|---:|---:|---:|']
    for label in ('parent','full','decisions','rules'):
        t=totals[label];lines.append(f"| {label} | {t['goal']} | {t['death']} | {t['kills']} | {t['received']} |")
    lines.extend(['','Confirmation gate: '+str(all(checks.values()))+'. Promotion: false.'])
    (args.out/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(args.out/'complete.json',dict(state='complete',report_sha256=sha(args.out/'report.json'),markdown_sha256=sha(args.out/'results.md')))
    print(dict(gate_passed=all(checks.values()),totals=totals,checks=checks))


if __name__=='__main__':main()
