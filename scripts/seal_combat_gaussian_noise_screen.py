"""Apply the frozen Gaussian calibration screen gate; native metadata arithmetic only."""
import argparse
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('evaluation','spec','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();h=sha(path);assert expected is None or expected==h,str(path)
        pins[str(path)]=h;return path
    seal=read(pin(a.evaluation/'complete.json'));spec=read(pin(a.spec,seal['spec_sha256']))
    p=read(pin(spec['experiment_protocol'],spec['experiment_protocol_sha256']))
    assert p['version']=='combat_gaussian_noise_screen_v1' and seal['native_cases']==32 and seal['cuda_cases']==24
    assert [v['label'] for v in spec['variants']]==['source','movement-quarter','all-quarter','rules']
    totals={};pairs={}
    for comparison in spec['comparisons']:
        q=read(pin(a.evaluation/('quality-'+comparison['name']+'.json')))
        assert q['state']=='complete' and len(q['pairs'])==8
        assert q['before_pool_sha256']==q['after_pool_sha256']==seal['pool_sha256']
        for side,label in [('before',comparison['before']),('after',comparison['after'])]:
            if label in totals:assert totals[label]==q['totals'][side]
            totals[label]=q['totals'][side]
        pairs[comparison['name']]=dict(gained=sum(not r['metrics']['before']['goal'] and r['metrics']['after']['goal'] for r in q['pairs']),
            lost=sum(r['metrics']['before']['goal'] and not r['metrics']['after']['goal'] for r in q['pairs']))
    shots=read(pin(a.evaluation/'machinegun-outcomes.json'));efficiency=read(pin(a.evaluation/'action-efficiency.json'))
    assert shots['state']==efficiency['state']=='complete'
    proofs={}
    for label in ('source','movement-quarter','all-quarter'):
        records=[]
        for path in sorted((a.evaluation/('processing-'+label)).glob('case-*/cuda/cuda-verification.json')):
            r=read(pin(path));assert r['state']=='passed' and r['device']=='cuda'
            assert r['frame_hash_candidate_assignments_verified'] and r['original_latent_transform_verified']
            assert r['shared_candidate_precision']=='float64' and r['shared_candidate_all_logits_max_error']<.001 and r['shared_candidate_all_probabilities_max_error']<.0001
            records.append(r)
        assert len(records)==8
        proofs[label]=dict(cases=8,eligible_frames=sum(r['shared_candidate_frames'] for r in records),
            max_logit_error=max(r['shared_candidate_all_logits_max_error'] for r in records))
    source,rules=totals['source'],totals['rules'];g=p['criteria'];checks={};passed=[]
    source_shot=shots['variants']['source']['all']['live_monster_damage_fraction'];assert source_shot is not None
    source_behavior=efficiency['variants']['source']['totals']
    for label in ('movement-quarter','all-quarter'):
        t=totals[label];s=shots['variants'][label]['all']['live_monster_damage_fraction'];b=efficiency['variants'][label]['totals']
        tests=dict(goals_source=t['goal']>=min(g['goal_ceiling'],source['goal']+g['goals_source_margin']),
            goals_rules=t['goal']>=min(g['goal_ceiling'],rules['goal']+g['goals_rules_margin']),
            deaths_source=t['death']<=source['death'],deaths_rules=t['death']<=rules['death'],
            received=t['received']<=g['received_rules_ratio_max']*rules['received'],
            shots=s is not None and s>=source_shot+g['mg_live_fraction_source_margin_min'],
            large_yaw=b['large_yaw_command_fraction'] is not None and source_behavior['large_yaw_command_fraction'] is not None and b['large_yaw_command_fraction']<=source_behavior['large_yaw_command_fraction']+g['large_yaw_source_margin_max'],
            targetless=b['attack_without_target_fraction'] is not None and source_behavior['attack_without_target_fraction'] is not None and b['attack_without_target_fraction']<=source_behavior['attack_without_target_fraction']+g['targetless_source_margin_max'])
        checks[label]=tests
        if all(tests.values()):passed.append(label)
    selected=sorted(passed,key=lambda label:(-totals[label]['goal'],totals[label]['death'],totals[label]['received'],label))[0] if passed else None
    for path,h in pins.items():assert sha(path)==h,path
    a.out.mkdir(parents=True)
    report=dict(state='complete',version='combat_gaussian_noise_screen_decision_v1',totals=totals,pairs=pairs,checks=checks,
        gate_passed=bool(passed),candidates_admitted_for_expansion=passed,selected_for_expansion=selected,
        machinegun={k:v['all'] for k,v in shots['variants'].items()},behavior={k:v['totals'] for k,v in efficiency['variants'].items()},
        cuda_proofs=proofs,inputs=pins,training_performed=False,promotion=False,
        scope='Fixed8-condition Gaussian calibration screen, not training or stable superiority. Passing only admits independent confirmation; campaign/coop acceptance remains open.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(dict(totals=totals,checks=checks,selected=selected))


if __name__=='__main__':main()
