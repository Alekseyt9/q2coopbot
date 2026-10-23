"""Apply predeclared shared-aim native screen or confirmation gates."""
import argparse
from pathlib import Path
from evaluate_registered_combat_models import read, save, sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('evaluation','spec','protocol','out'):ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--confirmation',action='store_true')
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(p,expected=None):
        p=Path(p).resolve();h=sha(p);assert expected is None or h==expected,str(p);pins[str(p)]=h;return p
    seal=read(pin(a.evaluation/'complete.json'));spec=read(pin(a.spec,seal['spec_sha256']));pin(a.protocol)
    count=24 if a.confirmation else 8
    assert seal['native_cases']==4*count and seal['cuda_cases']==3*count
    assert [v['label'] for v in spec['variants']]==['source','shared','constant','rules']
    totals={};pairs={}
    for c in spec['comparisons']:
        q=read(pin(a.evaluation/('quality-'+c['name']+'.json')))
        assert q['state']=='complete' and q['before_pool_sha256']==q['after_pool_sha256']==seal['pool_sha256']
        assert len(q['pairs'])==count
        for side,label in (('before',c['before']),('after',c['after'])):
            if label in totals:assert totals[label]==q['totals'][side]
            totals[label]=q['totals'][side]
        pairs[c['name']]=dict(gained=sum(not p['metrics']['before']['goal'] and p['metrics']['after']['goal'] for p in q['pairs']),
            lost=sum(p['metrics']['before']['goal'] and not p['metrics']['after']['goal'] for p in q['pairs']))
    shots=read(pin(a.evaluation/'machinegun-outcomes.json'));eff=read(pin(a.evaluation/'action-efficiency.json'))
    assert shots['state']==eff['state']=='complete'
    proofs=[]
    for p in sorted((a.evaluation/'processing-shared').glob('case-*/cuda/cuda-verification.json')):
        r=read(pin(p));assert r['state']=='passed' and r['device']=='cuda'
        assert r['frame_hash_candidate_assignments_verified'] and r['original_latent_transform_verified']
        assert r['shared_candidate_all_logits_max_error']<.001 and r['shared_candidate_all_probabilities_max_error']<.0001
        proofs.append(r)
    assert len(proofs)==count and sum(r['shared_candidate_frames'] for r in proofs)>0
    t,s,c,r=(totals[k] for k in ('shared','source','constant','rules'))
    if not a.confirmation:
        checks=dict(goals_source=t['goal']>=s['goal']+2,goals_constant=t['goal']>=c['goal']+1,goals_rules=t['goal']>=r['goal'],
                    deaths=t['death']<=s['death'],received=t['received']<=1.1*s['received'])
    else:
        se,te=(eff['variants'][k]['totals'] for k in ('source','shared'))
        checks=dict(goals_source=t['goal']>=s['goal']+4,goals_constant=t['goal']>=c['goal']+3,goals_rules=t['goal']>=r['goal']+3,
                    deaths=t['death']<=r['death'],received=t['received']<=1.1*r['received'],
                    mg_live_fraction=shots['variants']['shared']['all']['live_monster_damage_fraction']>=shots['variants']['source']['all']['live_monster_damage_fraction']-.03,
                    large_yaw=te['large_yaw_command_fraction']<=se['large_yaw_command_fraction']+.02,
                    targetless_attack=te['attack_without_target_fraction']<=se['attack_without_target_fraction']+.03)
    for p,h in pins.items():assert sha(p)==h,p
    a.out.mkdir(parents=True)
    result=dict(state='complete',version='combat_shared_aim_native_decision_v1',confirmation=a.confirmation,totals=totals,pairs=pairs,
                checks=checks,gate_passed=all(checks.values()),shared_cuda_frames=sum(r['shared_candidate_frames'] for r in proofs),
                shared_logit_error=max(r['shared_candidate_all_logits_max_error'] for r in proofs),
                machinegun={k:v['all'] for k,v in shots['variants'].items()},behavior={k:v['totals'] for k,v in eff['variants'].items()},
                inputs=pins,script_sha256=sha(__file__),promotion=False,training_performed=False,
                scope='Frozen finite registry cohort; a passing gate permits a broader check only, never automatic production promotion.')
    save(a.out/'report.json',result);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print(dict(totals=totals,pairs=pairs,checks=checks,gate_passed=result['gate_passed'],shared_cuda_frames=result['shared_cuda_frames']))


if __name__=='__main__':main()
