"""Seal the predeclared endpoint; passing a small screen never promotes a model."""
import argparse
from pathlib import Path
from evaluate_registered_combat_models import read, save, sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('evaluation','spec','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(p,expected=None):
        p=p.resolve();digest=sha(p);assert expected is None or expected==digest,str(p);pins[str(p)]=digest;return p
    closed=read(pin(a.evaluation/'complete.json'));spec=read(pin(a.spec,closed['spec_sha256']))
    assert closed['state']=='complete' and closed['native_cases']==48 and closed['cuda_cases']==40
    assert [v['label'] for v in spec['variants']]==['source','initial','B','G','S','rules']
    totals={};comparisons={}
    for c in spec['comparisons']:
        q=read(pin(a.evaluation/('quality-'+c['name']+'.json')));assert q['state']=='complete'
        assert q['before_pool_sha256']==q['after_pool_sha256']==closed['pool_sha256']
        for side,label in (('before',c['before']),('after',c['after'])):
            if label in totals:assert totals[label]==q['totals'][side]
            totals[label]=q['totals'][side]
        pairs=[(p['metrics']['before']['goal'],p['metrics']['after']['goal']) for p in q['pairs']]
        comparisons[c['name']]=dict(gained=sum(not b and t for b,t in pairs),lost=sum(b and not t for b,t in pairs))
    eff=read(pin(a.evaluation/'action-efficiency.json'));shots=read(pin(a.evaluation/'machinegun-outcomes.json'))
    assert eff['state']==shots['state']=='complete'
    variants={};eligible=[]
    for v in spec['variants']:
        label=v['label'];variants[label]=dict(totals=totals[label])
        if label=='rules':continue
        variants[label].update(behavior=eff['variants'][label]['totals'],machinegun=shots['variants'][label]['all'])
        proofs=[]
        for p in sorted((a.evaluation/('processing-'+label)).glob('case-*/cuda/cuda-verification.json')):
            proof=read(pin(p));assert proof['state']=='passed' and proof['device']=='cuda';proofs.append(proof)
            if label!='source':assert proof['frame_hash_candidate_assignments_verified'] and proof['original_latent_transform_verified']
        assert len(proofs)==8
        variants[label]['cuda']=dict(cases=8,rows=sum(p['rows'] for p in proofs))
        if label in ('B','G','S'):
            root=Path(v['update']);seal=read(pin(root/'complete.json'));trained=read(pin(root/'report.json',seal['report_sha256']))
            assert trained['arm']==label and trained['round']==4 and trained['device']=='cuda'
            controls=[totals['source'],totals['rules']];t=totals[label]
            passed=(all(t['goal']>=c['goal']+2 and t['death']<=c['death'] for c in controls)
                and t['received']<=1.1*min(c['received'] for c in controls))
            variants[label]['predeclared_expansion_gate_passed']=passed
            if passed:eligible.append(label)
    report=dict(state='complete',version='combat_joint_aim_budget_screen_decision_v1',variants=variants,comparisons=comparisons,
        native_cases=48,cuda_cases=40,eligible_for_new_24_case_confirmation=eligible,promotion=False,expanded_confirmation_performed=False,
        inputs=pins,script_sha256=sha(__file__),
        scope='Fixed endpoint after four own-policy updates per arm. Only an expansion gate on eight cases; no production superiority claim. Confirm all qualifying candidates on new paired seeds.')
    a.out.mkdir(parents=True);save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print({k:v['totals'] for k,v in variants.items()});print('Eligible for confirmation:',eligible)


if __name__=='__main__':main()
