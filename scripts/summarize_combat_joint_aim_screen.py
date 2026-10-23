"""Seal native quality, CUDA acceptance and actual shot outcomes for the pilot."""
import argparse
from pathlib import Path
from evaluate_registered_combat_models import read,sha,save


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('evaluation','spec','training','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();digest=sha(path);assert expected is None or digest==expected,str(path)
        pins[str(path)]=digest;return path
    closed=read(pin(a.evaluation/'complete.json'));assert closed['state']=='complete' and closed['native_cases']==32 and closed['cuda_cases']==24
    spec=read(pin(a.spec,closed['spec_sha256']));assert [v['label'] for v in spec['variants']]==['source','initial','trained','rules']
    training_seal=read(pin(a.training/'complete.json'))
    training=read(pin(a.training/'report.json',training_seal['report_sha256']))
    assert training['device']=='cuda' and training['old_actor_and_adam_exact'] and training['predictor_exact']
    totals={};comparisons={}
    for c in spec['comparisons']:
        q=read(pin(a.evaluation/('quality-'+c['name']+'.json')));assert q['state']=='complete'
        assert q['before_pool_sha256']==q['after_pool_sha256']==closed['pool_sha256']
        for side,label in (('before',c['before']),('after',c['after'])):
            if label in totals:assert totals[label]==q['totals'][side]
            totals[label]=q['totals'][side]
        pairs=[(v['metrics']['before']['goal'],v['metrics']['after']['goal']) for v in q['pairs']]
        comparisons[c['name']]=dict(gained=sum(not before and after for before,after in pairs),lost=sum(before and not after for before,after in pairs))
    aim=read(pin(a.evaluation/'machinegun-aim.json'));outcomes=read(pin(a.evaluation/'machinegun-outcomes.json'));eff=read(pin(a.evaluation/'action-efficiency.json'))
    assert aim['state']==outcomes['state']==eff['state']=='complete'
    variants={};cuda=[]
    for v in spec['variants']:
        label=v['label'];variants[label]=dict(totals=totals[label])
        if v['mode']=='rules':continue
        variants[label].update(machinegun=outcomes['variants'][label]['all'],
            native_aim=aim['variants'][label]['groups']['all']['summary'],behavior=eff['variants'][label]['totals'])
        verifications=[]
        for path in sorted((a.evaluation/('processing-'+label)).glob('case-*/cuda/cuda-verification.json')):
            p=read(pin(path));assert p['state']=='passed' and p['device']=='cuda'
            if label!='source':assert p['frame_hash_candidate_assignments_verified'] and p['original_latent_transform_verified']
            verifications.append(p)
        assert len(verifications)==8
        variants[label]['cuda']=dict(cases=8,rows=sum(p['rows'] for p in verifications),max_lp_error=max(p['log_probability_max_error'] for p in verifications))
        cuda.extend(verifications)
    assert variants['trained']['cuda']['rows']==848
    # This is a decision after the predeclared pilot, not a new predeclared gate.
    reasons=['No win-count improvement versus source:5/8 each,2gained/2lost.',
        'Actual MG living-monster damage fraction did not improve:111/173 versus149/229 source and113/165 initial.',
        'Stationary-clear and large-yaw command fractions increased in these trajectories.',
        'Eight cases per variant cannot establish superiority; source itself beat rules5/8versus4/8.']
    report=dict(state='complete',version='combat_joint_aim_head_only_screen_decision_v1',variants=variants,comparisons=comparisons,
        native_cases=32,cuda_cases=24,cuda_rows=sum(p['rows'] for p in cuda),head_training_complete=True,
        nonzero_head_native_cuda_accepted=True,promotion=False,expanded_confirmation=False,reasons=reasons,
        next_hypothesis='Matched base-only versus joint actor/candidate updates from the same initial behavior and same TRAIN; frozen predictor, unchanged reward. No validation fitting.',
        inputs=pins,script_sha256=sha(__file__),scope='Technical acceptance passed; head-only combat superiority unproven. Shot metrics compare different policy trajectories, not counterfactual identical shots.')
    for path,digest in pins.items():assert sha(Path(path))==digest
    a.out.mkdir(parents=True);save(a.out/'report.json',report)
    save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json'),promotion=False,expanded_confirmation=False))
    print({'state':'complete','wins':{label:v['totals']['goal'] for label,v in variants.items()},'comparisons':comparisons,'promotion':False,'expanded_confirmation':False})


if __name__=='__main__':main()
