"""Freeze live controls after CUDA-only distillation diagnosis."""
import argparse, copy, json, pathlib, shutil
from rebind_combat_architecture_evaluation import read, sha, save

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--template',type=pathlib.Path,required=True)
    ap.add_argument('--priors',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True);a=ap.parse_args()
    template=a.template.resolve();priors=a.priors.resolve();out=a.out.resolve();assert not out.exists()
    old=read(template/'protocol.json');report=read(priors/'report.json');audit=read(priors/'cuda-audit.json')
    assert report['state']=='complete' and report['source_unchanged'] and audit['device']=='cuda'
    entries=[]
    for i,seed in enumerate((20261007,20261008)):
        entries.extend([(f'm{i}','before',priors/f'attention64-identity-seed-{seed}',old['evaluations'][0]),
                        (f'm{i}','after',priors/f'attention128-balanced-seed-{seed}',old['evaluations'][1])])
    entries.append(('rules','baseline',None,next(e for e in old['evaluations'] if e['model']=='rules')))
    out.mkdir();shutil.copy2(template/'q2episode.exe',out/'q2episode.exe');plans=[];evaluations=[];reference=None
    for model,label,source,entry in entries:
        assert sha(entry['plan'])==entry['plan_sha256'];plan=read(entry['plan'])
        assert sha(plan['registry_path'])==plan['registry_sha256']
        conditions=[dict(episode=t['episode'],seeds=t['seeds'],instances=t.get('instances',[])) for t in plan['tasks']]
        if reference is None:reference=conditions
        assert conditions==reference
        folder=out/(model+'-'+label);folder.mkdir();digest=None
        if source:
            seal=read(source/'complete.json');assert sha(source/'weights.json')==seal['weights_sha256']
            assert sha(source/'initialization.pt')==seal['checkpoint_sha256'] and sha(source/'report.json')==seal['report_sha256']
            value=read(source/'weights.json');value['deterministic']=True
            save(folder/'weights.json',value);plan['model_path']=str(folder/'weights.json');digest=sha(folder/'weights.json');plan['model_sha256']=digest
        plan['output_root']=str(folder/'capture')
        for task in plan['tasks']:task['runner_sha256']=sha(task['runner_path'])
        path=folder/'plan.json';save(path,plan);plans.append(str(path))
        evaluations.append({'model':model,'label':label,'plan':str(path),'root':plan['output_root'],'plan_sha256':sha(path),
                            'deterministic_weights_sha256':digest,'source_weights':str(source/'weights.json') if source else None,
                            'source_weights_sha256':sha(source/'weights.json') if source else None})
    assert evaluations[0]['deterministic_weights_sha256']==evaluations[2]['deterministic_weights_sha256']
    protocol={'version':'combat_distillation_paired_validation_v1','comparison_stage':'cuda_distillation',
        'evaluations':evaluations,'families':old['families'],'episodes_per_model':80,'total_episodes':400,
        'slots':16,'timescale':2,'seed_offset':old['seed_offset'],'split':'validation','final_test_deferred':True,
        'prior_report_sha256':sha(priors/'report.json'),'cuda_audit_sha256':sha(priors/'cuda-audit.json'),
        'template_protocol_sha256':sha(template/'protocol.json'),
        'scope':'Before=exact FireBC identity control; after=balanced Attention128 CUDA distillation, seeds 20261007/08. No PPO update. Same 20 families/80 conditions; reused validation, not final test.'}
    save(out/'protocol.json',protocol);save(out/'plans.json',plans);save(out/'progress.json',{'stage':'plans_prepared'})
    print(json.dumps({'episodes':400,'root':str(out),'conditions_unchanged':True}))

if __name__=='__main__':main()
