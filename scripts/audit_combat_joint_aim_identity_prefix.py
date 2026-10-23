"""Native source compatibility before the first rare learned-head intervention."""
import argparse
import collections
import json
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('evaluation','spec','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();digest=sha(path);assert expected is None or digest==expected,str(path)
        pins[str(path)]=digest;return path
    closed=read(pin(a.evaluation/'complete.json'));assert closed['state']=='complete' and closed['native_cases']==closed['cuda_cases']==8
    spec=read(pin(a.spec,closed['spec_sha256']));assert [v['label'] for v in spec['variants']]==['source','near-identity']
    pool=read(pin(a.evaluation/'pool/report.json',closed['pool_sha256']));assert pool['usable_captures']==8 and pool['source_unchanged']
    pairs=collections.defaultdict(dict)
    for job in pool['jobs']:
        assert not job['error'];capture=read(pin(Path(job['root'])/'report.json'));trace=pin(Path(capture['results'][0]['root'])/'bot.jsonl')
        rows=[]
        with trace.open(encoding='utf-8-sig') as stream:
            for line in stream:
                v=json.loads(line);c=v.get('combat_policy') or {};s=c.get('selection',{})
                if s.get('owner')=='provider' and s.get('sample') and c['observation']['identity']['life']==1:rows.append((c,s['sample']))
        assert rows;pairs[job['seed']][job['plan_index']]=rows
    results=[];total_prefix=eligible=interventions=0;max_lp=0.
    for seed,pair in sorted(pairs.items()):
        source,near=pair[0],pair[1];prefix=0;first=None
        for c,s in near:
            annotation=s.get('aim_candidate')
            if annotation:eligible+=1;interventions+=int(annotation['candidate']!=0)
        for (c,s),(nc,ns) in zip(source,near):
            assert c['observation']['identity']==nc['observation']['identity']
            annotation=ns.get('aim_candidate');z=annotation['source_latent'] if annotation else ns['latent']
            assert s['latent']==z,'Source Gaussian draw changed before an intervention'
            for key in ('attack','vertical','weapon','target','aim_mode'):assert s.get(key,0)==ns.get(key,0)
            lp_error=abs((ns['log_probability']-(annotation['log_probability'] if annotation else 0))-s['log_probability'])
            max_lp=max(max_lp,lp_error);assert lp_error<1e-7
            if annotation and annotation['candidate']!=0:
                first=dict(frame=nc['observation']['identity']['frame'],candidate=annotation['candidate'])
                break
            for key in ('proposed_command','applied_command'):assert c[key]==nc[key]
            assert s['latent']==ns['latent'];prefix+=1
        if first is None:assert len(source)==len(near),'No intervention but trajectories differ'
        total_prefix+=prefix
        results.append(dict(seed=seed,source_steps=len(source),near_steps=len(near),identical_command_prefix=prefix,first_intervention=first))
    cuda=[]
    for label in ('source','near-identity'):
        for path in sorted((a.evaluation/('processing-'+label)).glob('case-*/cuda/cuda-verification.json')):
            v=read(pin(path));assert v['state']=='passed' and v['device']=='cuda';cuda.append(v)
    assert len(cuda)==8
    for path,digest in pins.items():assert sha(Path(path))==digest
    a.out.mkdir(parents=True)
    report=dict(state='complete',version='combat_joint_aim_near_identity_prefix_v1',native_cases=8,cuda_cases=8,paired_conditions=len(results),pairs=results,
        identical_command_prefix_steps=total_prefix,near_eligible_events=eligible,near_nonzero_events=interventions,
        source_lp_excluding_new_category_max_error=max_lp,full_cuda_lp_max_error=max(v['log_probability_max_error'] for v in cuda),
        inputs=pins,script_sha256=sha(__file__),training_performed=False,promotion=False,
        scope='Initial near-identity policy only. Source Gaussian/category draws and commands match until the first correction. Subsequent worlds may diverge; this is not full behavior identity, combat improvement or training acceptance.')
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print({k:v for k,v in report.items() if k not in ('inputs','pairs','scope')})


if __name__=='__main__':main()
