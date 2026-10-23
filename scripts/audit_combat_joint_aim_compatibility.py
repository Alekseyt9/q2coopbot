"""Compare native paired actions/latent events, with dedicated CUDA receipts."""
import argparse
import collections
from pathlib import Path
from evaluate_registered_combat_models import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('evaluation','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists()
    root=a.evaluation.resolve();closed=read(root/'complete.json')
    assert closed['state']=='complete' and closed['native_cases']==8 and closed['cuda_cases']==4
    pins={};pairs=collections.defaultdict(dict)
    def pin(path):pins[str(path.resolve())]=sha(path)
    pin(root/'complete.json');pin(root/'pool/report.json')
    pool=read(root/'pool/report.json');assert pool['state']=='complete' and pool['usable_captures']==8
    for job in pool['jobs']:
        assert not job['error'] and job['mode']=='learned'
        report=Path(job['root'])/'report.json';pin(report);cap=read(report)
        trace=Path(cap['results'][0]['root'])/'bot.jsonl';pin(trace)
        rows=[]
        with trace.open(encoding='utf-8-sig') as stream:
            import json
            for line in stream:
                row=json.loads(line);c=row.get('combat_policy') or {};s=c.get('selection',{})
                if s.get('owner')=='provider' and s.get('sample'):
                    assert c['observation']['identity']['life']==1
                    rows.append((c,s['sample'],bool(s.get('inference_budget_exceeded'))))
        assert job['plan_index'] not in pairs[job['seed']]
        pairs[job['seed']][job['plan_index']]=rows
    results=[];choices=collections.Counter();events=0;steps=0
    for seed,pair in sorted(pairs.items()):
        before,after=pair[0],pair[1];assert len(before)==len(after)>0
        budgets=[0,0];max_error=0
        for (c,s,budget),(nc,ns,nbudget) in zip(before,after):
            assert c['observation']['identity']==nc['observation']['identity']
            for key in ('proposed_command','applied_command'):assert c[key]==nc[key],(seed,key)
            for key in ('attack','target','aim_mode','vertical','weapon'):assert s.get(key,0)==ns.get(key,0),(seed,key)
            assert s['latent']==ns['latent']
            old,new=s.get('aim_exploration'),ns.get('aim_candidate')
            assert bool(old)==bool(new)
            if old:
                assert old['candidate']==new['candidate'] and old['candidates']==new['candidates']==25
                source=s['latent'][:2]+old['source_latent'];other=new['source_latent']
                choices[new['candidate']]+=1;events+=1
            else:source=s['latent'];other=ns['latent']
            error=max(abs(x-y) for x,y in zip(source,other));max_error=max(max_error,error)
            assert source==other,'Source RNG/latent differs'
            budgets[0]+=int(budget);budgets[1]+=int(nbudget)
        steps+=len(before)
        results.append(dict(seed=seed,provider_steps=len(before),raw_latent_max_error=max_error,
                            native_commands_exact=True,source_categories_exact=True,inference_budget_flags=budgets))
    cuda=[]
    for path in sorted((root/'processing-joint').glob('case-*/cuda/cuda-verification.json')):
        pin(path);v=read(path)
        assert v['state']=='passed' and v['device']=='cuda' and v['action_distribution']=='combat_joint_aim_candidates_v1'
        assert v['original_latent_transform_verified'] and v['frame_hash_candidate_assignments_verified']
        cuda.append(v)
    assert len(cuda)==4 and events>0
    a.out.mkdir(parents=True)
    report=dict(state='complete',version='combat_joint_aim_native_compatibility_v1',paired_cases=len(results),native_cases=8,
        provider_steps_per_variant=steps,candidate_events_per_variant=events,candidate_counts=dict(choices),pairs=results,
        cuda_cases=len(cuda),cuda_transitions=sum(v['rows'] for v in cuda),
        joint_log_probability_max_error=max(v['log_probability_max_error'] for v in cuda),
        candidate_log_probability_max_error=max(v['candidate_log_probability_max_error'] for v in cuda),
        source_latent_and_native_commands_exact=True,inputs=pins,script_sha256=sha(__file__),training_performed=False,promotion=False,
        scope='Finite native compatibility of initial head against exploratory control only. Original actor RNG/event and commands match. Full joint CUDA likelihood verified. No quality gain, TRAIN acceptance or learned nonuniform runtime claim.')
    for path,digest in pins.items():assert sha(Path(path))==digest
    save(a.out/'report.json',report);save(a.out/'complete.json',dict(state='complete',report_sha256=sha(a.out/'report.json')))
    print({k:v for k,v in report.items() if k not in ('inputs','pairs','scope')})


if __name__=='__main__':main()
