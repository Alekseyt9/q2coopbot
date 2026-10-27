"""Describe paired continuation uncertainty; never emit actor training targets."""
import argparse
import math
from pathlib import Path
from statistics import mean,stdev

from process_combat_architecture_pool import read,save,sha


def summarize(roots,out):
    assert not out.exists()
    cases={};pins={str(Path(__file__).resolve()):sha(__file__)}
    for root in roots:
        report=root/'results.json';complete=root/'complete.json'
        completion=read(complete);result=read(report)
        assert completion['state']==result['state']=='complete'
        assert completion['results_sha256']==sha(report)
        pins[str(report)]=sha(report);pins[str(complete)]=sha(complete)
        for case in result['cases']:
            ctx=case['context'];cid=ctx['record_id']
            if cid not in cases:cases[cid]=dict(context=ctx,pairs={})
            else:assert cases[cid]['context']==ctx
            for difference in case['differences']:
                future=difference['future']
                arms=[o for o in case['outcomes'] if o['future']==future and o['mode']!='baseline']
                assert len(arms)==2 and {v['mode'] for v in arms}=={'choice0','choice1'}
                zero=next(v for v in arms if v['mode']=='choice0');one=next(v for v in arms if v['mode']=='choice1')
                delta=one['utility']-zero['utility'];assert math.isfinite(delta) and delta==difference['delta']
                eligible=zero['ownership']['rules_combat_rows']==one['ownership']['rules_combat_rows']==0
                assert eligible==difference['eligible']
                pair=dict(delta=delta,eligible=eligible,
                          choice0={k:zero[k] for k in ('utility','goal','death','kills','dealt','received','ownership')},
                          choice1={k:one[k] for k in ('utility','goal','death','kills','dealt','received','ownership')})
                if future in cases[cid]['pairs']:assert cases[cid]['pairs'][future]==pair,'Reused original changed'
                else:cases[cid]['pairs'][future]=pair
    expected={'original','forkA','forkB','forkC','forkD','forkE','forkF'}
    assert len(cases)==8
    results=[]
    for case in cases.values():
        pairs=case['pairs'];assert set(pairs)==expected
        deltas=[p['delta'] for p in pairs.values()]
        autonomous=[p['delta'] for p in pairs.values() if p['eligible']]
        counts=dict(choice0=sum(d<=-.05 for d in autonomous),tie=sum(abs(d)<.05 for d in autonomous),
                    choice1=sum(d>=.05 for d in autonomous),hybrid=len(deltas)-len(autonomous))
        all_autonomous=counts['hybrid']==0
        # Never silently remove hybrid streams from the primary paired mean.
        results.append(dict(context=case['context'],pairs=pairs,counts=counts,
            observed_autonomous_preference_flip=counts['choice0']>0 and counts['choice1']>0,
            all_streams_autonomous=all_autonomous,
            all_streams_mean_delta=mean(deltas),all_streams_sample_stddev=stdev(deltas),
            autonomous_mean_delta=mean(deltas) if all_autonomous else None,
            minimum_delta=min(deltas),maximum_delta=max(deltas)))
    for path,digest in pins.items():assert sha(path)==digest
    save(out,dict(version='combat_wide_future_uncertainty_summary_v1',state='complete',cases=results,
        contexts=8,unique_stream_pairs=56,native_cases=168,cuda_cases=168,
        preference_flip_contexts=sum(c['observed_autonomous_preference_flip'] for c in results),
        inputs=pins,scope='Eight reused TRAIN worlds,seven fixed suffix streams each. Descriptive correlated within-world contrasts; not independent world replication,precise Q,quality confirmation or training labels.',
        training_targets_emitted=False,policy_training_performed=False,promotion=False))
    for case in results:print(case['context']['map'],case['counts'],case['autonomous_mean_delta'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,action='append',required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();summarize(a.root,a.out)
