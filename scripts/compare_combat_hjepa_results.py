"""Paired native outcomes and site-cluster bootstrap for H-JEPA comparisons."""
import argparse
import collections
import json
import pathlib
import random


def interval(site_differences, samples=10000):
    rng=random.Random(251115)
    n=len(site_differences)
    means=sorted(sum(rng.choices(site_differences,k=n))/n for _ in range(samples))
    return [means[int(.025*(samples-1))],means[int(.975*(samples-1))]]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--matrix',type=pathlib.Path,required=True)
    ap.add_argument('--reference',default='motor-bc')
    ap.add_argument('--out',type=pathlib.Path,required=True)
    a=ap.parse_args()
    plan=json.loads((a.matrix/'plan.json').read_text())
    variants=collections.defaultdict(dict)
    for path in a.matrix.glob('*/case-*/report.json'):
        report=json.loads(path.read_text())
        if not report['provenance_valid'] or not report['capture_complete']:raise ValueError('unverified capture')
        task=plan['tasks'][int(path.parent.name.split('-')[1])]
        name=path.parent.parent.name
        for r in report['results']:
            if not r['capture_valid'] or not r['dataset']['command_proof']['accepted']:raise ValueError('unverified dispatch')
            if r['seed'] not in task['seeds'] or r['seed'] in variants[name]:raise ValueError('seed binding/duplicate')
            native,guarded=0,0
            steps=pathlib.Path(r['root'])/'dataset/steps.jsonl'
            for line in steps.read_text().splitlines():
                s=json.loads(line)
                if s['owner']=='provider' and s.get('native_step') and s['observation']['health']>0:
                    native+=1;guarded+=bool(s.get('interventions'))
            variants[name][r['seed']]=dict(site=task['episode']['id'],win=int(bool(r['goal_stop'])),death=int(bool(r['death_stop'])),
                received_hp=r['first_life']['damage']['received_health_damage'],guard_events=r['guard_interventions'],
                guarded_fraction=guarded/native if native else 0.,native_frames=native,guarded_frames=guarded)
    reference=variants[a.reference]
    if not reference:raise ValueError('missing reference')
    comparisons={}
    for name,rows in variants.items():
        if set(rows)!=set(reference):raise ValueError('unpaired/incomplete cohort')
        if name==a.reference:continue
        metrics={}
        for key in ['win','death','received_hp','guard_events','guarded_fraction']:
            by_site=collections.defaultdict(list)
            for seed,r in rows.items():
                if r['site']!=reference[seed]['site']:raise ValueError('site binding mismatch')
                by_site[r['site']].append(r[key]-reference[seed][key])
            if len({len(values) for values in by_site.values()})!=1:raise ValueError('unequal site cluster sizes')
            diffs=[sum(values)/len(values) for values in by_site.values()]
            metrics[key]=dict(mean_paired_difference=sum(diffs)/len(diffs),site_cluster_95pct=interval(diffs),sites=len(diffs))
        comparisons[name]=metrics
    totals={name:dict(episodes=len(rows),native_frames=sum(r['native_frames'] for r in rows.values()),
                      guarded_frames=sum(r['guarded_frames'] for r in rows.values()),
                      provider_guarded_fraction=sum(r['guarded_frames'] for r in rows.values())/max(1,sum(r['native_frames'] for r in rows.values())))
            for name,rows in variants.items()}
    result=dict(version='hjepa_paired_comparison_v1',reference=a.reference,comparisons=comparisons,totals=totals,
        scope='Candidate minus reference; lower damage/guard/death is better. Approximate percentile bootstrap across paired sites,10000 resamples. Narrow fixed-MG evidence; not a general superiority claim.')
    a.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(reference=a.reference,totals=totals,comparisons=comparisons),indent=2))


if __name__=='__main__':main()
