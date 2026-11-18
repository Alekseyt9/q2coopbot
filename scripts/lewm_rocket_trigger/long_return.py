"""Compare 40- and 80-frame return targets using the same worlds, actors and modes."""
import argparse
from pathlib import Path
import shutil

import numpy as np

from experiment import BANK, blast_hp, collect, read, save
from mode_fit import MODES, feature_matrix, fit_layer
from temporal import utility


PRIOR = BANK/'lewm-mode-choice-v1'


def prepare(out,workers):
    plan=read(PRIOR/'fit-plan.json')
    if (out/'fit-plan.json').exists() and read(out/'fit-plan.json') != plan:
        raise ValueError('Fit worlds have changed')
    save(out/'fit-plan.json',plan)
    shutil.copyfile(PRIOR/'live/rocket-client.exe',out/'rocket-client.exe')
    for mode in MODES:
        shutil.copyfile(PRIOR/f'{mode}.json',out/f'{mode}.json')
    collect(out,plan['roots'],MODES,workers,fixture_rocket=False,port=29300,followup_frames=80)


def fit(out):
    rows=read(out/'fit-plan.json')['roots']
    maps=np.array([r['map'] for r in rows])
    x=feature_matrix(rows)
    runs=[[read(out/f"{r['seed']}-{k}.json") for k in MODES] for r in rows]
    if not all(t['followup_frames']==80 for rr in runs for t in rr):
        raise ValueError('Incomplete long labels')
    y=np.array([[utility(t['longer']) for t in rr] for rr in runs])
    blast=np.array([[blast_hp(t['longer']) for t in rr] for rr in runs])
    old=np.array([[utility(read(PRIOR/f"{r['seed']}-{m}.json")) for m in MODES] for r in rows])
    now=np.array([[utility(t) for t in rr] for rr in runs])
    # Record native repeatability as distinct from the change of training horizon.
    drift=np.argwhere(old!=now)
    variants={'equipment':(np.arange(64,76),1.),'latent':(np.arange(64),1.),
              'both':(np.arange(76),1.),'both16':(np.arange(76),16.),'both256':(np.arange(76),256.)}
    records=[]
    for cost in (0.,1.,3.):
        target=y-cost*blast
        for group,(columns,lp) in variants.items():
            for penalty in (128.,32.,8.,1.):
                choices=np.zeros(len(rows),dtype=int)
                for level in sorted(set(maps)):
                    test=maps==level
                    w,b=fit_layer(x[~test],target[~test],columns,penalty,lp)
                    choices[test]=(x[test]@w.T+b).argmax(1)
                ix=(np.arange(len(rows)),choices)
                records.append(dict(group=group,penalty=penalty,cost=cost,score=int(y[ix].sum()),
                                    own_blast=int(blast[ix].sum()),counts=np.bincount(choices,minlength=3).tolist()))
    best=max((r for r in records if r['cost']==0),key=lambda r:r['score'])
    safe=[r for r in records if r['own_blast']<=blast[:,0].sum()]
    constrained=max(safe,key=lambda r:r['score']) if safe else min(records,key=lambda r:(r['own_blast'],-r['score']))
    for name,record in [('long20',best),('safe20',constrained)]:
        columns,lp=variants[record['group']]
        w,b=fit_layer(x,y-record['cost']*blast,columns,record['penalty'],lp)
        model=read(PRIOR/'select20.json')
        model['Scheduler']['Readout']=dict(weight=w.tolist(),bias=b.tolist())
        save(out/f'{name}.json',model)
    save(out/'long-fit.json',dict(rows=len(rows),primary='safe20',best=best,constrained=constrained,
         training_constraint_met=bool(safe),fixed_rewards=y.sum(0).tolist(),fixed_own_blast=blast.sum(0).tolist(),
         different_first40=[dict(seed=rows[i]['seed'],mode=MODES[j],previous=int(old[i,j]),now=int(now[i,j])) for i,j in drift],
         all_scores=records))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output',type=Path)
    ap.add_argument('--workers',type=int,default=10)
    ap.add_argument('--fit-only',action='store_true')
    args=ap.parse_args()
    out=args.output.resolve()
    if not args.fit_only:prepare(out,args.workers)
    fit(out)
