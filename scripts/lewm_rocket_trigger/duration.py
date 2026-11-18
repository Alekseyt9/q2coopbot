"""Fit a state-dependent temporal termination head from native Quake II returns."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from experiment import BANK, V1, V2, collect, read, root_obs, save, totals
from temporal import build_temporal, utility


def cohorts():
    trigger = BANK/'lewm-rocket-trigger-v1'
    original = read(trigger/'plan.json')['roots']
    train = [r for r in original if r['split']=='train']
    diagnostic = [r for r in original if r['split']!='train'] + read(trigger/'external-plan.json')
    known = {r['seed'] for r in train+diagnostic}
    extra = []
    pool = [r for r in read(V1/'plan-verified.json')['roots'] if r['split']=='test'] + read(V2/'fresh-plan.json')
    for r in pool:
        if r['seed'] in known:
            continue
        obs = root_obs(Path(r['source'])/'bot.jsonl', r['frame'])
        assert obs['owner']=='provider'
        extra.append(dict(r, split='additional', feature_hash=hashlib.sha256(json.dumps(obs['features']).encode()).hexdigest()))
        known.add(r['seed'])
    return train, diagnostic, extra


def paired(out, roots, names):
    runs = {k:[read(out/f"{r['seed']}-{k}.json") for r in roots] for k in names}
    stats = totals(out, roots, names)
    for name, rows in runs.items():
        durations = Counter()
        for row in rows:
            for line in row['refreshes']:
                if line.endswith('continuous=true'):
                    durations[int(line.split()[2].split('=')[1])] += 1
        stats[name].update(utility=utility(stats[name]), completed_duration_histogram=dict(durations))
    comparisons = {}
    for reference in names[:-1]:
        gaps = [utility(a)-utility(b) for a,b in zip(runs[names[-1]],runs[reference])]
        comparisons[reference] = dict(wins=sum(g>0 for g in gaps), ties=gaps.count(0),
                                      losses=sum(g<0 for g in gaps), gaps=dict(zip([r['seed'] for r in roots], gaps)))
    return dict(stats=stats, paired=comparisons)


def evaluate(out, workers):
    _, diagnostic, extra = cohorts()
    selected = read(out/'selection.json')
    candidate, best_fixed = selected['best_adaptive'], selected['best_fixed']
    names = list(dict.fromkeys(['fixed8',best_fixed,candidate]))
    collect(out,diagnostic+extra,names,workers,fixture_rocket=False,port=49700,audit_duration=True)
    save(out/'evaluation.json',dict(candidate=candidate,best_fixed=best_fixed,
         diagnostic=paired(out,diagnostic,names), additional=paired(out,extra,names)))


def run(out, workers):
    out.mkdir(parents=True, exist_ok=False)
    train, diagnostic, extra = cohorts()
    parent = BANK/'lewm-temporal-reward-v1/learned.json'
    policy = read(parent)
    save(out/'protocol.json',dict(train=[r['seed'] for r in train], diagnostic=[r['seed'] for r in diagnostic],
         additional=[r['seed'] for r in extra], parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
         seed=869, fixed_periods=[4,8,12], scales=[.65,.35], directions_per_round=2,
         criterion='max train sum of first-life monster HP less all player HP lost in 40 frames'))
    build_temporal(out)
    def write(name, vec):
        save(out/f'{name}.json',dict(policy, Termination=dict(weight=[vec[:-1].tolist()],bias=[float(vec[-1])])))
    def play(group, names):
        collect(out,group,names,workers,fixture_rocket=False,port=49700,audit_duration=True)
    shutil.copyfile(parent,out/'fixed8.json')
    fixed = ['fixed8','fixed4','fixed12']
    # The age input is elapsed frames / 8, also used by the adaptive head.
    initial = np.array([0.,0.,0.,0.,0.,1.,-1.])
    for period in (4,12):
        vec = initial.copy()
        vec[-1] = -period/8
        write(f'fixed{period}', vec)
    play(train,fixed)
    history = totals(out,train,fixed)
    best_fixed = max(fixed,key=lambda k:utility(history[k]))
    center = initial.copy()
    center[-1] = -int(best_fixed[5:])/8
    incumbent = best_fixed
    adaptive = []
    rng = np.random.default_rng(869)
    for rnd,scale in enumerate((.65,.35)):
        params = {incumbent:center}
        for j in range(2):
            noise = rng.standard_normal(7)
            for sign in (-1,1):
                name = f'adaptive-{rnd}-{j}-{sign}'
                params[name] = center + sign*scale*noise
                write(name,params[name])
                adaptive.append(name)
        new = list(params)[1:]
        play(train,new)
        history.update(totals(out,train,new))
        incumbent = max(params,key=lambda k:utility(history[k]))
        center = params[incumbent]
        save(out/f'round-{rnd}.json', dict(best=incumbent, scores={k:utility(history[k]) for k in params}))
    candidate = max(adaptive,key=lambda k:utility(history[k]))
    save(out/'selection.json',dict(best_fixed=best_fixed,best_adaptive=candidate,incumbent=incumbent,
                                  train_scores={k:utility(v) for k,v in history.items()}))
    shutil.copyfile(out/f'{candidate}.json',out/'learned.json')
    evaluate(out,workers)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--evaluate-only',action='store_true')
    args=parser.parse_args()
    (evaluate if args.evaluate_only else run)(args.output.resolve(),args.workers)
