"""Native evaluation of contextual policies on held-back initial Quake II worlds."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np

from experiment import blast_hp, collect, read, root_obs, save
from mode_fit import MODES, feature_matrix
from temporal import utility


def stage(out, names):
    for name in names:
        src=out/f'{name}.json'
        dst=out/'live'/src.name
        if dst.exists():
            if hashlib.sha256(src.read_bytes()).digest()!=hashlib.sha256(dst.read_bytes()).digest():
                raise ValueError(f'Existing replays belong to different weights: {name}')
        else:
            tmp=dst.with_suffix('.tmp')
            shutil.copyfile(src,tmp)
            tmp.replace(dst)


def timed_damage(run,frames):
    bins = [0,0]
    for line in run['events']:
        e=dict(re.findall(r'(\w+)=([^ ]+)',line))
        if e['attacker_class']=='player' and e['target_class'].startswith('monster_'):
            lost=max(0,int(e['health_before']))-max(0,int(e['health_after']))
            bins[int(int(e['server_frame'])>1+frames//2)]+=lost
    if sum(bins)!=run['monster_hp']:
        raise ValueError('Event totals disagree')
    return bins


def roots(out):
    used={r['seed'] for r in read(out/'fit-plan.json')['roots']}
    cases,excluded=[],{}
    for case in read(out/'capture-plan.json'):
        seed=case['seed']
        assert seed not in used
        paths=[out/'recovery'/f'source-{seed}',out/f'source-retry-{seed}',out/f'source-{seed}']
        src=next((p for p in paths if (p/'report.json').exists() and
                  (read(p/'report.json').get('combat_checkpoint') or {}).get('accepted')),
                 next((p for p in paths if p.exists()),paths[-1]))
        mf=src/'combat-checkpoint/manifest.json'
        report=read(src/'report.json') if (src/'report.json').exists() else {}
        if not mf.exists() or not (report.get('combat_checkpoint') or {}).get('accepted'):
            excluded[seed]=report.get('reason','missing receipt')
            continue
        frame=read(mf)['barrier']['frame']
        obs=root_obs(src/'bot.jsonl',frame)
        if obs.get('owner')!='provider' or obs.get('features') is None:
            excluded[seed]='root not owned by policy'
            continue
        if case['rocket']:assert obs['features'][852]==1
        fixture=read(out/f'fixture-{seed}.json')
        cases.append(dict(case,frame=frame,source=str(src),split=fixture['split'],
                     manifest_hash=hashlib.sha256(mf.read_bytes()).hexdigest(),
                     feature_hash=hashlib.sha256(json.dumps(obs['features']).encode()).hexdigest()))
    return cases,excluded


def report(out, rows, names, frames=40, baseline='fixed8'):
    live=out/'live'
    x=feature_matrix(rows)
    expected={}
    for name in names:
        if name not in MODES:
            layer=read(live/f'{name}.json')['Scheduler']['Readout']
            expected[name]=(x@np.array(layer['weight']).T+layer['bias']).argmax(1)
    data=[]
    for i,r in enumerate(rows):
        runs={}
        for name in names:
            run=read(live/f"{r['seed']}-{name}.json")
            objective_verified=run.get('objective_verified',False)
            if frames==run.get('final_frames'):
                run=run['final']
            elif frames!=40:
                assert run['followup_frames']==frames
                run=run['longer']
            early,late=timed_damage(run,frames)
            modes=[int(line.split()[-1].split('=')[1]) for line in run['modes']]
            if name in expected and (not modes or modes[0]!=expected[name][i]):
                raise ValueError(f"Go mode differs from Python inference on {r['seed']}")
            runs[name]=dict(score=utility(run),early=early,late=late,self_hp=run['self_hp'],own_blast=blast_hp(run),other=run['other_hp'],
                            alive=int(run['alive']),modes=modes)
            if objective_verified:
                runs[name]['clear']=int(run['first_life_clear_frame'] is not None)
        data.append(dict(seed=r['seed'],map=r['map'],rocket=r['rocket'],policies=runs))
    groups={}
    for group in ('mixed','rocket','all'):
        selected=[r for r in data if group=='all' or r['rocket']==(group=='rocket')]
        stats={}
        for name in names:
            cases=[r['policies'][name] for r in selected]
            stats[name]={k:sum(c[k] for c in cases) for k in ('score','early','late','self_hp','own_blast','other','alive')}
            if cases and all('clear' in c for c in cases):
                stats[name]['clear']=sum(c['clear'] for c in cases)
            delta=[c['score']-r['policies'][baseline]['score'] for c,r in zip(cases,selected)]
            stats[name].update(wins=sum(d>0 for d in delta),ties=delta.count(0),losses=sum(d<0 for d in delta),
                 changed=sum(len(set(c['modes']))>1 for c in cases),
                 initial_modes=dict(Counter(MODES[c['modes'][0]] for c in cases if c['modes'])))
        groups[group]=stats
    return dict(groups=groups,worlds=data)


def run(out, workers, ready_only):
    cases,excluded=roots(out)
    live=out/'live'
    names=[*MODES,'select40','select20','comparison20']
    if not ready_only:
        save(out/'evaluation-plan.json',dict(roots=cases,excluded=excluded,policies=names,primary='select20',
             training_window=40,extended_window=80))
    stage(out,names)
    collect(live,cases,names,workers,fixture_rocket=False,port=28740,audit_duration=True,followup_frames=80)
    if not ready_only:
        save(out/'evaluation.json',dict(first40=report(out,cases,names),first80=report(out,cases,names,80)))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output',type=Path)
    ap.add_argument('--workers',type=int,default=8)
    ap.add_argument('--collect-ready',action='store_true')
    args=ap.parse_args()
    run(args.output.resolve(),args.workers,args.collect_ready)
