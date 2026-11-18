"""Search close to the eight-frame controller using both TRAIN cohorts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from duration import cohorts, paired
from experiment import V1, collect, read, root_obs, save
from temporal import utility


def key(row):
    return hashlib.sha256(f"local-{row['seed']}".encode()).digest()


def run(out, workers):
    train, diagnostic, archived = cohorts()
    ids = read(out/'selection-protocol.json')['seeds']
    extra = []
    for r in read(V1/'plan-verified.json')['roots']:
        if r['seed'] in ids:
            obs = root_obs(Path(r['source'])/'bot.jsonl',r['frame'])
            extra.append(dict(r,feature_hash=hashlib.sha256(json.dumps(obs['features']).encode()).hexdigest()))
    assert len(extra)==len(ids)
    first = sorted(train,key=key)[:6]+sorted(extra,key=key)[:6]
    remainder = [r for r in train+extra if r not in first]
    rng = np.random.default_rng(913)
    policy = read(out/'fixed8.json')
    candidates = []
    for i,scale in enumerate((.13,.22,.32)):
        delta = rng.standard_normal(6)*scale
        for sign in (-1,1):
            w=delta*sign
            label=f'local-{i}-{sign}'
            # A positive age coefficient fixes the arbitrary scale of the
            # classifier. Other coefficients move the crossing in either direction.
            save(out/f'{label}.json',dict(policy,Termination=dict(weight=[[*w[:5].tolist(),1.]],
                                                            bias=[float(w[5]-.9375)])))
            candidates.append(label)
    save(out/'local-protocol.json',dict(first=[r['seed'] for r in first], rest=[r['seed'] for r in remainder],
        random_seed=913, scales=[.13,.22,.32],retain=3, selection='max minimum mean paired gain on original and supplementary TRAIN groups'))
    def play(rows,names):
        collect(out,rows,names,workers,fixture_rocket=False,port=51220,audit_duration=True)
    def gain(rows,name):
        return sum(utility(read(out/f"{r['seed']}-{name}.json"))-
                   utility(read(out/f"{r['seed']}-fixed8.json")) for r in rows)/len(rows)
    play(first,candidates)
    finalists=sorted(candidates,key=lambda k:gain(first,k),reverse=True)[:3]
    play(remainder,finalists)
    scores={k:[gain(train,k),gain(extra,k)] for k in finalists}
    best=max(finalists,key=lambda k:(min(scores[k]),sum(scores[k])))
    save(out/'local-selection.json',dict(best=best,scores=scores,eligible=min(scores[best])>0))
    shutil.copyfile(out/f'{best}.json',out/'local-duration.json')
    fresh=read(out/'new-roots.json')['roots']
    play(fresh,[best])
    save(out/'local-validation.json',paired(out,fresh,['fixed8',best]))
    play(diagnostic+archived,[best])
    save(out/'local-archive.json',dict(diagnostic=paired(out,diagnostic,['fixed8',best]),
                                      archived=paired(out,archived,['fixed8',best])))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output',type=Path)
    ap.add_argument('--workers',type=int,default=8)
    args=ap.parse_args()
    run(args.output.resolve(),args.workers)
