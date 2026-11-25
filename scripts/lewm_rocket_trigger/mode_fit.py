"""Fit a mode-value readout to paired native returns. No tactical action labels."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from build import REPO
from experiment import BANK, V1, V2, collect, read, root_obs, save
from temporal import utility


MODES = ('fixed8','fixed12','local')


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(out):
    prior = BANK/'lewm-duration-v1'
    local = read(prior/'local-selection.json')['best']
    sources = [r for r in read(V1/'plan-verified.json')['roots'] if r['split']=='train']
    for f in ('new-roots.json','rocket-validation/new-roots.json'):
        sources.extend(read(prior/f)['roots'])
    assert len(sources)==len({r['seed'] for r in sources})==76
    shutil.copyfile(prior/'rocket-client.exe', out/'rocket-client.exe')
    for name, old in zip(MODES,('fixed8','fixed12',local)):
        shutil.copyfile(prior/f'{old}.json',out/f'{name}.json')
        for r in sources:
            seed=r['seed']
            previous=prior/f'{seed}-{old}.json'
            if previous.exists():
                shutil.copyfile(previous,out/f'{seed}-{name}.json')
    for r in sources:
        x=root_obs(Path(r['source'])/'bot.jsonl',r['frame'])
        assert x['owner']=='provider'
        r['feature_hash']=hashlib.sha256(json.dumps(x['features']).encode()).hexdigest()
    save(out/'fit-plan.json', dict(roots=sources,mode_names=MODES,periods=[40,20],
         former_validation_in_fit=30,selection='leave one map out native return'))
    collect(out,sources,MODES,8,fixture_rocket=False,port=52140)


def feature_matrix(rows):
    raw=np.array([root_obs(Path(r['source'])/'bot.jsonl',r['frame'])['features'] for r in rows])
    return encode_features(raw)


def encode_features(raw):
    import sys
    import torch
    sys.path.insert(0,str(REPO/'scripts'))
    from hjepa_lewm_weapon_damage import load
    with torch.inference_mode():
        z=load(V2/'damage.pt').world.encode(torch.tensor(raw,dtype=torch.float32)).numpy()
    return np.column_stack((z,raw[:,845:857])).astype(float)


def fit_layer(x, y, columns, penalty, latent_penalty=1.):
    use=x[:,columns]
    mean=use.mean(0)
    std=use.std(0).clip(.05)
    design=(use-mean)/std
    centered=y-y.mean(1,keepdims=True)
    bias=centered.mean(0)
    ridge=penalty*np.diag(np.where(columns<64,latent_penalty,1.))
    coef=np.linalg.solve(design.T@design+ridge,design.T@(centered-bias)).T/std
    weight=np.zeros((y.shape[1],x.shape[1]))
    weight[:,columns]=coef
    return weight, bias-coef@mean


def train(out):
    rows=read(out/'fit-plan.json')['roots']
    x=feature_matrix(rows)
    y=np.array([[utility(read(out/f"{r['seed']}-{m}.json")) for m in MODES] for r in rows])
    maps=np.array([r['map'] for r in rows])
    variants={'equipment':(np.arange(64,76),1.),'latent':(np.arange(64),1.),
              'both':(np.arange(76),1.),'both16':(np.arange(76),16.),'both256':(np.arange(76),256.)}
    records=[]
    for group,(columns,lpen) in variants.items():
        for penalty in (128.,32.,8.,1.):
            chosen=np.zeros(len(rows),dtype=int)
            for level in sorted(set(maps)):
                test=maps==level
                w,b=fit_layer(x[~test],y[~test],columns,penalty,lpen)
                chosen[test]=(x[test]@w.T+b).argmax(1)
            records.append(dict(group=group,penalty=penalty,utility=int(y[np.arange(len(y)),chosen].sum()),
                                mode_counts=np.bincount(chosen,minlength=len(MODES)).tolist()))
    best=max(records,key=lambda t:t['utility'])
    def full_readout(record):
        columns,lpen=variants[record['group']]
        w,b=fit_layer(x,y,columns,record['penalty'],lpen)
        return dict(weight=w.tolist(),bias=b.tolist())
    policy=read(out/'fixed8.json')
    heads=[]
    for name in MODES:
        cp=read(out/f'{name}.json')
        if any(cp[k]!=policy[k] for k in ('Encoder','Projection','Controller')):
            raise ValueError('Incompatible component encoders or motor policies')
        heads.append(cp.get('Termination',dict(weight=[[0,0,0,0,0,1]],bias=[-1.])))
    alternative=max((r for r in records if (r['group']=='equipment')!=(best['group']=='equipment')),
                    key=lambda r:r['utility'])
    for name,period,record in [('select40',40,best),('select20',20,best),('comparison20',20,alternative)]:
        save(out/f'{name}.json',dict(policy, Scheduler=dict(Readout=full_readout(record),Heads=heads,Frames=period)))
    save(out/'fit.json',dict(rows=len(rows),best=best,alternative=alternative,variants=records,
                             constant_rewards=dict(zip(MODES,y.sum(0).tolist())),
                             checkpoints={str(k):file_hash(out/f'select{k}.json') for k in (20,40)}))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output',type=Path)
    ap.add_argument('--fit-only',action='store_true')
    args=ap.parse_args()
    out=args.output.resolve()
    if not args.fit_only:
        prepare(out)
    train(out)
