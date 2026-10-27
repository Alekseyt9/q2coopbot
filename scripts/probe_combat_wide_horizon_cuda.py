"""Reuse verified continuations to describe n-step critic contrast uncertainty."""
import argparse
import json
from pathlib import Path

from ppo_combat import torch,training_devices
from process_combat_architecture_pool import read,save,sha


def estimate(rows,frame,horizon,gamma=.99):
    active=[r for r in rows if r['frame']>=frame and (horizon is None or r['frame']<frame+horizon)]
    if not active:return None
    frames=[r['frame'] for r in active]
    if horizon is not None and frames!=list(range(frame,active[-1]['next_frame'])):return None
    assert all(r['next_frame']==r['frame']+1 for r in active)
    terminal=active[-1]['bootstrap_zero']
    if horizon is not None and not terminal and active[-1]['next_frame']!=frame+horizon:return None
    offsets=torch.tensor([r['frame']-frame for r in active],device='cuda',dtype=torch.float64)
    rewards=torch.tensor([r['reward'] for r in active],device='cuda',dtype=torch.float64)
    value=(torch.pow(gamma,offsets)*rewards).sum()
    if horizon is not None and not terminal:
        value=value+gamma**(active[-1]['next_frame']-frame)*active[-1]['next_value']
    assert bool(torch.isfinite(value))
    return float(value.item())


def self_test():
    rows=[dict(frame=10,next_frame=11,reward=1.,next_value=10.,bootstrap_zero=False),
          dict(frame=11,next_frame=12,reward=2.,next_value=20.,bootstrap_zero=False)]
    assert estimate(rows,10,1,.5)==6.
    assert estimate(rows,10,2,.5)==7.
    assert estimate(rows,10,None,.5)==2.
    assert estimate(rows,10,4,.5) is None
    rows[-1]['bootstrap_zero']=True
    assert estimate(rows,10,2,.5)==estimate(rows,10,4,.5)==2.
    rows[-1].update(frame=12,next_frame=13)
    assert estimate(rows,10,3,.5) is None
    assert estimate(rows,10,None,.5)==1.5


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,action='append',required=True)
    ap.add_argument('--summary',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists()
    training_devices();torch.set_num_threads(2);self_test()
    pins={str(Path(__file__).resolve()):sha(__file__),str(a.summary.resolve()):sha(a.summary)}
    summary=read(a.summary);assert summary['state']=='complete' and summary['unique_stream_pairs']==56
    for p,h in summary['inputs'].items():assert sha(p)==h;pins[p]=h
    contexts={c['context']['id']:c for c in summary['cases']}
    entries={}
    for root in a.root:
        result=read(root/'results.json');completion=read(root/'complete.json')
        assert result['state']==completion['state']=='complete' and sha(root/'results.json')==completion['results_sha256']
        allowed={(c['context']['id'],o['future'],o['mode']):o for c in result['cases'] for o in c['outcomes']}
        path=root/'exports.json';pins[str(path.resolve())]=sha(path)
        for entry in read(path)['entries']:
            key=(entry['context_id'],entry['future'],entry['mode'])
            assert entry['data']==allowed[key]['data']
            if key in entries:assert entry['data']==entries[key]['data']
            else:entries[key]=entry
    assert len(entries)==168
    branches={};horizons=(1,4,8,16,32,64,None)
    for key,entry in entries.items():
        if key[-1]=='baseline':continue
        directory=Path(entry['data']);meta=read(directory/'report.json');proof=read(directory/'cuda-verification.json')
        assert proof['state']=='passed' and proof['device']=='cuda' and proof['model_sha256']==entry['model_sha256']
        for path,h in ((Path(entry['model']),entry['model_sha256']),
                       (directory/'rollout.jsonl',meta['rollout_sha256']),
                       (directory/'cuda-verification.json',meta['cuda_verification_sha256'])):
            assert sha(path)==h;pins[str(path)]=h
        pins[str(directory/'report.json')]=sha(directory/'report.json')
        with (directory/'rollout.jsonl').open(encoding='utf-8-sig') as stream:rows=list(map(json.loads,stream))
        frame=contexts[key[0]]['context']['frame']
        branches[key]={str(h):estimate(rows,frame,h) for h in horizons}
    cases=[]
    for cid,source in contexts.items():
        comparisons=[]
        for horizon in horizons:
            deltas={};missing=[]
            for future,pair in source['pairs'].items():
                left=branches[(cid,future,'choice0')][str(horizon)];right=branches[(cid,future,'choice1')][str(horizon)]
                if left is None or right is None:missing.append(future);continue
                delta=right-left;deltas[future]=delta
                if horizon is None:assert abs(delta-pair['delta'])<1e-6
            values=torch.tensor(list(deltas.values()),device='cuda',dtype=torch.float64)
            comparisons.append(dict(horizon_frames=horizon,paired_streams=len(deltas),missing_streams=missing,deltas=deltas,
                mean_delta=float(values.mean().item()) if len(deltas)==7 else None,
                sample_stddev=float(values.std(correction=1).item()) if len(deltas)==7 else None,
                choice0=sum(v<=-.05 for v in deltas.values()),tie=sum(abs(v)<.05 for v in deltas.values()),choice1=sum(v>=.05 for v in deltas.values())))
        cases.append(dict(context=source['context'],full_life_hybrid_streams=source['counts']['hybrid'],comparisons=comparisons))
    for p,h in pins.items():assert sha(p)==h
    a.out.mkdir(parents=True)
    save(a.out/'report.json',dict(version='combat_wide_horizon_uncertainty_v1',state='complete',device='cuda',
        self_test_passed=True,cases=cases,inputs=pins,training_performed=False,promotion=False,
        scope='Descriptive seven-stream n-step reward plus source critic bootstrap; horizon=None is finite observed reward only. Lower variance is not unbiased Q or better combat. Missing short windows retained. Finite observed sum retains all admitted rows including ownership gaps, matching the original target; full-life hybrid streams flagged. One-step stream variance is near zero by construction because RNG forks start at root+1. No training labels emitted.'))
    print(a.out/'report.json',flush=True)
    for c in cases:print(c['context']['map'],[(r['horizon_frames'],r['mean_delta'],r['sample_stddev']) for r in c['comparisons']],flush=True)


if __name__=='__main__':main()
